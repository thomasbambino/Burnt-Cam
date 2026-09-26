"""GPU renderer (moderngl / OpenGL 3.3).

Pass 1 - "unwrap": draws the tracked face mesh with the webcam as its texture
          into a front-facing 512x512 face texture.
Pass 2 - "scene":  ray-marches the signed-distance character, paints the face
          texture onto its head through a soft mask, lights it and draws the
          background.  Rendered at ``supersample`` x resolution.
Pass 3 - "resolve": downsamples to the output size (anti-aliasing).
"""

from __future__ import annotations

import numpy as np
import moderngl

from tracker import FACE_TEX_SIZE, MASK_TEX_SIZE, FaceState

FULLSCREEN_VS = """
#version 330
out vec2 v_uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

UNWRAP_VS = """
#version 330
in vec2 in_tex;
in vec2 in_img;
out vec2 v_img;
void main() {
    v_img = in_img;
    gl_Position = vec4(in_tex * 2.0 - 1.0, 0.0, 1.0);
}
"""

UNWRAP_FS = """
#version 330
uniform sampler2D uCam;
in vec2 v_img;
out vec4 fragColor;
void main() {
    fragColor = vec4(texture(uCam, v_img).bgr, 1.0);
}
"""

RESOLVE_FS = """
#version 330
uniform sampler2D uSrc;
uniform vec2 uOutRes;
in vec2 v_uv;
out vec4 fragColor;
void main() {
    vec2 o = 0.25 / uOutRes;
    vec3 c = texture(uSrc, v_uv + vec2(-o.x, -o.y)).rgb
           + texture(uSrc, v_uv + vec2( o.x, -o.y)).rgb
           + texture(uSrc, v_uv + vec2(-o.x,  o.y)).rgb
           + texture(uSrc, v_uv + vec2( o.x,  o.y)).rgb;
    fragColor = vec4(c * 0.25, 1.0);
}
"""

SCENE_FS = """
#version 330
in vec2 v_uv;
out vec4 fragColor;

uniform vec2  uRes;
uniform float uTime;
uniform float uZoom;
uniform float uCamY;
uniform vec3  uOffset;
uniform mat3  uHeadInv;
uniform vec3  uNeck;
uniform vec3  uNeckW;
uniform mat3  uBodyInv;
uniform vec3  uPivot;

uniform vec4  uHead;
uniform vec3  uHeadScale;
uniform vec4  uBody;
uniform vec3  uBodyScale;
uniform float uBlend;
uniform float uBumps;
uniform float uStretch;
uniform int   uLimbs;
uniform float uLimbR;
uniform int   uShape;       // 0 = two-lobe character (peanut, egg...), 1 = lizard
uniform vec4  uEye;         // lizard: xyz = eyeball center (head space, +x side), w = radius
uniform vec2  uBlink;       // lizard: eyelid closure, x = screen-left eye, y = screen-right eye
uniform float uBodyYaw;     // lizard: body turned this far (radians) toward back-right
uniform float uNetScale;
uniform vec3  uBelly;
uniform float uBellyAmt;
uniform vec3  uMouthLine;   // x = mouth height, y = rise toward the back, z = strength

uniform vec3  uBase;
uniform vec3  uDark;
uniform vec3  uLine;
uniform vec3  uLimbCol;
uniform vec3  uGloveCol;
uniform float uBurn;
uniform float uNet;
uniform float uGloss;

uniform sampler2D uFace;
uniform sampler2D uMask;
uniform sampler2D uCam;
uniform float uFaceAlpha;
uniform float uFaceSize;
uniform float uFaceY;
uniform vec3  uTint;
uniform float uTintAmt;

const float GROUND = -1.66;

uniform int   uBg;          // 0 studio gradient, 1 green screen, 2 webcam
uniform float uCamAspect;
uniform vec3  uBgTop;
uniform vec3  uBgBot;


// ---------------------------------------------------------------- SDF helpers
float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}
float sdEllipsoid(vec3 p, vec3 r) {
    float k0 = length(p / r);
    float k1 = length(p / (r * r));
    return k0 * (k0 - 1.0) / k1;
}
float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}
float sdRoundCone(vec3 p, vec3 a, vec3 b, float r1, float r2) {
    vec3 ba = b - a;
    float l2 = dot(ba, ba), rr = r1 - r2, a2 = l2 - rr * rr, il2 = 1.0 / l2;
    vec3 pa = p - a;
    float y = dot(pa, ba), z = y - l2;
    vec3 xv = pa * l2 - ba * y;
    float x2 = dot(xv, xv), y2 = y * y * l2, z2 = z * z * l2;
    float k = sign(rr) * rr * rr * x2;
    if (sign(z) * a2 * z2 > k) return sqrt(x2 + z2) * il2 - r2;
    if (sign(y) * a2 * y2 < k) return sqrt(x2 + y2) * il2 - r1;
    return (sqrt(x2 * a2 * il2) + y * rr) * il2 - r1;
}
float sdToes(vec3 q, vec3 c, vec3 dir, float len, float r) {
    // Three splayed toes with round toe pads, starting at c, pointing along dir.
    vec3 side = normalize(cross(dir, vec3(0.0, 1.0, 0.0)) + vec3(0.0, 0.0, 1e-4));
    float d = 1e9;
    for (int i = -1; i <= 1; i++) {
        vec3 tip = c + normalize(dir + side * 0.55 * float(i)) * len;
        d = min(d, sdCapsule(q, c, tip, r));
        d = min(d, length(q - tip) - r * 1.5);
    }
    return d;
}
vec3 toBody(vec3 p) { return uBodyInv * (p - uOffset - uPivot) + uPivot; }
vec3 toHead(vec3 p) { return uHeadInv * (p - uOffset - uNeckW) + uNeck; }

float bumps(vec3 p) {
    return sin(p.x * 17.0 + sin(p.y * 7.0)) * sin(p.y * 19.0 + 1.3) * sin(p.z * 16.0 + sin(p.x * 5.0));
}

// ------------------------------------------------------------------ lizard
// Body-local frame: the body is turned by uBodyYaw so it runs off to the
// back-right while the head faces the camera.
vec3 lizardLocal(vec3 pb) {
    float c = cos(uBodyYaw), s = sin(uBodyYaw);
    return vec3(pb.x * c + pb.z * s, pb.y, -pb.x * s + pb.z * c);
}
float lizardTail(vec3 q) {
    vec3 t0 = vec3(0.0, -1.30, -1.20), t1 = vec3(0.05, -1.52, -1.85);
    vec3 t2 = vec3(0.55, -1.58, -2.35), t3 = vec3(1.15, -1.58, -2.30), t4 = vec3(1.45, -1.50, -1.90);
    float d = sdRoundCone(q, t0, t1, 0.24, 0.15);
    d = smin(d, sdRoundCone(q, t1, t2, 0.15, 0.10), 0.05);
    d = smin(d, sdRoundCone(q, t2, t3, 0.10, 0.06), 0.04);
    return smin(d, sdRoundCone(q, t3, t4, 0.06, 0.025), 0.03);
}
float lizardLeg(vec3 q, vec3 a, vec3 b, vec3 c, vec3 toe) {
    // Cheap bound: skip the whole leg when we're far away from it.
    vec3 mid = (a + b + c) / 3.0;
    if (length(q - mid) > 0.8) return length(q - mid) - 0.6;
    float d = smin(sdRoundCone(q, a, b, 0.11, 0.08), sdRoundCone(q, b, c, 0.08, 0.06), 0.05);
    return smin(d, sdToes(q, c, toe, 0.20, 0.028), 0.05);
}
vec3 mapLizard(vec3 p) {
    vec3 ph = toHead(p);
    vec3 pb = toBody(p);
    vec3 q = lizardLocal(pb);

    // Big, rounded head with a short blunt snout ("huggable", not realistic).
    vec3 hs = vec3(1.0 - 0.3 * uStretch, 1.0 + uStretch, 1.0);
    float head = sdEllipsoid(ph - vec3(0.0, 0.06, 0.06), vec3(0.56, 0.43, 0.53) * hs);
    head = smin(head, sdEllipsoid(ph - vec3(0.0, -0.05, 0.42), vec3(0.40, 0.27, 0.38) * hs), 0.14);

    // Beady black eyes on the sides of the head, pointing outward (wall-eyed),
    // sitting in skin sockets with an upper lid that closes when you blink.
    vec3 qe = ph;
    float side = sign(qe.x);
    qe.x = abs(qe.x);
    vec3 E = uEye.xyz;
    float re = uEye.w;
    vec3 outward = normalize(E - vec3(0.0, 0.0, 0.05));
    head = smin(head, length(qe - (E - outward * re * 0.45)) - re * 1.05, 0.05);
    float blink = side < 0.0 ? uBlink.x : uBlink.y;
    float lid = max(length(qe - E) - re * 1.12, (E.y + re * (0.95 - 2.2 * blink)) - qe.y);
    head = min(head, lid);
    float eye = length(qe - E) - re;

    // Chest raised on the front legs, body and tail trailing behind.
    float body = sdEllipsoid(q - vec3(0.0, -0.66, 0.0), vec3(0.38, 0.38, 0.40));
    body = smin(body, sdRoundCone(q, vec3(0.0, -0.75, -0.10), vec3(0.0, -1.26, -1.00), 0.34, 0.28), 0.15);
    body = smin(body, sdRoundCone(pb, vec3(0.0, -0.60, 0.0), vec3(0.0, -0.25, 0.06), 0.33, 0.32), 0.12);
    body = smin(body, lizardTail(q), 0.10);

    vec3 qm = q;
    qm.x = abs(qm.x);
    float legs = lizardLeg(qm, vec3(0.24, -0.76, 0.08), vec3(0.57, -1.06, 0.14), vec3(0.60, -1.57, 0.22),
                           normalize(vec3(0.35, -0.05, 1.0)));
    legs = min(legs, lizardLeg(qm, vec3(0.24, -1.30, -0.98), vec3(0.62, -1.18, -0.82), vec3(0.66, -1.58, -0.78),
                                normalize(vec3(0.8, -0.05, 0.6))));
    body = smin(body, legs, 0.08);

    float d = smin(head, body, 0.16);
    vec3 res = vec3(d, 0.0, clamp(0.5 + (body - head) * 4.0, 0.0, 1.0));
    if (eye < res.x) res = vec3(eye, 3.0, 1.0);
    return res;
}

// x = distance, y = material (0 skin/shell, 1 limbs, 2 gloves, 3 eyes), z = head weight
vec3 map(vec3 p) {
    if (uShape == 1) return mapLizard(p);
    vec3 ph = toHead(p);
    vec3 pb = toBody(p);
    vec3 rh = uHead.w * uHeadScale * vec3(1.0 - 0.35 * uStretch, 1.0 + uStretch, 1.0 - 0.35 * uStretch);
    float dh = sdEllipsoid(ph - uHead.xyz, rh);
    float db = sdEllipsoid(pb - uBody.xyz, uBody.w * uBodyScale);
    float d = smin(dh, db, uBlend);
    float hw = clamp(0.5 + (db - dh) * 4.0, 0.0, 1.0);
    if (uBumps > 0.0) d += uBumps * mix(bumps(pb), bumps(ph), hw);
    vec3 res = vec3(d, 0.0, hw);

    if (uLimbs == 1) {
        vec3 q = pb;
        float side = sign(q.x);
        q.x = abs(q.x);
        vec3 br = uBody.w * uBodyScale;
        vec3 shoulder = vec3(br.x * 0.8, uBody.y + 0.12, 0.05);
        float wave = 0.06 * sin(uTime * 2.2 + side * 1.3);
        vec3 hand = vec3(br.x + 0.36, uBody.y - 0.32 + wave, 0.2);
        float arm = sdCapsule(q, shoulder, hand, uLimbR);
        float glove = sdEllipsoid(q - hand, vec3(0.11, 0.12, 0.10));
        float footY = GROUND + 0.07;
        vec3 hip = vec3(0.22, uBody.y - br.y * 0.8, 0.0);
        vec3 ankle = vec3(0.28, footY + 0.04, 0.02);
        float leg = sdCapsule(q, hip, ankle, uLimbR * 1.1);
        float shoe = sdEllipsoid(q - vec3(0.30, footY, 0.12), vec3(0.14, 0.08, 0.21));
        float limb = min(min(arm, leg), shoe);
        if (limb < res.x) res = vec3(limb, 1.0, 0.0);
        if (glove < res.x) res = vec3(glove, 2.0, 0.0);
    }
    return res;
}

vec3 calcNormal(vec3 p) {
    const vec2 e = vec2(1.0, -1.0) * 0.0015;
    return normalize(e.xyy * map(p + e.xyy).x + e.yyx * map(p + e.yyx).x +
                     e.yxy * map(p + e.yxy).x + e.xxx * map(p + e.xxx).x);
}

float calcAO(vec3 p, vec3 n) {
    float occ = 0.0, sca = 1.0;
    for (int i = 0; i < 5; i++) {
        float h = 0.01 + 0.15 * float(i) / 4.0;
        occ += (h - map(p + h * n).x) * sca;
        sca *= 0.9;
    }
    return clamp(1.0 - 2.0 * occ, 0.0, 1.0);
}

// ------------------------------------------------------------ shell material
vec3 hash3(vec3 p) {
    p = vec3(dot(p, vec3(127.1, 311.7, 74.7)), dot(p, vec3(269.5, 183.3, 246.1)),
             dot(p, vec3(113.5, 271.9, 124.6)));
    return fract(sin(p) * 43758.5453123);
}
float noise(vec3 x) {
    vec3 i = floor(x), f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    float n000 = hash3(i).x,                   n100 = hash3(i + vec3(1, 0, 0)).x;
    float n010 = hash3(i + vec3(0, 1, 0)).x,   n110 = hash3(i + vec3(1, 1, 0)).x;
    float n001 = hash3(i + vec3(0, 0, 1)).x,   n101 = hash3(i + vec3(1, 0, 1)).x;
    float n011 = hash3(i + vec3(0, 1, 1)).x,   n111 = hash3(i + vec3(1, 1, 1)).x;
    return mix(mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y),
               mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y), f.z);
}
float fbm(vec3 p) {
    float a = 0.5, s = 0.0;
    for (int i = 0; i < 4; i++) { s += a * noise(p); p = p * 2.03 + 11.7; a *= 0.5; }
    return s;
}
float voronoiEdge(vec3 x) {
    vec3 n = floor(x), f = fract(x);
    float f1 = 8.0, f2 = 8.0;
    for (int k = -1; k <= 1; k++)
    for (int j = -1; j <= 1; j++)
    for (int i = -1; i <= 1; i++) {
        vec3 g = vec3(i, j, k);
        vec3 r = g + hash3(n + g) - f;
        float d = dot(r, r);
        if (d < f1) { f2 = f1; f1 = d; } else if (d < f2) { f2 = d; }
    }
    return sqrt(f2) - sqrt(f1);
}
vec3 shell(vec3 q) {
    float burn = smoothstep(0.38, 0.72, fbm(q * 2.2 + 3.1)) * uBurn;
    float speck = smoothstep(0.6, 0.78, fbm(q * 9.0)) * uBurn;
    vec3 c = mix(uBase, uDark, clamp(burn + 0.6 * speck, 0.0, 1.0));
    c *= 0.88 + 0.24 * fbm(q * 6.0);
    if (uNet > 0.0) {
        float e = voronoiEdge(q * vec3(9.0, 6.0, 9.0) * uNetScale + 0.15 * fbm(q * 4.0));
        float net = (1.0 - smoothstep(0.02, 0.09, e)) * uNet;
        c = mix(c, uLine * mix(1.0, 0.35, burn), net);
    }
    return c;
}

// -------------------------------------------------------------------- main
void main() {
    vec2 frag = v_uv * uRes;
    vec2 p = (2.0 * frag - uRes) / uRes.y;
    vec3 ro = vec3(0.0, mix(-0.2, 0.5, clamp(uZoom - 1.0, 0.0, 1.0)) + uCamY, 5.2);
    vec3 rd = normalize(vec3(p, -2.75 * uZoom));

    vec3 col;
    if (uBg == 1) {
        col = vec3(0.0, 1.0, 0.0);
    } else if (uBg == 2) {
        vec2 cuv = v_uv - 0.5;
        float ra = (uRes.x / uRes.y) / uCamAspect;
        if (ra > 1.0) cuv.y /= ra; else cuv.x *= ra;
        cuv += 0.5;
        col = texture(uCam, vec2(cuv.x, 1.0 - cuv.y)).bgr * 0.85;
    } else {
        col = mix(uBgBot, uBgTop, smoothstep(-0.2, 0.9, v_uv.y));
        col *= 1.0 - 0.18 * dot(p, p);
        if (rd.y < 0.0) {
            float tg = (GROUND + uOffset.y - ro.y) / rd.y;
            vec2 d = (ro + rd * tg).xz - uOffset.xz;
            float s = exp(-dot(d * vec2(1.4, 3.0), d * vec2(1.4, 3.0)));
            col *= 1.0 - 0.5 * s;
        }
    }

    // Bounding sphere around the character to skip empty pixels quickly.
    vec3 oc = ro - (uOffset + vec3(0.0, -0.25, 0.0));
    float b = dot(oc, rd);
    float bR = uShape == 1 ? 3.2 : 2.1;
    float h = b * b - (dot(oc, oc) - bR * bR);
    if (h > 0.0) {
        h = sqrt(h);
        float t = max(0.0, -b - h), tmax = -b + h;
        vec3 m = vec3(0.0);
        bool hit = false;
        for (int i = 0; i < 128; i++) {
            m = map(ro + rd * t);
            if (m.x < 0.0008 * t) { hit = true; break; }
            t += m.x * 0.85;
            if (t > tmax) break;
        }
        if (hit) {
            vec3 pos = ro + rd * t;
            vec3 n = calcNormal(pos);
            vec3 ph = toHead(pos);
            vec3 alb;
            float gloss = uGloss;
            float faceM = 0.0;
            vec3 faceC = vec3(0.0);

            if (m.y < 0.5) {
                vec3 pb = toBody(pos);
                alb = shell(mix(pb, ph, m.z));
                if (uBellyAmt > 0.0) {
                    // Lighter throat and belly on the undersides.
                    vec3 nl = m.z > 0.5 ? uHeadInv * n : uBodyInv * n;
                    alb = mix(alb, uBelly, smoothstep(-0.05, -0.55, nl.y) * uBellyAmt);
                }
                if (uMouthLine.z > 0.0 && m.z > 0.5) {
                    // Wide lizard grin: from the corners of your mouth back along the jaw.
                    float my = uMouthLine.x + uMouthLine.y * max(0.0, 0.75 - ph.z);
                    float line = (1.0 - smoothstep(0.006, 0.016, abs(ph.y - my)))
                               * smoothstep(0.10, 0.16, abs(ph.x)) * step(0.05, ph.z);
                    alb = mix(alb, uDark * 0.45, line * uMouthLine.z);
                }
                vec3 nh = uHeadInv * n;
                vec2 fuv = vec2(0.5 + (ph.x - uHead.x) / uFaceSize, 0.5 - (ph.y - uFaceY) / uFaceSize);
                if (all(greaterThan(fuv, vec2(0.0))) && all(lessThan(fuv, vec2(1.0)))) {
                    faceM = texture(uMask, fuv).r * smoothstep(0.1, 0.5, nh.z) * m.z * uFaceAlpha;
                    faceC = texture(uFace, fuv).rgb * mix(vec3(1.0), uTint, uTintAmt);
                }
            } else if (m.y < 1.5) {
                alb = uLimbCol; gloss = 0.5;
            } else if (m.y < 2.5) {
                alb = uGloveCol; gloss = 0.3;
            } else {
                alb = vec3(0.012, 0.012, 0.016); gloss = 3.0;   // glossy black eyes
            }

            vec3 L = normalize(vec3(-0.5, 0.7, 0.6));
            float dif = clamp(dot(n, L), 0.0, 1.0);
            float sky = 0.5 + 0.5 * n.y;
            float ao = calcAO(pos, n);
            float fre = pow(clamp(1.0 + dot(n, rd), 0.0, 1.0), 3.0);
            float spe = pow(clamp(dot(n, normalize(L - rd)), 0.0, 1.0), 32.0) * gloss;
            vec3 lin = 1.2 * dif * vec3(1.0, 0.93, 0.82)
                     + 0.5 * sky * vec3(0.55, 0.65, 0.85) * ao
                     + 0.25 * ao * vec3(0.35, 0.25, 0.18);
            vec3 shaded = alb * lin + spe * vec3(1.0, 0.95, 0.9) * ao + 0.3 * fre * vec3(1.0, 0.9, 0.8) * ao;
            // The face keeps most of its own lighting so it stays readable.
            vec3 faceLit = faceC * (0.7 + 0.4 * dif) * mix(1.0, ao, 0.5);
            col = mix(shaded, faceLit, faceM);
        }
    }
    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""


def _create_context() -> moderngl.Context:
    try:
        return moderngl.create_standalone_context(require=330)
    except Exception:
        # Headless Linux (no X/Wayland) - fall back to EGL.
        return moderngl.create_standalone_context(require=330, backend="egl")


class Renderer:
    def __init__(self, width: int, height: int, supersample: float = 1.5):
        self.ctx = ctx = _create_context()
        self.size = (int(width), int(height))
        self.ss_size = (max(1, int(width * supersample)), max(1, int(height * supersample)))

        self.cam_tex = None
        self.cam_aspect = 16 / 9

        self.face_tex = ctx.texture((FACE_TEX_SIZE, FACE_TEX_SIZE), 3)
        self.face_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.face_tex.repeat_x = self.face_tex.repeat_y = False
        self.face_fbo = ctx.framebuffer([self.face_tex])
        self.face_fbo.clear(0.55, 0.42, 0.35)

        self.mask_tex = ctx.texture((MASK_TEX_SIZE, MASK_TEX_SIZE), 1)
        self.mask_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.mask_tex.repeat_x = self.mask_tex.repeat_y = False
        self.mask_tex.write(np.zeros((MASK_TEX_SIZE, MASK_TEX_SIZE), np.uint8).tobytes())

        self.unwrap_prog = ctx.program(vertex_shader=UNWRAP_VS, fragment_shader=UNWRAP_FS)
        self.unwrap_vbo = None
        self.unwrap_vao = None

        self.scene_prog = ctx.program(vertex_shader=FULLSCREEN_VS, fragment_shader=SCENE_FS)
        self.scene_vao = ctx.vertex_array(self.scene_prog, [])
        self.scene_tex = ctx.texture(self.ss_size, 3)
        self.scene_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.scene_fbo = ctx.framebuffer([self.scene_tex])

        self.resolve_prog = ctx.program(vertex_shader=FULLSCREEN_VS, fragment_shader=RESOLVE_FS)
        self.resolve_vao = ctx.vertex_array(self.resolve_prog, [])
        self.out_fbo = ctx.simple_framebuffer(self.size, components=3)

        self.scene_prog["uFace"].value = 0
        self.scene_prog["uMask"].value = 1
        self.scene_prog["uCam"].value = 2
        self.unwrap_prog["uCam"].value = 2
        self.resolve_prog["uSrc"].value = 3

    # ------------------------------------------------------------------ input
    def upload_camera(self, frame_bgr: np.ndarray):
        h, w = frame_bgr.shape[:2]
        if self.cam_tex is None or self.cam_tex.size != (w, h):
            if self.cam_tex is not None:
                self.cam_tex.release()
            self.cam_tex = self.ctx.texture((w, h), 3, alignment=1)
            self.cam_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.cam_tex.repeat_x = self.cam_tex.repeat_y = False
            self.cam_aspect = w / h
        self.cam_tex.write(np.ascontiguousarray(frame_bgr).tobytes())

    def update_face(self, face: FaceState, mask: np.ndarray, triangles: np.ndarray):
        verts = np.hstack([face.tex_uv, face.img_uv]).astype("f4")
        if self.unwrap_vao is None:
            self.unwrap_vbo = self.ctx.buffer(reserve=verts.nbytes, dynamic=True)
            ibo = self.ctx.buffer(np.ascontiguousarray(triangles, dtype="i4").tobytes())
            self.unwrap_vao = self.ctx.vertex_array(
                self.unwrap_prog, [(self.unwrap_vbo, "2f 2f", "in_tex", "in_img")], ibo
            )
        self.unwrap_vbo.write(verts.tobytes())
        self.face_fbo.use()
        self.ctx.disable(moderngl.CULL_FACE | moderngl.DEPTH_TEST)
        self.cam_tex.use(2)
        self.unwrap_vao.render(moderngl.TRIANGLES)
        self.mask_tex.write(np.ascontiguousarray(mask).tobytes())

    # ----------------------------------------------------------------- render
    def render(self, uniforms: dict) -> np.ndarray:
        """Render one frame, returns an (H, W, 3) BGR uint8 image."""
        prog = self.scene_prog
        uniforms = dict(uniforms)
        uniforms["uRes"] = self.ss_size
        uniforms["uCamAspect"] = self.cam_aspect
        for name, value in uniforms.items():
            if name in prog:
                prog[name].value = value

        self.face_tex.use(0)
        self.mask_tex.use(1)
        if self.cam_tex is not None:
            self.cam_tex.use(2)
        self.scene_fbo.use()
        self.scene_vao.render(moderngl.TRIANGLES, vertices=3)

        self.out_fbo.use()
        self.scene_tex.use(3)
        self.resolve_prog["uOutRes"].value = self.size
        self.resolve_vao.render(moderngl.TRIANGLES, vertices=3)

        data = self.out_fbo.read(components=3, alignment=1)
        img = np.frombuffer(data, np.uint8).reshape(self.size[1], self.size[0], 3)
        return np.ascontiguousarray(img[::-1, :, ::-1])  # GL is bottom-up RGB
