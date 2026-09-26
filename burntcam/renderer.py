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
uniform float uCamY;        // camera height (set by the view preset)
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
uniform vec2  uBlink;       // lizard: eyelid closure, x = screen-left eye, y = screen-right eye
uniform float uJaw;         // lizard: mouth opening 0..1
uniform float uTongue;      // lizard: tongue sticking out 0..1
uniform float uBodyYaw;     // lizard: body turned this far (radians)
uniform float uNetScale;
uniform vec3  uBelly;
uniform float uBellyAmt;

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

uniform int   uBg;          // 0 studio gradient, 1 green screen, 2 webcam, 3 custom image/video
uniform float uCamAspect;
uniform sampler2D uBgTex;
uniform float uBgAspect;
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

float smax(float a, float b, float k) { return -smin(-a, -b, k); }

float bumps(vec3 p) {
    return sin(p.x * 17.0 + sin(p.y * 7.0)) * sin(p.y * 19.0 + 1.3) * sin(p.z * 16.0 + sin(p.x * 5.0));
}

// ------------------------------------------------------------------ lizard
// A cute cartoon lizard: a soft bean-shaped body whose top is the head, big
// googly eyes on top, a wide mouth that opens with yours, thin arms, short
// bow legs and a thick curling tail. Materials: 0 skin, 3 eyes, 4 mouth, 5 tongue.
const vec3  LZ_EYE = vec3(0.18, 0.69, 0.10);
const float LZ_EYE_R = 0.155;

vec3 lizardLocal(vec3 pb) {
    float c = cos(uBodyYaw), s = sin(uBodyYaw);
    return vec3(pb.x * c + pb.z * s, pb.y, -pb.x * s + pb.z * c);
}
float lizardTail(vec3 q) {
    // Sweeps out along the ground, then curls up into a hook.
    const vec4 P[7] = vec4[7](
        vec4(-0.05, -1.00, -0.25, 0.27), vec4(-0.45, -1.38, -0.40, 0.21),
        vec4(-0.95, -1.47, -0.40, 0.16), vec4(-1.36, -1.36, -0.35, 0.125),
        vec4(-1.56, -1.06, -0.30, 0.095), vec4(-1.49, -0.77, -0.28, 0.075),
        vec4(-1.27, -0.64, -0.26, 0.06));
    if (length(q - vec3(-0.85, -1.05, -0.33)) > 1.35) return length(q - vec3(-0.85, -1.05, -0.33)) - 1.0;
    float d = 1e9;
    for (int i = 0; i < 6; i++)
        d = smin(d, sdRoundCone(q, P[i].xyz, P[i + 1].xyz, P[i].w, P[i + 1].w), 0.09);
    return d;
}
float lizardDigits(vec3 q, vec3 c, vec3 dir, vec3 up, float len, float r, float spread) {
    // Four little fingers/toes with round tips, fanned out around dir.
    vec3 side = normalize(cross(dir, up));
    float d = 1e9;
    for (int i = 0; i < 4; i++) {
        vec3 tip = c + normalize(dir + side * spread * (float(i) - 1.5)) * len;
        d = min(d, sdCapsule(q, c, tip, r));
        d = min(d, length(q - tip) - r * 1.45);
    }
    return d;
}
float lizardLimbs(vec3 qm, float side) {
    float d = 1e9;
    // Arms: thin, held out to the sides with a little idle wave.
    if (length(qm - vec3(0.45, -0.30, 0.10)) < 0.65) {
        float wave = 0.05 * sin(uTime * 2.3 + side * 1.7);
        vec3 sh = vec3(0.24, -0.10, 0.04), el = vec3(0.47, -0.27, 0.10), wr = vec3(0.64, -0.40 + wave, 0.15);
        d = smin(sdRoundCone(qm, sh, el, 0.065, 0.048), sdRoundCone(qm, el, wr, 0.048, 0.042), 0.03);
        vec3 dir = normalize(wr - el);
        d = smin(d, length(qm - (wr + dir * 0.03)) - 0.055, 0.03);
        d = smin(d, lizardDigits(qm, wr + dir * 0.04, dir, vec3(0.0, 0.0, 1.0), 0.10, 0.017, 0.55), 0.02);
    } else {
        d = length(qm - vec3(0.45, -0.30, 0.10)) - 0.5;
    }
    // Legs: short and bent, feet planted wide.
    if (length(qm - vec3(0.32, -1.35, 0.08)) < 0.6) {
        vec3 hp = vec3(0.20, -1.00, 0.02), kn = vec3(0.42, -1.27, 0.14), an = vec3(0.42, -1.56, 0.05);
        float l = smin(sdRoundCone(qm, hp, kn, 0.13, 0.095), sdRoundCone(qm, kn, an, 0.095, 0.075), 0.05);
        l = smin(l, sdEllipsoid(qm - vec3(0.44, -1.61, 0.11), vec3(0.11, 0.055, 0.13)), 0.05);
        l = smin(l, lizardDigits(qm, vec3(0.44, -1.625, 0.16), normalize(vec3(0.30, 0.0, 1.0)),
                                 vec3(0.0, 1.0, 0.0), 0.12, 0.026, 0.45), 0.03);
        d = min(d, l);
    } else {
        d = min(d, length(qm - vec3(0.32, -1.35, 0.08)) - 0.45);
    }
    return d;
}
// A flat, tongue-shaped segment from a to b (flattened across its direction).
float sdTongue(vec3 p, vec3 a, vec3 b, float r1, float r2) {
    vec3 dir = b - a;
    float len = length(dir);
    if (len < 1e-4) return length(p - a) - r1;
    vec3 f = normalize(cross(dir / len, vec3(1.0, 0.0, 0.0)));
    vec3 q = p - a;
    q += f * dot(q, f) * 1.4;                         // 2.4x thinner than wide
    return sdRoundCone(q, vec3(0.0), dir, r1, r2) / 2.4;
}

// How far the mouth is open: your jaw, or enough to let the tongue out.
float lizardJaw() { return max(uJaw, 0.45 * uTongue); }

// Mouth cavity in head space: a smile-shaped slit that opens with the jaw.
float lizardMouth(vec3 ph) {
    float uJaw = lizardJaw();
    vec3 c = ph - vec3(0.0, 0.305 - 0.10 * uJaw, 0.36);
    c.y -= 0.55 * c.x * c.x;                               // corners curl up into a smile
    return sdEllipsoid(c, vec3(0.27 + 0.02 * uJaw, 0.010 + 0.21 * uJaw, 0.27));
}
vec3 mapLizard(vec3 p) {
    float uJaw = lizardJaw();
    vec3 ph = toHead(p);
    vec3 pb = toBody(p);
    vec3 q = lizardLocal(pb);

    // Head: the rounded top of the body plus a blunt snout.
    vec3 hs = vec3(1.0, 1.0 + 0.6 * uStretch, 1.0);
    float head = sdEllipsoid(ph - vec3(0.0, 0.41, -0.02), vec3(0.37, 0.28, 0.32) * hs);
    head = smin(head, sdEllipsoid(ph - vec3(0.0, 0.36, 0.23), vec3(0.36, 0.215 + 0.10 * uJaw, 0.32)), 0.12);
    head = smin(head, sdRoundCone(ph, vec3(0.0, -0.05, -0.02), vec3(0.0, 0.32, -0.02), 0.31, 0.30), 0.10);
    vec3 qe = ph;
    float side = sign(qe.x);
    qe.x = abs(qe.x);
    head = smin(head, length(qe - (LZ_EYE - vec3(0.0, 0.07, 0.0))) - LZ_EYE_R * 0.85, 0.07);  // eye bumps
    float cav = lizardMouth(ph);
    float mouth = smax(head, -cav, 0.02);

    // Eyelids (skin) slide down over the eyeballs when you blink.
    float blink = side < 0.0 ? uBlink.x : uBlink.y;
    float lid = max(length(qe - LZ_EYE) - LZ_EYE_R * 1.07,
                    (LZ_EYE.y + LZ_EYE_R * (1.25 - 2.5 * blink)) - qe.y);
    float eye = length(qe - LZ_EYE) - LZ_EYE_R;

    // Body: a soft bean, a little wider at the bottom.
    float body = sdEllipsoid(q - vec3(0.0, -0.76, 0.02), vec3(0.41, 0.44, 0.37));
    body = smin(body, sdRoundCone(q, vec3(0.0, -0.70, 0.0), vec3(0.0, 0.10, -0.01), 0.37, 0.31), 0.15);
    body = smin(body, lizardTail(q), 0.10);
    vec3 qm = q;
    qm.x = abs(qm.x);
    body = smin(body, lizardLimbs(qm, sign(q.x)), 0.05);

    float d = smin(mouth, body, 0.18);
    float hw = clamp(0.5 + (body - head) * 4.0, 0.0, 1.0);
    vec3 res = vec3(d, -cav > head - 0.004 ? 4.0 : 0.0, hw);
    if (lid < res.x) res = vec3(lid, 0.0, 1.0);
    if (eye < res.x) res = vec3(eye, 3.0, 1.0);
    // Tongue: rests inside the mouth, visible once it opens, and sticks out
    // (with a little wiggle) when you stick yours out.
    float tongue = sdEllipsoid(ph - vec3(0.0, 0.26 - 0.17 * uJaw, 0.30 + 0.10 * uJaw), vec3(0.15, 0.075, 0.19));
    if (uTongue > 0.02) {
        float wig = 0.05 * sin(uTime * 7.0) * uTongue;
        vec3 base = vec3(0.0, 0.25 - 0.12 * uJaw, 0.36);
        vec3 mid = base + vec3(wig, -0.05 * uTongue, 0.30 * uTongue);    // out toward you...
        vec3 tip = mid + vec3(wig * 1.5, -0.17 * uTongue, 0.07 * uTongue);  // ...then droops
        tongue = smin(tongue, sdTongue(ph, base, mid, 0.135, 0.125), 0.05);
        tongue = smin(tongue, sdTongue(ph, mid, tip, 0.125, 0.10), 0.05);
    } else {
        tongue = smax(tongue, head - 0.005, 0.01);      // keep it inside the closed mouth
    }
    if (tongue < res.x) res = vec3(tongue, 5.0, 1.0);
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

vec3 lizardAlbedo(vec3 pos, vec3 n, vec3 m, out float gloss) {
    vec3 ph = toHead(pos);
    vec3 q = lizardLocal(toBody(pos));
    gloss = uGloss;
    if (m.y > 4.5) {                                                        // tongue
        gloss = 0.6;
        float wig = 0.05 * sin(uTime * 7.0) * uTongue;
        float groove = 1.0 - smoothstep(0.006, 0.02, abs(ph.x - wig * 1.2)) * step(0.02, uTongue);
        return mix(vec3(0.80, 0.32, 0.45), vec3(0.93, 0.45, 0.58), groove);
    }
    if (m.y > 3.5) { gloss = 0.1; return vec3(0.36, 0.08, 0.15); }          // inside of mouth
    if (m.y > 2.5) {
        // Googly eye: ivory white with a small black pupil. The pupils point
        // outward and drift a little, so the eyes never quite agree.
        vec3 qe = ph;
        float side = sign(qe.x);
        qe.x = abs(qe.x);
        vec3 e = normalize(qe - LZ_EYE);
        vec3 look = normalize(vec3(0.30 + 0.10 * sin(uTime * 0.7 + side * 2.0),
                                   0.12 + 0.06 * sin(uTime * 0.9 + side), 1.0));
        float pupil = smoothstep(0.955, 0.965, dot(e, look));
        gloss = 1.4;
        return mix(vec3(0.97, 0.95, 0.86), vec3(0.02, 0.03, 0.06), pupil);
    }
    vec3 col = uBase;
    // Paler belly and throat.
    vec3 nb = uBodyInv * n;
    float belly = smoothstep(0.25, 0.75, nb.z) * (1.0 - smoothstep(0.18, 0.30, abs(q.x)))
                * smoothstep(-1.25, -0.9, q.y) * (1.0 - m.z * 0.6);
    col = mix(col, uBelly, belly * uBellyAmt);
    // Soft scales on top of the tail only.
    float onTail = 1.0 - smoothstep(0.0, 0.06, lizardTail(q));
    if (onTail > 0.0) {
        float e = voronoiEdge(q * vec3(11.0, 11.0, 11.0) * uNetScale);
        col = mix(col, uLine, (1.0 - smoothstep(0.03, 0.12, e)) * onTail * uNet * smoothstep(-0.3, 0.4, n.y));
    }
    col *= 0.94 + 0.08 * fbm(pos * 3.0);
    if (m.z > 0.5) {
        // Two little nostrils on top of the snout.
        vec3 qn = ph;
        qn.x = abs(qn.x);
        vec3 nh = uHeadInv * n;
        float nos = (1.0 - smoothstep(0.012, 0.02, length(qn.xz - vec2(0.045, 0.47))))
                  * step(0.35, nh.y) * step(0.40, qn.y);
        col = mix(col, uDark, nos);
    }
    return col;
}

// -------------------------------------------------------------------- main
// Fill the frame with an image, cropping (not stretching) to fit.
vec3 coverSample(sampler2D tex, float aspect) {
    vec2 cuv = v_uv - 0.5;
    float ra = (uRes.x / uRes.y) / aspect;
    if (ra > 1.0) cuv.y /= ra; else cuv.x *= ra;
    cuv += 0.5;
    return texture(tex, vec2(cuv.x, 1.0 - cuv.y)).bgr;
}

void main() {
    vec2 frag = v_uv * uRes;
    vec2 p = (2.0 * frag - uRes) / uRes.y;
    vec3 ro = vec3(0.0, uCamY, 5.2);
    vec3 rd = normalize(vec3(p, -2.75 * uZoom));

    vec3 col;
    if (uBg == 1) {
        col = vec3(0.0, 1.0, 0.0);
    } else if (uBg == 2) {
        col = coverSample(uCam, uCamAspect) * 0.85;
    } else if (uBg == 3) {
        col = coverSample(uBgTex, uBgAspect);
    } else {
        col = mix(uBgBot, uBgTop, smoothstep(-0.2, 0.9, v_uv.y));
        col *= 1.0 - 0.18 * dot(p, p);
    }
    // Soft contact shadow so the character stands on the floor.
    if ((uBg == 0 || uBg == 3) && rd.y < 0.0) {
        float tg = (GROUND + uOffset.y - ro.y) / rd.y;
        vec2 d = (ro + rd * tg).xz - uOffset.xz;
        float s = exp(-dot(d * vec2(1.4, 3.0), d * vec2(1.4, 3.0)));
        col *= 1.0 - (uBg == 0 ? 0.5 : 0.35) * s;
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

            if (uShape == 1) {
                alb = lizardAlbedo(pos, n, m, gloss);
            } else if (m.y < 0.5) {
                alb = shell(mix(toBody(pos), ph, m.z));
            } else if (m.y < 1.5) {
                alb = uLimbCol; gloss = 0.5;
            } else {
                alb = uGloveCol; gloss = 0.3;
            }
            if (m.y < 0.5) {
                vec3 nh = uHeadInv * n;
                vec2 fuv = vec2(0.5 + (ph.x - uHead.x) / uFaceSize, 0.5 - (ph.y - uFaceY) / uFaceSize);
                if (all(greaterThan(fuv, vec2(0.0))) && all(lessThan(fuv, vec2(1.0)))) {
                    faceM = texture(uMask, fuv).r * smoothstep(0.1, 0.5, nh.z) * m.z * uFaceAlpha;
                    faceC = texture(uFace, fuv).rgb * mix(vec3(1.0), uTint, uTintAmt);
                }
            }

            vec3 L = normalize(vec3(-0.5, 0.7, 0.6));
            // Lizard skin is soft and matte, so light wraps around it a little.
            float wrap = uShape == 1 ? 0.35 : 0.0;
            float dif = clamp((dot(n, L) + wrap) / (1.0 + wrap), 0.0, 1.0);
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
        self.scene_prog["uBgTex"].value = 4
        self.bg_tex = None
        self.bg_aspect = 16 / 9
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

    def upload_background(self, image_bgr: np.ndarray):
        """Set the custom background (an image, or the current video frame)."""
        h, w = image_bgr.shape[:2]
        if self.bg_tex is None or self.bg_tex.size != (w, h):
            if self.bg_tex is not None:
                self.bg_tex.release()
            self.bg_tex = self.ctx.texture((w, h), 3, alignment=1)
            self.bg_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.bg_tex.repeat_x = self.bg_tex.repeat_y = False
            self.bg_aspect = w / h
        self.bg_tex.write(np.ascontiguousarray(image_bgr).tobytes())

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
        uniforms["uBgAspect"] = self.bg_aspect
        for name, value in uniforms.items():
            if name in prog:
                prog[name].value = value

        self.face_tex.use(0)
        self.mask_tex.use(1)
        if self.cam_tex is not None:
            self.cam_tex.use(2)
        if self.bg_tex is not None:
            self.bg_tex.use(4)
        self.scene_fbo.use()
        self.scene_vao.render(moderngl.TRIANGLES, vertices=3)

        self.out_fbo.use()
        self.scene_tex.use(3)
        self.resolve_prog["uOutRes"].value = self.size
        self.resolve_vao.render(moderngl.TRIANGLES, vertices=3)

        data = self.out_fbo.read(components=3, alignment=1)
        img = np.frombuffer(data, np.uint8).reshape(self.size[1], self.size[0], 3)
        return np.ascontiguousarray(img[::-1, :, ::-1])  # GL is bottom-up RGB
