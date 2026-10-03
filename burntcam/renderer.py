"""GPU renderer (moderngl / OpenGL 3.3).

Pass 1 - "unwrap": draws the tracked face mesh with the webcam as its texture
          into a front-facing 512x512 face texture.
Pass 2 - "scene":  ray-marches the signed-distance character, paints the face
          texture onto its head through a soft mask, lights it and draws the
          background.  Rendered at ``supersample`` x resolution.
Pass 3 - "resolve": downsamples to the output size (anti-aliasing).
"""

from __future__ import annotations

import cv2
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
uniform int   uHeadOnly;    // 1 = both lobes are the head (a floating peanut); no body, no limbs
uniform int   uHat;         // 0 none, 1 propeller beanie, 2 cowboy hat
uniform float uCrack;       // dark dry cracks in the shell
uniform int   uShape;       // 0 = two-lobe character (peanut, egg...), 1 = lizard, 2 = dark mage, 3 = duelist, 4 = sea sponge
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
uniform sampler2D uLipMask;
uniform vec3  uLipTint;
uniform float uLipAmt;
uniform vec2  uEyeL;        // feature centers in face-texture space
uniform vec2  uEyeR;
uniform vec2  uMouth;
uniform float uEyeScale;    // magnify the eyes / mouth on the character (1 = as they are)
uniform float uMouthScale;
uniform float uEyeSpread;   // push the eyes apart (face-texture units)
uniform sampler2D uFeatMask; // eyes + mouth: these stay opaque when the rest of the face is see-through
uniform float uFaceOpacity; // how solid the face is outside the eyes and mouth (1 = solid)
uniform sampler2D uShellTex; // photo of a peanut shell, wrapped around the head (optional)
uniform int   uShellOn;
uniform vec3  uShellTint;
uniform float uShellBump;   // fake relief from the shell photo
uniform vec2  uShellRepeat; // times the photo repeats around / along the peanut

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
float sdBox(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
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

// Shows the face texture around c enlarged s times (within radius r), with
// the enlarged feature moved by `shift`. Outside r nothing changes.
vec2 magnify(vec2 uv, vec2 c, float r, float s, vec2 shift) {
    vec2 cs = c + shift;
    vec2 d = uv - cs;
    float l = length(d) / r;
    if (l >= 1.0 || (s <= 1.0 && shift == vec2(0.0))) return uv;
    float w = 1.0 - smoothstep(0.0, 1.0, l);
    return cs + d * mix(1.0, 1.0 / s, w) - shift * w;
}

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

// ---------------------------------------------------------------- dark mage
// A tall, slim sorcerer: a narrow hat with spiral bands curling forward and
// a fin behind it, big layered pauldrons, chrome-purple armor with lilac
// trim, a blue-violet robe open at the front over armored legs, and a long
// teal staff with a leaf-shaped loop and a glowing orb. Your face shows in
// the hat's opening.
// Materials: 0 skin, 6 hat, 7 trim, 8 orb, 9 staff, 10 robe, 11 armor,
// 12 pauldrons, 13 hair.
float sdTorus(vec3 p, float R, float r) {
    vec2 q = vec2(length(p.xz) - R, p.y);
    return length(q) - r;
}
vec3 rotZ(vec3 p, float a) {
    float c = cos(a), s = sin(a);
    return vec3(c * p.x + s * p.y, -s * p.x + c * p.y, p.z);
}
const vec3 MG_STAFF = vec3(-0.44, 0.0, 0.33);
const vec3 MG_LOOP = vec3(-0.44, 1.04, 0.33);
float mageHat(vec3 ph) {
    float helm = sdEllipsoid(ph - vec3(0.0, 0.52, -0.03), vec3(0.27, 0.30, 0.28));
    helm = smax(helm, -sdEllipsoid(ph - vec3(0.0, 0.37, 0.22), vec3(0.21, 0.31, 0.26)), 0.02);
    const vec3 H0 = vec3(0.0, 0.62, -0.04), H1 = vec3(0.0, 0.97, -0.07), H2 = vec3(0.03, 1.26, -0.03);
    const vec3 H3 = vec3(0.10, 1.48, 0.05), H4 = vec3(0.21, 1.61, 0.15);
    float cone = sdRoundCone(ph, H0, H1, 0.255, 0.19);
    cone = smin(cone, sdRoundCone(ph, H1, H2, 0.19, 0.12), 0.04);
    cone = smin(cone, sdRoundCone(ph, H2, H3, 0.12, 0.06), 0.03);
    cone = smin(cone, sdRoundCone(ph, H3, H4, 0.06, 0.006), 0.02);
    float hat = smin(helm, cone, 0.06);
    // The flat pointed fin behind the hat.
    vec3 qf = ph;
    qf.z = (qf.z + 0.24) * 3.0;
    hat = smin(hat, sdRoundCone(qf, vec3(-0.06, 0.58, 0.0), vec3(-0.36, 1.32, 0.0), 0.17, 0.005) / 3.0, 0.03);
    vec3 qh = ph;
    qh.x = abs(qh.x);
    return smin(hat, sdEllipsoid(qh - vec3(0.225, 0.33, 0.04), vec3(0.06, 0.22, 0.13)), 0.04);  // cheek guards
}
float magePauldron(vec3 qs, out float edge) {
    // Two big curved plates per shoulder, the upper one flaring up and out.
    vec3 q1 = rotZ(qs - vec3(0.40, -0.01, 0.0), 0.36);
    vec3 q2 = rotZ(qs - vec3(0.47, 0.15, -0.04), 0.78);
    q1.y += 0.35 * q1.x * q1.x;                           // curve the plates down at the ends
    q2.y += 0.30 * q2.x * q2.x;
    float d1 = sdEllipsoid(q1, vec3(0.35, 0.09, 0.29)) * 0.8;
    float d2 = sdEllipsoid(q2, vec3(0.33, 0.065, 0.25)) * 0.8;
    edge = d1 < d2 ? length(q1.xz / vec2(0.35, 0.29)) : length(q2.xz / vec2(0.33, 0.25));
    return min(min(d1, d2), length(qs - vec3(0.31, -0.08, 0.0)) - 0.12);
}
vec3 mapMage(vec3 p) {
    // Evaluated in four groups (head+hat, upper body, lower body, staff). A
    // group is skipped when its bounding sphere is farther than the nearest
    // surface found so far, which keeps this detailed model fast.
    vec3 ph = toHead(p);
    vec3 pb = toBody(p);
    vec3 qs = pb;
    qs.x = abs(qs.x);
    vec3 res = vec3(1e9, 0.0, 0.0);
    float bound;

    // Head, hat and a dark fringe of hair framing the face.
    bound = length(ph - vec3(0.0, 0.85, -0.10)) - 0.95;
    if (bound < res.x) {
        res = vec3(sdEllipsoid(ph - vec3(0.0, 0.39, 0.0), vec3(0.21, 0.29, 0.22)), 0.0, 1.0);
        float hat = mageHat(ph);
        if (hat < res.x) res = vec3(hat, 6.0, 1.0);
        vec3 qh = ph;
        qh.x = abs(qh.x);
        float hair = smin(sdEllipsoid(qh - vec3(0.215, 0.58, 0.10), vec3(0.045, 0.10, 0.06)),
                          sdEllipsoid(qh - vec3(0.21, 0.42, 0.09), vec3(0.035, 0.13, 0.05)), 0.04);
        if (hair < res.x) res = vec3(hair, 13.0, 1.0);
    } else {
        res.x = bound;
    }

    float wave = 0.03 * sin(uTime * 1.7);
    // Upper body: collar, chest, arms, hands, pauldrons, wrist cuffs.
    bound = length(pb - vec3(0.0, -0.20, 0.0)) - 0.95;
    if (bound < res.x) {
        float armor = sdRoundCone(pb, vec3(0.0, -0.02, 0.0), vec3(0.0, 0.20, -0.01), 0.15, 0.14);
        armor = smin(armor, sdEllipsoid(pb - vec3(0.0, -0.22, 0.0), vec3(0.27, 0.30, 0.17)), 0.06);
        armor = smin(armor, sdEllipsoid(pb - vec3(0.0, -0.60, 0.0), vec3(0.20, 0.18, 0.15)), 0.10);
        // The right arm (screen left) grips the staff, the left hangs down.
        armor = min(armor, smin(sdRoundCone(pb, vec3(-0.30, -0.08, 0.0), vec3(-0.46, -0.40, 0.10), 0.075, 0.06),
                                sdRoundCone(pb, vec3(-0.46, -0.40, 0.10), vec3(-0.44, -0.31, 0.26), 0.06, 0.05), 0.03));
        armor = min(armor, smin(sdRoundCone(pb, vec3(0.30, -0.08, 0.0), vec3(0.42, -0.46, 0.03), 0.075, 0.06),
                                sdRoundCone(pb, vec3(0.42, -0.46, 0.03), vec3(0.47, -0.75, 0.10 + wave), 0.06, 0.05), 0.03));
        if (armor < res.x) res = vec3(armor, 11.0, 0.0);
        float edge;
        float pa = magePauldron(qs, edge);
        if (pa < res.x) res = vec3(pa, 12.0, 0.0);
        float cuffs = min(sdTorus(pb - vec3(-0.45, -0.36, 0.20), 0.06, 0.015),
                          sdTorus(pb - vec3(0.46, -0.70, 0.09 + wave), 0.058, 0.015));
        if (cuffs < res.x) res = vec3(cuffs, 7.0, 0.0);
        float hands = min(sdEllipsoid(pb - vec3(-0.44, -0.30, 0.33), vec3(0.065, 0.075, 0.06)),
                          sdEllipsoid(pb - vec3(0.48, -0.82, 0.12 + wave), vec3(0.05, 0.08, 0.045)));
        if (hands < res.x) res = vec3(hands, 0.0, 0.0);
    } else {
        res.x = min(res.x, bound);
    }

    // Lower body: armored legs with pointed boots, knee spirals, and the robe
    // flaring from the waist, open at the front.
    bound = length(pb - vec3(0.0, -1.15, 0.0)) - 0.75;
    if (bound < res.x) {
        float legs = smin(sdRoundCone(qs, vec3(0.11, -0.68, 0.0), vec3(0.15, -1.13, 0.05), 0.10, 0.07),
                          sdRoundCone(qs, vec3(0.15, -1.13, 0.05), vec3(0.15, -1.55, 0.0), 0.07, 0.055), 0.03);
        legs = smin(legs, sdRoundCone(qs, vec3(0.15, -1.58, -0.03), vec3(0.17, -1.645, 0.27), 0.07, 0.006), 0.04);
        if (legs < res.x) res = vec3(legs, 11.0, 0.0);
        float knees = sdTorus((qs - vec3(0.16, -1.12, 0.115)).xzy, 0.045, 0.014);
        if (knees < res.x) res = vec3(knees, 7.0, 0.0);
        float ang = atan(pb.x, pb.z);                    // 0 = straight ahead
        float robe = sdRoundCone(pb, vec3(0.0, -0.58, 0.0), vec3(0.0, -1.50, -0.05), 0.23, 0.50);
        robe += 0.02 * sin(ang * 9.0 + 0.5) * smoothstep(-0.7, -1.5, pb.y);   // folds
        robe = abs(robe + 0.02) - 0.02;                  // a cloth shell, not a solid cone
        robe = max(robe, -(abs(ang) - mix(0.15, 0.95, smoothstep(-0.62, -1.35, pb.y))));
        robe = max(robe, (GROUND + 0.05) - pb.y);
        if (robe < res.x) res = vec3(robe, 10.0, 0.0);
    } else {
        res.x = min(res.x, bound);
    }

    // Teal staff with a leaf-shaped loop, a glowing orb and a spike on top.
    bound = sdCapsule(pb, vec3(MG_STAFF.x, GROUND, MG_STAFF.z), vec3(MG_STAFF.x, 1.40, MG_STAFF.z), 0.16);
    if (bound < res.x) {
        float staff = sdCapsule(pb, vec3(MG_STAFF.x, GROUND + 0.04, MG_STAFF.z), vec3(MG_STAFF.x, 0.92, MG_STAFF.z), 0.024);
        vec3 qo = pb - MG_LOOP;
        staff = min(staff, sdTorus(vec3(qo.x, qo.z, qo.y * 0.62), 0.085, 0.017));
        staff = min(staff, sdRoundCone(pb, MG_LOOP + vec3(0.0, 0.13, 0.0), MG_LOOP + vec3(0.0, 0.32, 0.0), 0.03, 0.004));
        if (staff < res.x) res = vec3(staff, 9.0, 0.0);
        float orb = length(qo) - 0.042;
        if (orb < res.x) res = vec3(orb, 8.0, 0.0);
    } else {
        res.x = min(res.x, bound);
    }
    return res;
}

// ------------------------------------------------------------------ duelist
// A card duelist: huge spiky black hair with magenta tips and blonde bangs,
// a blue school jacket with a tall collar over a black shirt, a choker, a
// chain with a gold pyramid pendant, and one hand holding up a card.
// Materials: 0 skin, 14 dark hair, 15 blonde hair, 16 black clothes,
// 17 blue jacket, 18 white lining, 19 silver, 20 gold, 21 card.
const vec3 DU_HAIR_C = vec3(0.0, 0.62, -0.10);
const int DU_SPIKES = 9;
const vec3 DU_TIPS[9] = vec3[9](
    vec3(0.00, 1.26, -0.28), vec3(0.34, 1.16, -0.26), vec3(-0.34, 1.16, -0.26),
    vec3(0.60, 0.95, -0.24), vec3(-0.60, 0.95, -0.24), vec3(0.72, 0.66, -0.22),
    vec3(-0.72, 0.66, -0.22), vec3(0.18, 0.98, -0.62), vec3(-0.18, 0.98, -0.62));

// Distance to the big spikes; t = how far along the nearest spike (0 root, 1 tip).
float duelSpikes(vec3 ph, out float t) {
    float d = 1e9;
    t = 0.0;
    for (int i = 0; i < DU_SPIKES; i++) {
        vec3 a = DU_HAIR_C, b = DU_TIPS[i];
        float di = sdRoundCone(ph, a, b, 0.23, 0.012);
        if (di < d) {
            vec3 ba = b - a;
            t = clamp(dot(ph - a, ba) / dot(ba, ba), 0.0, 1.0);
        }
        d = smin(d, di, 0.035);
    }
    return d;
}
float duelBangs(vec3 ph) {
    vec3 q = ph;
    q.x = abs(q.x);
    // Blonde spikes standing up at the front...
    float d = sdRoundCone(ph, vec3(0.0, 0.66, 0.14), vec3(0.0, 0.98, 0.14), 0.07, 0.006);
    d = min(d, sdRoundCone(q, vec3(0.09, 0.66, 0.13), vec3(0.20, 0.93, 0.12), 0.065, 0.006));
    // ...short lightning-bolt locks over the forehead...
    d = min(d, sdRoundCone(q, vec3(0.05, 0.70, 0.17), vec3(0.12, 0.585, 0.225), 0.045, 0.006));
    // ...and long locks framing the face.
    d = min(d, smin(sdRoundCone(q, vec3(0.18, 0.66, 0.11), vec3(0.235, 0.42, 0.12), 0.06, 0.035),
                    sdRoundCone(q, vec3(0.235, 0.42, 0.12), vec3(0.20, 0.16, 0.10), 0.035, 0.006), 0.02));
    return d;
}
float sdPyramid(vec3 p, float h) {
    // Square base 1x1 on y = 0, apex at y = h (Inigo Quilez).
    float m2 = h * h + 0.25;
    p.xz = abs(p.xz);
    p.xz = (p.z > p.x) ? p.zx : p.xz;
    p.xz -= 0.5;
    vec3 q = vec3(p.z, h * p.y - 0.5 * p.x, h * p.x + 0.5 * p.y);
    float s = max(-q.x, 0.0);
    float t = clamp((q.y - 0.5 * p.z) / (m2 + 0.25), 0.0, 1.0);
    float a = m2 * (q.x + s) * (q.x + s) + q.y * q.y;
    float b = m2 * (q.x + 0.5 * t) * (q.x + 0.5 * t) + (q.y - m2 * t) * (q.y - m2 * t);
    float d2 = min(q.y, -q.x * m2 - q.y * 0.5) > 0.0 ? 0.0 : min(a, b);
    return sqrt((d2 + q.z * q.z) / m2) * sign(max(q.z, -p.y));
}
const vec3 DU_PENDANT = vec3(0.0, -0.40, 0.22);     // top of the upside-down pyramid
const vec3 DU_CARD = vec3(-0.50, 0.22, 0.36);
vec3 mapDuelist(vec3 p) {
    vec3 ph = toHead(p);
    vec3 pb = toBody(p);
    vec3 qs = pb;
    qs.x = abs(qs.x);
    vec3 res = vec3(1e9, 0.0, 0.0);
    float bound;

    // Head and hair.
    bound = length(ph - vec3(0.0, 0.75, -0.15)) - 1.0;
    if (bound < res.x) {
        res = vec3(sdEllipsoid(ph - vec3(0.0, 0.39, 0.0), vec3(0.21, 0.29, 0.22)), 0.0, 1.0);
        float cap = sdEllipsoid(ph - vec3(0.0, 0.56, -0.07), vec3(0.28, 0.27, 0.28));
        cap = smax(cap, -sdEllipsoid(ph - vec3(0.0, 0.36, 0.22), vec3(0.21, 0.30, 0.26)), 0.02);
        float t;
        float hair = smin(cap, duelSpikes(ph, t), 0.06);
        if (hair < res.x) res = vec3(hair, 14.0, 1.0);
        float bangs = duelBangs(ph);
        if (bangs < res.x) res = vec3(bangs, 15.0, 1.0);
    } else {
        res.x = bound;
    }

    // Upper body: neck, choker, collar, shirt, open jacket, arms, pendant, card.
    bound = length(pb - vec3(0.0, -0.30, 0.0)) - 1.0;
    if (bound < res.x) {
        float neck = sdCapsule(pb, vec3(0.0, -0.05, 0.0), vec3(0.0, 0.25, 0.0), 0.11);
        if (neck < res.x) res = vec3(neck, 0.0, 0.0);
        float choker = sdTorus(pb - vec3(0.0, 0.08, 0.0), 0.115, 0.022);
        if (choker < res.x) res = vec3(choker, 16.0, 0.0);
        float metal = sdBox(pb - vec3(0.0, 0.08, 0.13), vec3(0.035, 0.025, 0.012)) - 0.005;   // buckle
        // Chain: two strands from the collar down to the pendant.
        metal = min(metal, sdCapsule(qs, vec3(0.10, 0.02, 0.09), vec3(0.015, DU_PENDANT.y + 0.02, DU_PENDANT.z), 0.011));
        if (metal < res.x) res = vec3(metal, 19.0, 0.0);
        vec3 qp = pb - DU_PENDANT;
        qp.y = -qp.y;
        float pendant = sdPyramid(qp / 0.20, 1.1) * 0.20;
        if (pendant < res.x) res = vec3(pendant, 20.0, 0.0);

        float shirt = sdEllipsoid(pb - vec3(0.0, -0.40, 0.0), vec3(0.28, 0.45, 0.18));
        if (shirt < res.x) res = vec3(shirt, 16.0, 0.0);
        // Jacket: a shell around the torso, open down the front.
        float jacket = sdEllipsoid(pb - vec3(0.0, -0.40, -0.01), vec3(0.34, 0.52, 0.225));
        jacket = smin(jacket, sdEllipsoid(qs - vec3(0.26, -0.08, 0.0), vec3(0.14, 0.12, 0.15)), 0.08);  // shoulders
        jacket = abs(jacket) - 0.012;
        float ang = atan(pb.x, pb.z);
        jacket = max(jacket, -(abs(ang) - mix(0.30, 0.55, smoothstep(-0.1, -0.9, pb.y))));
        jacket = max(jacket, (-0.98) - pb.y);
        // Tall standing collar, open at the front.
        vec3 qc = pb;
        qc.xz *= 1.0 - 0.9 * clamp(qc.y, 0.0, 0.3);           // flares outward toward the top
        float collar = abs(sdCapsule(qc, vec3(0.0, -0.02, 0.0), vec3(0.0, 0.30, -0.02), 0.16)) - 0.013;
        collar = max(collar, -(abs(ang) - 0.62));
        collar = max(collar, pb.y - (0.33 - 0.10 * smoothstep(0.4, 1.4, abs(ang))));
        float blue = min(jacket, collar);
        // Sleeves: the left arm hangs down, the right one holds up a card.
        float wave = 0.02 * sin(uTime * 1.5);
        blue = min(blue, smin(sdRoundCone(pb, vec3(0.30, -0.10, 0.0), vec3(0.40, -0.48, 0.02), 0.095, 0.08),
                              sdRoundCone(pb, vec3(0.40, -0.48, 0.02), vec3(0.44, -0.82, 0.08), 0.08, 0.07), 0.03));
        blue = min(blue, smin(sdRoundCone(pb, vec3(-0.30, -0.10, 0.0), vec3(-0.56, -0.26, 0.14), 0.095, 0.08),
                              sdRoundCone(pb, vec3(-0.56, -0.26, 0.14), DU_CARD + vec3(0.02, -0.20 + wave, -0.04), 0.08, 0.065), 0.03));
        if (blue < res.x) res = vec3(blue, 17.0, 0.0);
        float hands = min(sdEllipsoid(pb - vec3(0.45, -0.89, 0.10), vec3(0.055, 0.075, 0.05)),
                          sdEllipsoid(pb - (DU_CARD + vec3(0.02, -0.12 + wave, -0.03)), vec3(0.06, 0.07, 0.05)));
        if (hands < res.x) res = vec3(hands, 0.0, 0.0);
        float card = sdBox(rotZ(pb - (DU_CARD + vec3(0.0, wave, 0.0)), -0.15), vec3(0.085, 0.12, 0.004)) - 0.003;
        if (card < res.x) res = vec3(card, 21.0, 0.0);
    } else {
        res.x = min(res.x, bound);
    }

    // Legs and shoes.
    bound = length(pb - vec3(0.0, -1.30, 0.05)) - 0.55;
    if (bound < res.x) {
        float legs = smin(sdRoundCone(qs, vec3(0.13, -0.88, 0.0), vec3(0.15, -1.25, 0.02), 0.12, 0.095),
                          sdRoundCone(qs, vec3(0.15, -1.25, 0.02), vec3(0.15, -1.56, 0.0), 0.095, 0.08), 0.03);
        legs = min(legs, sdEllipsoid(qs - vec3(0.16, -1.60, 0.07), vec3(0.08, 0.06, 0.15)));
        if (legs < res.x) res = vec3(legs, 16.0, 0.0);
    } else {
        res.x = min(res.x, bound);
    }
    return res;
}

// ---------------------------------------------------------------- sea sponge
// A yellow, slightly wobbly sponge: white shirt with a red tie, brown square
// pants with a belt, thin arms from short sleeves, striped socks and shiny
// shoes. With your face off (or not found) it shows its own cartoon face.
// Materials: 0 sponge, 22 white, 23 pants, 25 black.
float spongeBox(vec3 ph) {
    vec3 q = ph - vec3(0.0, 0.15, 0.0);
    // Wobbly sides and top, like a real sponge.
    q.x += 0.018 * sin(q.y * 11.0 + 1.0) * smoothstep(0.35, 0.56, abs(q.x));
    q.y += 0.015 * sin(q.x * 13.0) * smoothstep(0.45, 0.68, q.y);
    return sdBox(q, vec3(0.56, 0.68, 0.16)) - 0.06;
}
vec3 mapSponge(vec3 p) {
    vec3 ph = toHead(p);
    vec3 pb = toBody(p);
    vec3 qs = pb;
    qs.x = abs(qs.x);
    vec3 res = vec3(1e9, 0.0, 0.0);
    float bound;

    // The sponge (head + chest) and its long nose.
    bound = length(ph - vec3(0.0, 0.15, 0.0)) - 1.0;
    if (bound < res.x) {
        float sponge = min(spongeBox(ph), sdCapsule(ph, vec3(0.0, 0.30, 0.16), vec3(0.0, 0.27, 0.42), 0.048));
        res = vec3(sponge, 0.0, 1.0);
    } else {
        res.x = bound;
    }

    // Shirt + pants block, sleeves, arms, legs, socks, shoes.
    bound = length(pb - vec3(0.0, -1.0, 0.0)) - 1.05;
    if (bound < res.x) {
        float clothes = sdBox(pb - vec3(0.0, -0.82, 0.0), vec3(0.53, 0.27, 0.15)) - 0.05;
        clothes = min(clothes, sdCapsule(qs, vec3(0.24, -1.02, 0.0), vec3(0.24, -1.18, 0.0), 0.10));   // pant legs
        if (clothes < res.x) res = vec3(clothes, pb.y > -0.77 ? 22.0 : 23.0, 0.0);
        float sleeves = sdEllipsoid(qs - vec3(0.63, -0.62, 0.0), vec3(0.11, 0.10, 0.10));
        float socks = sdCapsule(qs, vec3(0.24, -1.38, 0.0), vec3(0.24, -1.55, 0.0), 0.045);
        float white = min(sleeves, socks);
        if (white < res.x) res = vec3(white, 22.0, 0.0);
        float wave = 0.04 * sin(uTime * 2.0 + sign(pb.x));
        vec3 hand = vec3(0.86, -0.96 + wave, 0.10);
        float yellow = sdCapsule(qs, vec3(0.70, -0.64, 0.0), hand, 0.034);
        yellow = smin(yellow, sdEllipsoid(qs - hand - vec3(0.02, -0.05, 0.0), vec3(0.065, 0.08, 0.05)), 0.03);
        yellow = min(yellow, sdCapsule(qs, vec3(0.24, -1.15, 0.0), vec3(0.24, -1.42, 0.0), 0.034));   // legs
        if (yellow < res.x) res = vec3(yellow, 0.0, 0.0);
        float shoes = sdEllipsoid(qs - vec3(0.26, -1.595, 0.06), vec3(0.11, 0.07, 0.17));
        if (shoes < res.x) res = vec3(shoes, 25.0, 0.0);
    } else {
        res.x = min(res.x, bound);
    }
    return res;
}

// x = distance, y = material (0 skin/shell, 1 limbs, 2 gloves, 3 eyes), z = head weight
vec3 map(vec3 p) {
    if (uShape == 1) return mapLizard(p);
    if (uShape == 2) return mapMage(p);
    if (uShape == 3) return mapDuelist(p);
    if (uShape == 4) return mapSponge(p);
    vec3 ph = toHead(p);
    // A floating peanut: the lower lobe turns with the head too.
    vec3 pb = uHeadOnly == 1 ? ph : toBody(p);
    vec3 rh = uHead.w * uHeadScale * vec3(1.0 - 0.35 * uStretch, 1.0 + uStretch, 1.0 - 0.35 * uStretch);
    float dh = sdEllipsoid(ph - uHead.xyz, rh);
    float db = sdEllipsoid(pb - uBody.xyz, uBody.w * uBodyScale);
    float d = smin(dh, db, uBlend);
    float hw = uHeadOnly == 1 ? 1.0 : clamp(0.5 + (db - dh) * 4.0, 0.0, 1.0);
    if (uBumps > 0.0) d += uBumps * mix(bumps(pb), bumps(ph), hw);
    vec3 res = vec3(d, 0.0, hw);

    if (uHat == 1) {
        // Propeller beanie: a cap over the top of the head, a stick, a red hub
        // and two spinning blades.
        vec3 hc = uHead.xyz;
        float capY = hc.y + rh.y * 0.44;
        float cap = max(sdEllipsoid(ph - hc, rh + 0.035), capY - ph.y);
        float ringR = sqrt(max(0.0, 1.0 - 0.44 * 0.44));
        float brim = length(vec2(length((ph - hc).xz) - ringR * rh.x - 0.015, ph.y - capY)) - 0.05;
        float hat = min(cap, brim);
        if (hat < res.x) res = vec3(hat, 3.0, 0.0);
        float topY = hc.y + rh.y;
        float stick = sdCapsule(ph, vec3(hc.x, topY, hc.z), vec3(hc.x, topY + 0.15, hc.z), 0.025);
        float hub = length(ph - vec3(hc.x, topY + 0.17, hc.z)) - 0.055;
        float a = uTime * 7.0;
        vec3 pr = ph - vec3(hc.x, topY + 0.17, hc.z);
        pr.xz = mat2(cos(a), -sin(a), sin(a), cos(a)) * pr.xz;
        pr.y -= 0.03 * sin(pr.x * 6.0);                 // a little twist in the blade
        float blade = sdBox(pr, vec3(0.44, 0.012, 0.05)) - 0.01;
        float black = min(stick, blade);
        if (black < res.x) res = vec3(black, 4.0, 0.0);
        if (hub < res.x) res = vec3(hub, 5.0, 0.0);
    } else if (uHat == 2) {
        // Cowboy hat: a dented crown and a wide brim that curls up at the sides.
        vec3 hc = uHead.xyz;
        float brimY = hc.y + rh.y * 0.72;
        vec3 q = ph - vec3(hc.x, brimY, hc.z);
        q.y -= 0.30 * q.x * q.x;
        float brim = sdEllipsoid(q, vec3(rh.x * 1.65, 0.035, rh.z * 1.35));
        vec3 c = ph - vec3(hc.x, brimY + 0.22, hc.z);
        float crown = sdEllipsoid(c, vec3(rh.x * 0.80, 0.36, rh.z * 0.78));
        crown = smax(crown, -sdEllipsoid(c - vec3(0.0, 0.40, 0.0), vec3(0.12, 0.16, 0.9)), 0.05);
        float hat = min(brim, crown);
        if (hat < res.x) res = vec3(hat, 6.0, 0.0);
    }

    if (uLimbs == 1 && uHeadOnly == 0) {
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
// Cylindrical wrap of the shell photo: ridges run along the peanut.
vec2 shellUV(vec3 q) {
    return vec2(atan(q.x, q.z) / 6.2831853 * uShellRepeat.x, q.y * uShellRepeat.y);
}
float shellLum(vec2 uv) { return dot(texture(uShellTex, uv).rgb, vec3(0.3, 0.59, 0.11)); }

vec3 shell(vec3 q) {
    float burn = smoothstep(0.38, 0.72, fbm(q * 2.2 + 3.1)) * uBurn;
    float speck = smoothstep(0.6, 0.78, fbm(q * 9.0)) * uBurn;
    vec3 c;
    if (uShellOn == 1) {
        c = texture(uShellTex, shellUV(q)).rgb * uShellTint;
        c *= 0.90 + 0.20 * fbm(q * 16.0);          // fine grain the photo is too soft to carry
        c = mix(c, uDark, clamp(burn + 0.6 * speck, 0.0, 1.0));
    } else {
        c = mix(uBase, uDark, clamp(burn + 0.6 * speck, 0.0, 1.0));
        c *= 0.88 + 0.24 * fbm(q * 6.0);
    }
    if (uCrack > 0.0) {
        // Dry, dark cracks: thin cell edges, broken up so they don't read as a grid.
        float e = voronoiEdge(q * vec3(3.2, 2.4, 3.2) + 0.35 * fbm(q * 3.0));
        float crack = (1.0 - smoothstep(0.0, 0.025, e)) * smoothstep(0.35, 0.6, fbm(q * 5.0 + 7.0));
        c = mix(c, uDark * 0.6, crack * uCrack);
    }
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

vec3 mageAlbedo(vec3 pos, vec3 n, vec3 m, out float gloss, out vec3 emit) {
    const vec3 ARMOR = vec3(0.44, 0.15, 0.64);
    const vec3 TRIM = vec3(0.92, 0.60, 0.96);
    emit = vec3(0.0);
    vec3 ph = toHead(pos);
    vec3 pb = toBody(pos);
    if (m.y > 12.5) { gloss = 0.3; return vec3(0.18, 0.09, 0.30); }                  // hair
    if (m.y > 11.5) {                                                                // pauldrons
        gloss = 1.3;
        vec3 qs = pb;
        qs.x = abs(qs.x);
        float edge;
        magePauldron(qs, edge);
        return mix(ARMOR, TRIM, smoothstep(0.80, 0.86, edge));
    }
    if (m.y > 10.5) {                                                                // armor
        gloss = 1.2;
        float band = 0.0;
        bool limb = pb.y < -0.72 || abs(pb.x) > 0.29;
        if (limb) band = 1.0 - smoothstep(0.05, 0.09, abs(fract(pb.y * 5.5) - 0.5));
        else {
            float v = abs(pb.y - (0.02 - 1.1 * (0.20 - min(abs(pb.x), 0.20))));   // V-neck
            band = max(1.0 - smoothstep(0.008, 0.016, abs(pb.y + 0.50)),        // chest plate edge
                       (1.0 - smoothstep(0.008, 0.016, v)) * step(0.0, pb.z) * step(abs(pb.x), 0.2));
        }
        return mix(ARMOR, TRIM, band);
    }
    if (m.y > 9.5) {                                                                 // robe
        gloss = 0.6;
        return vec3(0.25, 0.15, 0.55) * (0.85 + 0.3 * fbm(pos * 4.0));
    }
    if (m.y > 8.5) {                                                                 // staff
        gloss = 1.6;
        float band = 1.0 - smoothstep(0.04, 0.08, abs(fract(pb.y * 3.0) - 0.5));
        return mix(vec3(0.16, 0.70, 0.68), vec3(0.70, 0.96, 0.92), band * step(pb.y, 0.9));
    }
    if (m.y > 7.5) {                                                                 // glowing orb
        gloss = 2.0;
        emit = vec3(0.35, 0.50, 0.08) * (0.8 + 0.2 * sin(uTime * 3.0));
        return vec3(0.80, 0.95, 0.30);
    }
    if (m.y > 6.5) { gloss = 1.3; return TRIM; }                                     // trim
    if (m.y > 5.5) {                                                                 // hat
        gloss = 1.3;
        float a = atan(ph.x, ph.z) / 6.2832;
        float spiral = 1.0 - smoothstep(0.05, 0.09, abs(fract(ph.y * 2.6 - a) - 0.5));
        return mix(ARMOR, TRIM, spiral * smoothstep(0.66, 0.72, ph.y));
    }
    gloss = 0.2;
    return vec3(0.95, 0.80, 0.66);                                                   // skin
}

vec3 duelistAlbedo(vec3 pos, vec3 n, vec3 m, out float gloss) {
    vec3 ph = toHead(pos);
    vec3 pb = toBody(pos);
    if (m.y > 20.5) {                                                        // card
        gloss = 0.5;
        vec3 q = rotZ(pb - (DU_CARD + vec3(0.0, 0.02 * sin(uTime * 1.5), 0.0)), -0.15);
        if (q.z < 0.0) return vec3(0.45, 0.28, 0.14);                        // card back
        vec2 a = abs(q.xy - vec2(0.0, 0.015));
        bool art = a.x < 0.065 && a.y < 0.055;
        vec3 c = vec3(0.80, 0.62, 0.32);                                     // gold-brown frame
        if (art) c = mix(vec3(0.20, 0.35, 0.55), vec3(0.45, 0.20, 0.65), smoothstep(-0.05, 0.08, q.y));
        if (!art && q.y < -0.05 && a.x < 0.07) c = vec3(0.90, 0.80, 0.60);   // text box
        return c;
    }
    if (m.y > 19.5) {                                                        // gold pendant with an eye
        gloss = 1.6;
        vec3 q = pb - DU_PENDANT;
        float eye = abs(length((q.xy - vec2(0.0, -0.07)) * vec2(1.0, 2.2)) - 0.035);
        float pupil = length(q.xy - vec2(0.0, -0.07)) - 0.012;
        float mark = max(1.0 - smoothstep(0.004, 0.008, eye), 1.0 - smoothstep(0.0, 0.004, pupil));
        return mix(vec3(0.95, 0.75, 0.25), vec3(0.35, 0.22, 0.05), mark * step(0.0, q.z));
    }
    if (m.y > 18.5) { gloss = 1.8; return vec3(0.78, 0.80, 0.84); }         // silver
    if (m.y > 17.5) { gloss = 0.3; return vec3(0.95, 0.95, 0.97); }         // white
    if (m.y > 16.5) {                                                        // blue jacket
        gloss = 0.35;
        // The inside of the collar and jacket front is lined in white.
        vec3 nb = uBodyInv * n;
        vec3 radial = normalize(vec3(pb.x, 0.0, pb.z) + 1e-5);
        bool inner = dot(nb, radial) < -0.15 && abs(pb.x) < 0.36 && (pb.y > -0.02 || pb.z > 0.05);
        return inner ? vec3(0.95, 0.95, 0.97) : vec3(0.16, 0.26, 0.68);
    }
    if (m.y > 15.5) { gloss = 0.25; return vec3(0.07, 0.07, 0.09); }        // black clothes
    if (m.y > 14.5) { gloss = 0.4; return vec3(0.98, 0.82, 0.32); }         // blonde
    if (m.y > 13.5) {                                                        // spiky hair
        gloss = 0.4;
        float t;
        duelSpikes(ph, t);
        return mix(vec3(0.07, 0.05, 0.09), vec3(0.72, 0.08, 0.38), smoothstep(0.42, 0.58, t));
    }
    gloss = 0.2;
    return vec3(0.96, 0.82, 0.70);                                           // skin
}

// 2D helpers for the sponge's drawn-on details.
float segDist(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    return length(pa - ba * clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0));
}
float spongePores(vec3 q) {
    // Distance-like value: < 0 inside one of the round pores.
    vec3 n = floor(q), f = fract(q);
    float d = 1.0;
    for (int k = -1; k <= 1; k++)
    for (int j = -1; j <= 1; j++)
    for (int i = -1; i <= 1; i++) {
        vec3 g = vec3(i, j, k);
        vec3 h = hash3(n + g);
        d = min(d, length(g + h - f) - (0.12 + 0.16 * h.x) * step(0.30, h.y));
    }
    return d;
}
vec3 spongeFace(vec2 f, vec3 col) {
    // Big eyes with lashes, rosy freckled cheeks and a buck-toothed grin.
    const vec3 INK = vec3(0.05, 0.04, 0.03);
    for (int i = 0; i < 2; i++) {
        float s = i == 0 ? -1.0 : 1.0;
        vec2 e = vec2(s * 0.21, 0.43);
        float d = length(f - e);
        float blink = s < 0.0 ? uBlink.x : uBlink.y;
        float lid = e.y + 0.19 - 0.40 * blink;
        vec2 lash0 = e + 0.19 * vec2(s * 0.50, 0.866), lash1 = e + vec2(0.0, 0.19), lash2 = e + 0.19 * vec2(-s * 0.50, 0.866);
        float lashes = min(segDist(f, lash0, lash0 + vec2(s * 0.05, 0.06)),
                       min(segDist(f, lash1, lash1 + vec2(0.0, 0.075)), segDist(f, lash2, lash2 + vec2(-s * 0.03, 0.065))));
        if (blink < 0.6 && lashes < 0.011) col = INK;
        if (d < 0.19) {
            vec2 ir = e + vec2(s * 0.02, -0.01);
            float di = length(f - ir);
            col = vec3(0.98);
            if (di < 0.09) col = mix(vec3(0.20, 0.50, 0.90), vec3(0.45, 0.75, 1.0), smoothstep(0.09, 0.03, di));
            if (abs(di - 0.09) < 0.007) col = vec3(0.10, 0.25, 0.50);
            if (di < 0.04) col = INK;
            if (length(f - ir - vec2(-0.02, 0.025)) < 0.014) col = vec3(1.0);
            if (f.y > lid) col = vec3(1.0, 0.90, 0.25);                       // eyelid
            if (blink > 0.05 && abs(f.y - lid) < 0.008) col = INK;
        }
        if (abs(d - 0.19) < 0.009) col = INK;
        // Cheek with freckles.
        vec2 c = vec2(s * 0.37, 0.10);
        float dc = length(f - c);
        if (dc < 0.075) {
            col = vec3(0.95, 0.50, 0.40);
            for (int k = 0; k < 3; k++) {
                vec2 fr = c + 0.035 * vec2(cos(2.1 * float(k) + 0.4), sin(2.1 * float(k) + 0.4));
                if (length(f - fr) < 0.009) col = vec3(0.65, 0.20, 0.15);
            }
        }
        if (abs(dc - 0.075) < 0.006) col = vec3(0.75, 0.30, 0.20);
    }
    // Grin: opens with your jaw, tongue pokes out with yours.
    float jaw = max(uJaw, 0.4 * uTongue);
    float x = f.x;
    if (abs(x) < 0.31) {
        float top = 0.08 + 0.95 * x * x;
        float bot = top - jaw * 0.30 * (1.0 - pow(x / 0.31, 2.0));
        if (f.y < top && f.y > bot) {
            col = vec3(0.40, 0.06, 0.10);
            if (f.y < bot + 0.09 * jaw) col = vec3(0.95, 0.45, 0.55);          // tongue
        }
        if (abs(f.y - top) < 0.010 || (jaw > 0.05 && abs(f.y - bot) < 0.010)) col = INK;
        // Two big front teeth hanging from the top lip.
        float tx = abs(x) - 0.042;
        if (abs(tx) < 0.036 && f.y < top && f.y > top - 0.095) {
            col = vec3(0.98);
            if (abs(tx) > 0.029 || f.y < top - 0.088) col = INK;
        }
    }
    if (uTongue > 0.05) {
        float bot0 = 0.08 - jaw * 0.30;
        vec2 tc = vec2(0.0, bot0 - 0.07 * uTongue);
        float dt = length((f - tc) / vec2(0.10, 0.07 + 0.05 * uTongue));
        if (dt < 1.0) col = mix(vec3(0.95, 0.45, 0.55), vec3(0.80, 0.30, 0.40), step(abs(f.x), 0.006));
        if (abs(dt - 1.0) < 0.08 && f.y < bot0) col = INK;
    }
    // Little dimples at the ends of the smile.
    for (int i = 0; i < 2; i++) {
        float s = i == 0 ? -1.0 : 1.0;
        vec2 dc = vec2(s * 0.34, 0.20);
        if (abs(length(f - dc) - 0.045) < 0.008 && f.y < 0.20 && s * (f.x - dc.x) < 0.0) col = INK;
    }
    return col;
}
vec3 spongeAlbedo(vec3 pos, vec3 n, vec3 m, out float gloss) {
    vec3 ph = toHead(pos);
    vec3 pb = toBody(pos);
    gloss = 0.2;
    if (m.y > 24.5) { gloss = 1.8; return vec3(0.04, 0.04, 0.05); }         // shoes
    if (m.y > 22.5) {                                                       // pants + belt
        vec3 c = vec3(0.55, 0.32, 0.12);
        if (abs(pb.y + 0.80) < 0.025 && abs(pb.x) < 0.58) {
            c = vec3(0.06, 0.05, 0.05);
            if (abs(abs(pb.x) - 0.18) < 0.03 || abs(pb.x) < 0.02) c = vec3(0.55, 0.32, 0.12);   // belt loops
        }
        return c;
    }
    if (m.y > 21.5) {                                                       // white shirt, socks
        vec3 c = vec3(0.97, 0.97, 0.95);
        vec3 nb = uBodyInv * n;
        if (pb.y < -1.3) {                                                  // sock stripes
            float y = pb.y;
            if (abs(y + 1.42) < 0.012) c = vec3(0.85, 0.15, 0.15);
            if (abs(y + 1.46) < 0.012) c = vec3(0.15, 0.35, 0.85);
        } else if (nb.z > 0.5 && pb.y > -0.78) {
            // Collar points and a red tie.
            vec2 f = pb.xy - vec2(0.0, -0.53);
            float collar = min(segDist(f, vec2(-0.16, 0.0), vec2(-0.05, -0.08)), segDist(f, vec2(0.16, 0.0), vec2(0.05, -0.08)));
            if (collar < 0.008) c = vec3(0.3);
            vec2 t = f - vec2(0.0, -0.10);
            float knot = length(t / vec2(0.035, 0.03));
            float tie = abs(t.x) - 0.045 * clamp((-t.y) / 0.06, 0.0, 1.0) * (1.0 - clamp((-t.y - 0.08) / 0.07, 0.0, 1.0));
            if (knot < 1.0 || (tie < 0.0 && t.y < 0.0 && t.y > -0.15)) c = vec3(0.85, 0.12, 0.12);
        }
        return c;
    }
    // Yellow sponge with olive pores; its own face when yours isn't shown.
    vec3 c = vec3(1.0, 0.90, 0.25);
    vec3 q = m.z > 0.5 ? ph : pb;
    float pore = spongePores(q * 5.5);
    if (pore < 0.0) c = mix(vec3(0.62, 0.66, 0.10), vec3(0.78, 0.76, 0.16), smoothstep(-0.12, 0.0, pore));
    vec3 nh = uHeadInv * n;
    if (m.z > 0.5 && nh.z > 0.6 && ph.z > 0.15 && ph.y > -0.5) {
        vec3 face = spongeFace(ph.xy, c);
        c = mix(c, face, 1.0 - uFaceAlpha);
    }
    return c;
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
    float bR = uShape == 1 ? 3.2 : (uShape >= 2 ? 2.5 : 2.1);
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
            if (uShellOn == 1 && m.y < 0.5 && uShellBump > 0.0) {
                // Tilt the normal by the brightness slope of the shell photo so
                // the pits look sunken.
                vec3 q = mix(toBody(pos), ph, m.z);
                vec2 uv = shellUV(q);
                float ang = atan(q.x, q.z);
                vec2 e = vec2(0.003, 0.0);
                float l0 = shellLum(uv);
                float du = (shellLum(uv + e.xy) - l0) / e.x;
                float dv = (shellLum(uv + e.yx) - l0) / e.x;
                mat3 headToWorld = transpose(uHeadInv);
                vec3 tu = headToWorld * vec3(cos(ang), 0.0, -sin(ang));
                vec3 tv = headToWorld * vec3(0.0, 1.0, 0.0);
                n = normalize(n - uShellBump * 0.02 * (du * tu + dv * tv));
            }
            vec3 alb;
            float gloss = uGloss;
            float faceM = 0.0;
            vec3 emit = vec3(0.0);
            vec3 faceC = vec3(0.0);

            if (uShape == 4) {
                alb = spongeAlbedo(pos, n, m, gloss);
            } else if (uShape == 3) {
                alb = duelistAlbedo(pos, n, m, gloss);
            } else if (uShape == 2) {
                alb = mageAlbedo(pos, n, m, gloss, emit);
            } else if (uShape == 1) {
                alb = lizardAlbedo(pos, n, m, gloss);
            } else if (m.y < 0.5) {
                alb = shell(mix(toBody(pos), ph, m.z));
            } else if (m.y < 1.5) {
                alb = uLimbCol; gloss = 0.5;
            } else if (m.y < 2.5) {
                alb = uGloveCol; gloss = 0.3;
            } else if (m.y < 3.5) {
                // Rainbow beanie: orange, yellow, green and purple panels.
                vec3 hp = ph - uHead.xyz;
                float ang = atan(hp.x, hp.z) / 6.2831853 + 0.5 + 0.125;
                int panel = int(floor(fract(ang) * 8.0)) % 4;
                alb = panel == 0 ? vec3(0.95, 0.40, 0.10) : panel == 1 ? vec3(0.98, 0.80, 0.15)
                    : panel == 2 ? vec3(0.20, 0.62, 0.30) : vec3(0.45, 0.16, 0.65);
                gloss = 0.15;
            } else if (m.y < 4.5) {
                alb = vec3(0.04, 0.04, 0.05); gloss = 0.6;
            } else if (m.y < 5.5) {
                alb = vec3(0.85, 0.10, 0.10); gloss = 0.8;
            } else {
                // Cowboy hat: black felt with a slightly lighter band around the crown.
                float hy = ph.y - uHead.y - uHead.w * uHeadScale.y * 0.72;
                float band = smoothstep(0.03, 0.05, hy) * (1.0 - smoothstep(0.11, 0.13, hy))
                           * step(0.05, length((ph - uHead.xyz).xz) - 0.3);
                alb = mix(vec3(0.05, 0.05, 0.06), vec3(0.16, 0.14, 0.13), band);
                gloss = 0.2;
            }
            if (m.y < 0.5) {
                vec3 nh = uHeadInv * n;
                vec2 fuv = vec2(0.5 + (ph.x - uHead.x) / uFaceSize, 0.5 - (ph.y - uFaceY) / uFaceSize);
                if (all(greaterThan(fuv, vec2(0.0))) && all(lessThan(fuv, vec2(1.0)))) {
                    // Blow up the eyes and the mouth on the shell (the streamer look).
                    fuv = magnify(fuv, uEyeL, 0.15, uEyeScale, vec2(-uEyeSpread, 0.0));
                    fuv = magnify(fuv, uEyeR, 0.15, uEyeScale, vec2(uEyeSpread, 0.0));
                    fuv = magnify(fuv, uMouth, 0.19, uMouthScale, vec2(0.0));
                    faceM = texture(uMask, fuv).r * smoothstep(0.1, 0.5, nh.z) * m.z * uFaceAlpha;
                    faceM *= mix(uFaceOpacity, 1.0, texture(uFeatMask, fuv).r);
                    faceC = texture(uFace, fuv).rgb * mix(vec3(1.0), uTint, uTintAmt);
                    if (uLipAmt > 0.0) {
                        float lip = texture(uLipMask, fuv).r * uLipAmt;
                        float lum = dot(faceC, vec3(0.3, 0.59, 0.11));
                        faceC = mix(faceC, uLipTint * (0.3 + 1.1 * lum), lip);
                    }
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
            col = mix(shaded, faceLit, faceM) + emit;
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
        self.supersample = supersample
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
        self.lip_tex = ctx.texture((MASK_TEX_SIZE, MASK_TEX_SIZE), 1)
        self.lip_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.lip_tex.repeat_x = self.lip_tex.repeat_y = False
        self.lip_tex.write(np.zeros((MASK_TEX_SIZE, MASK_TEX_SIZE), np.uint8).tobytes())
        self.feat_tex = ctx.texture((MASK_TEX_SIZE, MASK_TEX_SIZE), 1)
        self.feat_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.feat_tex.repeat_x = self.feat_tex.repeat_y = False
        self.feat_tex.write(np.zeros((MASK_TEX_SIZE, MASK_TEX_SIZE), np.uint8).tobytes())
        self.shell_textures: dict[str, moderngl.Texture] = {}
        self.shell_tex = None

        self.unwrap_prog = ctx.program(vertex_shader=UNWRAP_VS, fragment_shader=UNWRAP_FS)
        self.unwrap_vbo = None
        self.unwrap_vao = None

        self.scene_prog = ctx.program(vertex_shader=FULLSCREEN_VS, fragment_shader=SCENE_FS)
        self.scene_vao = ctx.vertex_array(self.scene_prog, [])
        self._make_scene_target()

        self.resolve_prog = ctx.program(vertex_shader=FULLSCREEN_VS, fragment_shader=RESOLVE_FS)
        self.resolve_vao = ctx.vertex_array(self.resolve_prog, [])
        self.out_fbo = ctx.simple_framebuffer(self.size, components=3)

        self.scene_prog["uFace"].value = 0
        self.scene_prog["uMask"].value = 1
        self.scene_prog["uCam"].value = 2
        self.scene_prog["uBgTex"].value = 4
        self.scene_prog["uLipMask"].value = 5
        self.scene_prog["uFeatMask"].value = 6
        self.scene_prog["uShellTex"].value = 7
        self.bg_tex = None
        self.bg_aspect = 16 / 9
        self.unwrap_prog["uCam"].value = 2
        self.resolve_prog["uSrc"].value = 3

    def _make_scene_target(self):
        self.scene_tex = self.ctx.texture(self.ss_size, 3)
        self.scene_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.scene_fbo = self.ctx.framebuffer([self.scene_tex])

    def set_supersample(self, supersample: float):
        """Change the internal render resolution (quality vs. speed)."""
        if abs(supersample - self.supersample) < 1e-3:
            return
        self.supersample = supersample
        w, h = self.size
        self.ss_size = (max(1, int(w * supersample)), max(1, int(h * supersample)))
        self.scene_fbo.release()
        self.scene_tex.release()
        self._make_scene_target()

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

    def set_shell_texture(self, path: str | None):
        """Wrap a photo of a shell around the character (None = procedural shell)."""
        if path is None:
            self.shell_tex = None
            return
        tex = self.shell_textures.get(path)
        if tex is None:
            img = cv2.imread(path)
            if img is None:
                raise RuntimeError(f"Could not read shell texture {path}")
            rgb = np.ascontiguousarray(img[:, :, ::-1])
            tex = self.ctx.texture((rgb.shape[1], rgb.shape[0]), 3, rgb.tobytes(), alignment=1)
            tex.build_mipmaps()
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            tex.repeat_x = tex.repeat_y = True
            self.shell_textures[path] = tex
        self.shell_tex = tex

    def update_face(self, face: FaceState, mask: np.ndarray, triangles: np.ndarray,
                    lip_mask: np.ndarray | None = None, feat_mask: np.ndarray | None = None):
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
        if lip_mask is not None:
            self.lip_tex.write(np.ascontiguousarray(lip_mask).tobytes())
        if feat_mask is not None:
            self.feat_tex.write(np.ascontiguousarray(feat_mask).tobytes())

    # ----------------------------------------------------------------- render
    def render(self, uniforms: dict) -> np.ndarray:
        """Render one frame, returns an (H, W, 3) BGR uint8 image."""
        prog = self.scene_prog
        uniforms = dict(uniforms)
        uniforms["uRes"] = self.ss_size
        uniforms["uCamAspect"] = self.cam_aspect
        uniforms["uBgAspect"] = self.bg_aspect
        uniforms["uShellOn"] = int(self.shell_tex is not None)
        for name, value in uniforms.items():
            if name in prog:
                prog[name].value = value

        self.face_tex.use(0)
        self.mask_tex.use(1)
        if self.cam_tex is not None:
            self.cam_tex.use(2)
        if self.bg_tex is not None:
            self.bg_tex.use(4)
        self.lip_tex.use(5)
        self.feat_tex.use(6)
        if self.shell_tex is not None:
            self.shell_tex.use(7)
        self.scene_fbo.use()
        self.scene_vao.render(moderngl.TRIANGLES, vertices=3)

        self.out_fbo.use()
        self.scene_tex.use(3)
        self.resolve_prog["uOutRes"].value = self.size
        self.resolve_vao.render(moderngl.TRIANGLES, vertices=3)

        data = self.out_fbo.read(components=3, alignment=1)
        img = np.frombuffer(data, np.uint8).reshape(self.size[1], self.size[0], 3)
        return np.ascontiguousarray(img[::-1, :, ::-1])  # GL is bottom-up RGB
