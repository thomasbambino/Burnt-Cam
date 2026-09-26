"""PeanutCam - put your real face on a 3D character and use it as a webcam.

    webcam -> MediaPipe face tracking -> 3D character (GPU) -> virtual camera

Pick "OBS Virtual Camera" as your camera in Zoom / Discord / Teams / OBS.
Run ``python peanutcam.py --help`` for options.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import cv2
import numpy as np

from renderer import Renderer
from skins import SKINS
from tracker import FaceTracker

HERE = os.path.dirname(os.path.abspath(__file__))
BACKGROUNDS = ["studio", "green", "webcam"]
MASKS = ["features", "full"]

# Image space (x right, y down, z away) -> render space (x right, y up, z toward viewer).
FLIP = np.diag([1.0, -1.0, -1.0])

NECK = np.array([0.0, 0.02, 0.0])
PIVOT = np.array([0.0, -1.66, 0.0])  # the feet; the body leans around them


# ---------------------------------------------------------------- video input
class CameraSource:
    """Grabs frames on a background thread so we always process the newest one."""

    def __init__(self, index: int, width: int, height: int, fps: int):
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam #{index}. Try --camera 1 (or 2).")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.frame = None
        self.seq = 0
        self.cond = threading.Condition()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.cond:
                self.frame = frame
                self.seq += 1
                self.cond.notify_all()

    def read(self, last_seq: int):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last_seq or not self.running, timeout=1.0)
            if self.frame is None:
                return None, last_seq
            return cv2.flip(self.frame, 1), self.seq  # mirror, like a selfie cam

    def close(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


class FileSource:
    """Image or video file instead of a webcam (handy for testing)."""

    def __init__(self, path: str):
        self.image = cv2.imread(path)
        self.cap = None if self.image is not None else cv2.VideoCapture(path)
        if self.image is None and not self.cap.isOpened():
            raise RuntimeError(f"Could not read {path}")
        self.seq = 0

    def read(self, last_seq: int):
        self.seq += 1
        if self.image is not None:
            return self.image.copy(), self.seq
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return (frame if ok else None), self.seq

    def close(self):
        if self.cap is not None:
            self.cap.release()


# ------------------------------------------------------------------ pose math
def rot_scale(R: np.ndarray, k: float) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(R)
    out, _ = cv2.Rodrigues(rvec * k)
    return out


def rot_clamp(R: np.ndarray, max_deg: float) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(R)
    ang = np.linalg.norm(rvec)
    lim = np.radians(max_deg)
    if ang > lim:
        rvec *= lim / ang
    return cv2.Rodrigues(rvec)[0]


def mat3(M: np.ndarray) -> tuple:
    """numpy 3x3 -> GLSL mat3 uniform (column-major)."""
    return tuple(float(v) for v in M.T.flatten())


class Pose:
    """Turns raw head rotation into smoothed character motion."""

    def __init__(self, follow: float, pitch_offset: float):
        self.follow = follow
        self.neutral = cv2.Rodrigues(np.array([np.radians(pitch_offset), 0.0, 0.0]))[0]
        self.rvec = np.zeros(3)
        self.offset = np.zeros(3)
        self.stretch = 0.0
        self.last_raw = None

    def calibrate(self):
        if self.last_raw is not None:
            self.neutral = self.last_raw.copy()

    def reset(self):
        self.neutral = np.eye(3)

    def update(self, face, dt: float):
        a = 1.0 - np.exp(-dt * 18.0)  # ~55 ms time constant
        if face is not None:
            R = FLIP @ face.rotation @ FLIP
            self.last_raw = R
            rel = rot_clamp(R @ self.neutral.T, 55.0)
            target_r = cv2.Rodrigues(rel)[0].ravel()
            cx, cy = face.center
            target_o = np.array([(cx - 0.5) * 2.2, -(cy - 0.45) * 0.8, 0.0]) * self.follow
            target_s = 0.08 * np.clip(face.jaw_open * 1.4 - 0.1, 0.0, 1.0)
        else:
            target_r = self.rvec * 0.9
            target_o = self.offset * 0.9
            target_s = 0.0
        self.rvec += (target_r - self.rvec) * a
        self.offset += (target_o - self.offset) * a
        self.stretch += (target_s - self.stretch) * min(1.0, a * 1.5)

    def uniforms(self, t: float) -> dict:
        Rh = cv2.Rodrigues(self.rvec)[0]
        Rb = rot_scale(Rh, 0.3)
        neck_w = Rb @ (NECK - PIVOT) + PIVOT
        bob = 0.015 * np.sin(t * 2.0)
        return {
            "uHeadInv": mat3(Rh.T),
            "uBodyInv": mat3(Rb.T),
            "uNeck": tuple(NECK),
            "uNeckW": tuple(neck_w),
            "uPivot": tuple(PIVOT),
            "uOffset": tuple(self.offset + np.array([0.0, bob, 0.0])),
            "uStretch": float(self.stretch),
        }


# Where the eyes and mouth corners land in the face texture (0..1), measured
# from the tracker's frontal layout. Used to line up eye domes and the grin.
EYE_TEX = (0.14, 0.35)    # (distance from center, y)
MOUTH_TEX_Y = 0.67


def eye_dome(skin: dict) -> tuple:
    """Put bulging eyes under where the user's eyes are painted, nudged apart."""
    r = skin.get("eye_domes", 0.0)
    if not r:
        return (0.0, 0.0, 0.0, 0.0)
    cx, cy, cz, rad = skin["head"]
    rx, ry, rz = (rad * s for s in skin["head_scale"])
    x = cx + EYE_TEX[0] * skin["face_size"] * 1.15
    y = skin["face_y"] + (0.5 - EYE_TEX[1]) * skin["face_size"]
    inside = max(0.0, 1.0 - ((x - cx) / rx) ** 2 - ((y - cy) / ry) ** 2)
    z = cz + rz * np.sqrt(inside) - 0.35 * r
    return (x, y, z, r)


def skin_uniforms(skin: dict) -> dict:
    mouth_y = skin["face_y"] + (0.5 - MOUTH_TEX_Y) * skin["face_size"]
    return {
        "uHead": skin["head"],
        "uHeadScale": skin["head_scale"],
        "uBody": skin["body"],
        "uBodyScale": skin["body_scale"],
        "uBlend": skin["blend"],
        "uBumps": skin["bumps"],
        "uLimbs": int(skin["limbs"]),
        "uLimbR": skin.get("limb_radius", 0.045),
        "uEyeDome": eye_dome(skin),
        "uTail": int(skin.get("tail", False)),
        "uToes": int(skin.get("toes", False)),
        "uNetScale": skin.get("net_scale", 1.0),
        "uBelly": skin.get("belly", (1.0, 1.0, 1.0)),
        "uBellyAmt": skin.get("belly_amount", 0.0),
        "uMouthLine": (mouth_y, 0.35, skin.get("mouth_line", 0.0)),
        "uBase": skin["base"],
        "uDark": skin["dark"],
        "uLine": skin["line"],
        "uLimbCol": skin["limb_color"],
        "uGloveCol": skin["glove_color"],
        "uBurn": skin["burn"],
        "uNet": skin["net"],
        "uGloss": skin["gloss"],
        "uFaceSize": skin["face_size"],
        "uFaceY": skin["face_y"],
        "uTint": skin["face_tint"],
        "uTintAmt": skin["tint_amount"],
    }


# ------------------------------------------------------------ virtual camera
def open_virtual_camera(args):
    try:
        import pyvirtualcam
    except ImportError:
        print("[peanutcam] pyvirtualcam not installed - preview only.")
        return None
    try:
        cam = pyvirtualcam.Camera(
            args.width, args.height, args.fps,
            fmt=pyvirtualcam.PixelFormat.RGB, device=args.device,
        )
    except Exception as e:  # noqa: BLE001
        print(
            "[peanutcam] Could not start the virtual camera:\n"
            f"    {e}\n"
            "  -> Install OBS Studio (https://obsproject.com). PeanutCam sends video to\n"
            "     the 'OBS Virtual Camera' device. Make sure OBS itself is NOT running\n"
            "     its own virtual camera at the same time.\n"
            "  Continuing in preview-only mode."
        )
        return None
    print(f"[peanutcam] Virtual camera running: {cam.device}  "
          f"-> select it as your camera in Zoom/Discord/Teams.")
    return cam


# ----------------------------------------------------------------------- HUD
HELP = [
    "1-9 / N : skin",
    "M : face mask (features / full)",
    "B : background",
    "C : calibrate (look straight, press C)",
    "R : reset calibration",
    "P : webcam picture-in-picture",
    "H : hide this help",
    "Q / Esc : quit",
]


def draw_hud(img, state, skin_name, fps):
    lines = [f"{skin_name} | mask: {state['mask']} | bg: {state['bg']} | {fps:4.1f} fps"]
    if state["help"]:
        lines += HELP
    y = 26
    for line in lines:
        cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22


def draw_pip(img, frame, face_found):
    h, w = img.shape[:2]
    pw = w // 4
    ph = int(frame.shape[0] * pw / frame.shape[1])
    small = cv2.resize(frame, (pw, ph))
    color = (0, 200, 0) if face_found else (0, 0, 255)
    cv2.rectangle(small, (0, 0), (pw - 1, ph - 1), color, 2)
    img[h - ph - 10:h - 10, w - pw - 10:w - 10] = small


# ----------------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="3D face-cam character filter (virtual webcam).")
    ap.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    ap.add_argument("--input", help="use an image/video file instead of the webcam")
    ap.add_argument("--cam-width", type=int, default=1280)
    ap.add_argument("--cam-height", type=int, default=720)
    ap.add_argument("--width", type=int, default=1280, help="output width")
    ap.add_argument("--height", type=int, default=720, help="output height")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--skin", default="0", help="skin index or name (see skins.py)")
    ap.add_argument("--mask", choices=MASKS, default="features",
                    help="features = your eyes, brows and mouth; full = whole face")
    ap.add_argument("--bg", choices=BACKGROUNDS, default="studio")
    ap.add_argument("--zoom", type=float, default=1.25,
                    help="camera zoom: 1 = full body with room, 2 = head and shoulders")
    ap.add_argument("--follow", type=float, default=0.6,
                    help="how much the character follows your head around the frame (0-1)")
    ap.add_argument("--smoothing", type=float, default=0.5, help="landmark smoothing 0-0.95")
    ap.add_argument("--pitch-offset", type=float, default=0.0,
                    help="degrees; tilt the neutral head pose (or press C to calibrate)")
    ap.add_argument("--supersample", type=float, default=1.5,
                    help="render scale for anti-aliasing; lower (1.0) for slow GPUs")
    ap.add_argument("--device", default=None, help="virtual camera device name (optional)")
    ap.add_argument("--model", default=os.path.join(HERE, "models", "face_landmarker.task"))
    ap.add_argument("--no-vcam", action="store_true", help="don't send to a virtual camera")
    ap.add_argument("--no-preview", action="store_true", help="don't open a preview window")
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = run forever)")
    ap.add_argument("--save", help="write the last rendered frame to this image file")
    return ap.parse_args(argv)


def pick_skin(value: str) -> int:
    if value.isdigit():
        return int(value) % len(SKINS)
    for i, s in enumerate(SKINS):
        if s["name"].lower().replace(" ", "") == value.lower().replace(" ", ""):
            return i
    raise SystemExit(f"Unknown skin {value!r}. Options: {[s['name'] for s in SKINS]}")


def main(argv=None):
    args = parse_args(argv)
    state = {
        "skin": pick_skin(args.skin),
        "mask": args.mask,
        "bg": args.bg,
        "help": True,
        "pip": True,
    }

    source = FileSource(args.input) if args.input else CameraSource(
        args.camera, args.cam_width, args.cam_height, args.fps)
    tracker = FaceTracker(args.model, smoothing=args.smoothing)
    renderer = Renderer(args.width, args.height, supersample=args.supersample)
    pose = Pose(args.follow, args.pitch_offset)
    vcam = None if args.no_vcam else open_virtual_camera(args)
    preview = not args.no_preview
    window = "PeanutCam (preview)"

    seq, frames = 0, 0
    t0 = last = time.perf_counter()
    last_seen = -1e9
    face_alpha = 0.0
    fps = 0.0
    out = None
    try:
        while True:
            frame, seq = source.read(seq)
            if frame is None:
                continue
            now = time.perf_counter()
            dt = min(0.1, now - last) if not args.input else 1.0 / args.fps
            last = now
            t = (now - t0) if not args.input else frames / args.fps
            fps = 0.9 * fps + 0.1 / max(dt, 1e-3)

            face = tracker.process(frame, int(t * 1000))
            renderer.upload_camera(frame)
            if face is not None:
                last_seen = t
                mask = face.mask_features if state["mask"] == "features" else face.mask_full
                renderer.update_face(face, mask, tracker.topology.triangles)
            # Fade the face in/out instead of popping when tracking is lost.
            target = 1.0 if t - last_seen < 0.25 else 0.0
            if args.input:
                face_alpha = target
            else:
                face_alpha += (target - face_alpha) * min(1.0, dt * 10.0)
            pose.update(face, dt)

            u = skin_uniforms(SKINS[state["skin"]])
            u.update(pose.uniforms(t))
            u.update({
                "uTime": t,
                "uZoom": args.zoom,
                "uFaceAlpha": face_alpha,
                "uBg": BACKGROUNDS.index(state["bg"]),
                "uBgTop": (0.36, 0.50, 0.72),
                "uBgBot": (0.93, 0.83, 0.70),
            })
            out = renderer.render(u)
            if vcam is not None:
                vcam.send(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

            frames += 1
            if args.frames and frames >= args.frames:
                break

            if preview:
                view = out.copy()
                if state["pip"]:
                    draw_pip(view, frame, face is not None)
                draw_hud(view, state, SKINS[state["skin"]]["name"], fps)
                cv2.imshow(window, view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if ord("1") <= key <= ord("9") and key - ord("1") < len(SKINS):
                    state["skin"] = key - ord("1")
                elif key == ord("n"):
                    state["skin"] = (state["skin"] + 1) % len(SKINS)
                elif key == ord("m"):
                    state["mask"] = MASKS[(MASKS.index(state["mask"]) + 1) % len(MASKS)]
                elif key == ord("b"):
                    state["bg"] = BACKGROUNDS[(BACKGROUNDS.index(state["bg"]) + 1) % len(BACKGROUNDS)]
                elif key == ord("c"):
                    pose.calibrate()
                elif key == ord("r"):
                    pose.reset()
                elif key == ord("p"):
                    state["pip"] = not state["pip"]
                elif key == ord("h"):
                    state["help"] = not state["help"]
    except KeyboardInterrupt:
        pass
    finally:
        if args.save and out is not None:
            cv2.imwrite(args.save, out)
            print(f"[peanutcam] saved {args.save}")
        source.close()
        tracker.close()
        if vcam is not None:
            vcam.close()
        if preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
