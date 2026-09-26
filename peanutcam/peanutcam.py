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
MASKS = ["features", "full", "off"]

# Image space (x right, y down, z away) -> render space (x right, y up, z toward viewer).
FLIP = np.diag([1.0, -1.0, -1.0])

NECK = np.array([0.0, 0.02, 0.0])
PIVOT = np.array([0.0, -1.66, 0.0])  # the feet; the body leans around them


# ---------------------------------------------------------------- video input
# Virtual cameras (including the one PeanutCam itself sends to) are skipped
# when picking a webcam automatically.
VIRTUAL_CAMERA_WORDS = ("obs", "virtual", "unity video capture", "snap camera", "xsplit",
                        "manycam", "peanutcam")


def list_cameras() -> list[tuple[int, int, str]]:
    """(index, backend, name) for every camera we can find."""
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    try:
        from cv2_enumerate_cameras import enumerate_cameras
        cams = [(c.index, c.backend, c.name) for c in enumerate_cameras(backend)]
        if cams:
            return cams
    except Exception:  # noqa: BLE001 - optional package / unsupported platform
        pass
    # No names available: probe the first few indices (quietly - OpenCV logs
    # an error for every index that doesn't exist).
    found = []
    try:
        level = cv2.utils.logging.getLogLevel()
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except AttributeError:
        level = None
    for i in range(6):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            found.append((i, backend, f"Camera {i}"))
        cap.release()
    if level is not None:
        cv2.utils.logging.setLogLevel(level)
    return found


def is_virtual(name: str) -> bool:
    return any(w in name.lower() for w in VIRTUAL_CAMERA_WORDS)


def pick_cameras(spec: str) -> list[tuple[int, int, str]]:
    """Order cameras by preference for --camera (auto, an index, or part of a name)."""
    cams = list_cameras()
    if spec.isdigit():
        chosen = [c for c in cams if c[0] == int(spec)]
        if not chosen:
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            chosen = [(int(spec), backend, f"Camera {spec}")]
    elif spec != "auto":
        chosen = [c for c in cams if spec.lower() in c[2].lower()]
        if not chosen:
            raise RuntimeError(f"No camera named like {spec!r}. Found: {[c[2] for c in cams]}")
    else:
        chosen = []
    real = [c for c in cams if not is_virtual(c[2]) and c not in chosen]
    virtual = [c for c in cams if is_virtual(c[2]) and c not in chosen]
    return chosen + real + virtual


class CameraSource:
    """Grabs frames on a background thread so we always process the newest one."""

    def __init__(self, cam: tuple[int, int, str], width: int, height: int, fps: int):
        self.index, self.backend, self.name = cam
        self.cap = None
        # Some webcams only deliver black frames in MJPG mode or through
        # DirectShow, so fall back through a few ways of opening them.
        attempts = [(self.backend, True), (self.backend, False)]
        if sys.platform == "win32":
            attempts.append((cv2.CAP_MSMF, False))
        for backend, mjpg in attempts:
            cap = cv2.VideoCapture(self.index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            if mjpg:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            if self._delivers_picture(cap):
                self.cap = cap
                break
            cap.release()
        if self.cap is None:
            raise RuntimeError(f"Webcam '{self.name}' gave no picture.")
        self.frame = None
        self.seq = 0
        self.cond = threading.Condition()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    @staticmethod
    def _delivers_picture(cap, timeout: float = 3.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            ok, frame = cap.read()
            if ok and frame is not None and frame.mean() > 4:
                return True
        return False

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


def open_camera(cams, start: int, args):
    """Open the first working camera from cams[start:], wrapping around."""
    for k in range(len(cams)):
        i = (start + k) % len(cams)
        cam = cams[i]
        print(f"[peanutcam] Trying webcam: {cam[2]} ...")
        try:
            src = CameraSource(cam, args.cam_width, args.cam_height, args.fps)
        except RuntimeError as e:
            print(f"[peanutcam]   {e}")
            continue
        print(f"[peanutcam] Using webcam: {cam[2]}  (press K in the preview to switch)")
        return src, i
    raise SystemExit(
        "[peanutcam] Could not get a picture from any webcam.\n"
        "  - Close other apps that may be using it (Zoom, Teams, Discord, the Camera app, OBS).\n"
        "  - Check Windows Settings > Privacy & security > Camera: allow desktop apps.\n"
        "  - Run with --list-cameras to see what PeanutCam can find."
    )


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
        self.jaw = 0.0
        self.blink = np.zeros(2)
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
            target_b = np.array(face.blink)
            target_j = float(np.clip(face.jaw_open * 1.6 - 0.08, 0.0, 1.0))
        else:
            target_r = self.rvec * 0.9
            target_o = self.offset * 0.9
            target_s = 0.0
            target_b = np.zeros(2)
            target_j = 0.0
        self.rvec += (target_r - self.rvec) * a
        self.offset += (target_o - self.offset) * a
        self.stretch += (target_s - self.stretch) * min(1.0, a * 1.5)
        self.blink += (target_b - self.blink) * min(1.0, a * 2.5)  # blinks are fast
        self.jaw += (target_j - self.jaw) * min(1.0, a * 1.5)

    def uniforms(self, t: float, body_follow: float = 0.3) -> dict:
        Rh = cv2.Rodrigues(self.rvec)[0]
        Rb = rot_scale(Rh, body_follow)
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
            "uJaw": float(self.jaw),
            "uBlink": tuple(float(b) for b in self.blink),
        }


def skin_uniforms(skin: dict) -> dict:
    return {
        "uHead": skin["head"],
        "uHeadScale": skin["head_scale"],
        "uBody": skin["body"],
        "uBodyScale": skin["body_scale"],
        "uBlend": skin["blend"],
        "uBumps": skin["bumps"],
        "uLimbs": int(skin["limbs"]),
        "uLimbR": skin.get("limb_radius", 0.045),
        "uShape": 1 if skin.get("shape") == "lizard" else 0,
        "uBodyYaw": float(np.radians(skin.get("body_yaw", 0.0))),
        "uCamY": skin.get("camera_y", 0.0),
        "uNetScale": skin.get("net_scale", 1.0),
        "uBelly": skin.get("belly", (1.0, 1.0, 1.0)),
        "uBellyAmt": skin.get("belly_amount", 0.0),
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
    "M : your face on it: features / full / off",
    "B : background",
    "C : calibrate (look straight, press C)",
    "R : reset calibration",
    "P : webcam picture-in-picture",
    "K : switch to the next webcam",
    "H : hide this help",
    "Q / Esc : quit",
]


def draw_hud(img, state, skin_name, fps, warning=None):
    lines = [f"{skin_name} | mask: {state['mask']} | bg: {state['bg']} | {fps:4.1f} fps",
             f"webcam: {state['camera']}"]
    if state["help"]:
        lines += HELP
    y = 26
    for line in lines:
        cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    if warning:
        h = img.shape[0]
        cv2.putText(img, warning, (12, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, warning, (12, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 200, 255), 2, cv2.LINE_AA)


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
    ap.add_argument("--camera", default="auto",
                    help="auto (first real webcam), a number, or part of the webcam's name")
    ap.add_argument("--list-cameras", action="store_true", help="list webcams and exit")
    ap.add_argument("--input", help="use an image/video file instead of the webcam")
    ap.add_argument("--cam-width", type=int, default=1280)
    ap.add_argument("--cam-height", type=int, default=720)
    ap.add_argument("--width", type=int, default=1280, help="output width")
    ap.add_argument("--height", type=int, default=720, help="output height")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--skin", default="0", help="skin index or name (see skins.py)")
    ap.add_argument("--mask", choices=MASKS, default=None,
                    help="features = your eyes, brows and mouth; full = whole face; "
                         "off = puppet mode, your face only animates the character "
                         "(default: whatever suits the skin)")
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
        "mask": args.mask or SKINS[pick_skin(args.skin)].get("default_mask", "features"),
        "bg": args.bg,
        "help": True,
        "pip": True,
        "camera": args.input or "",
    }

    if args.list_cameras:
        for index, _, name in list_cameras():
            print(f"  {index}: {name}{'  (virtual - skipped by auto)' if is_virtual(name) else ''}")
        return
    cams, cam_i = [], 0
    if args.input:
        source = FileSource(args.input)
    else:
        cams = pick_cameras(str(args.camera))
        if not cams:
            raise SystemExit("[peanutcam] No webcams found. Is one plugged in and allowed in "
                             "Windows Settings > Privacy & security > Camera?")
        source, cam_i = open_camera(cams, 0, args)
        state["camera"] = cams[cam_i][2]
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
                skin = SKINS[state["skin"]]
                if state["mask"] == "full":
                    mask = face.mask_full
                elif skin.get("mouth_only"):
                    mask = face.mask_mouth
                else:
                    mask = face.mask_features
                renderer.update_face(face, mask, tracker.topology.triangles)
            # Fade the face in/out instead of popping when tracking is lost.
            target = 1.0 if t - last_seen < 0.25 else 0.0
            if args.input:
                face_alpha = target
            else:
                face_alpha += (target - face_alpha) * min(1.0, dt * 10.0)
            pose.update(face, dt)

            skin = SKINS[state["skin"]]
            u = skin_uniforms(skin)
            u.update(pose.uniforms(t, skin.get("body_follow", 0.3)))
            u.update({
                "uTime": t,
                "uZoom": args.zoom * skin.get("zoom", 1.0),
                "uFaceAlpha": 0.0 if state["mask"] == "off" else face_alpha,
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
                warning = None
                if t - last_seen > 1.5:
                    warning = ("No face found - face the camera in good light"
                               + (", or press K to try another webcam" if len(cams) > 1 else ""))
                draw_hud(view, state, SKINS[state["skin"]]["name"], fps, warning)
                cv2.imshow(window, view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                new_skin = None
                if ord("1") <= key <= ord("9") and key - ord("1") < len(SKINS):
                    new_skin = key - ord("1")
                elif key == ord("n"):
                    new_skin = (state["skin"] + 1) % len(SKINS)
                if new_skin is not None:
                    state["skin"] = new_skin
                    state["mask"] = SKINS[new_skin].get("default_mask", "features")
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
                elif key == ord("k") and len(cams) > 1:
                    source.close()
                    source, cam_i = open_camera(cams, cam_i + 1, args)
                    state["camera"] = cams[cam_i][2]
                    seq = 0
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
