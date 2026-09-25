"""Face tracking: MediaPipe Face Landmarker -> head pose + a frontalized face mesh.

The tracker turns every webcam frame into a ``FaceState``:

* ``img_uv``   - where each mesh vertex sits in the webcam image (0..1)
* ``tex_uv``   - where that vertex should land in a *front-facing*, pose-free
                 "face texture" (0..1).  Rendering the mesh with these two sets
                 of coordinates unwraps the user's face into a stable texture
                 that can be painted onto the 3D character.
* ``rotation`` - 3x3 head rotation (image space: x right, y down, z away).
* ``center``   - face center in normalized image coordinates.
* ``jaw_open`` - 0..1 blendshape, used for a little squash & stretch.
"""

from __future__ import annotations

import os
import sys
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections as FLC

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# Landmark ids used to build the head coordinate frame.
L_EYE_OUTER, R_EYE_OUTER = 33, 263
FOREHEAD, CHIN = 10, 152
N_MESH = 468  # the 10 iris points (468..477) are not part of the tessellation

# Frontal face layout inside the face texture.  Units are "eye-corner
# distances" so the layout is the same no matter how close you sit.
TEX_SCALE = 0.40

FACE_TEX_SIZE = 512
MASK_TEX_SIZE = 256


def ensure_model(path: str) -> str:
    if not os.path.exists(path):
        print(f"[peanutcam] Downloading face model to {path} ...", file=sys.stderr)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".part"
        urllib.request.urlretrieve(MODEL_URL, tmp)
        os.replace(tmp, path)
    return path


def _edges(conns) -> list[tuple[int, int]]:
    return [(c.start, c.end) for c in conns]


def _indices(conns) -> np.ndarray:
    return np.array(sorted({i for e in _edges(conns) for i in e}), dtype=np.int32)


def _loops(conns) -> list[list[int]]:
    """Split a set of connections into closed, ordered vertex loops."""
    adj: dict[int, list[int]] = {}
    for a, b in _edges(conns):
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    seen: set[int] = set()
    loops = []
    for start in adj:
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxt = [n for n in adj[cur] if n != prev and (n not in seen or n == start)]
            if not nxt or nxt[0] == start:
                break
            prev, cur = cur, nxt[0]
            seen.add(cur)
            loop.append(cur)
        loops.append(loop)
    return loops


def _triangles_from_tessellation() -> np.ndarray:
    """MediaPipe ships the face mesh as edges; recover its triangles."""
    edges = {(min(a, b), max(a, b)) for a, b in _edges(FLC.FACE_LANDMARKS_TESSELATION)}
    adj: dict[int, set[int]] = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    tris = {tuple(sorted((a, b, c))) for a, b in edges for c in adj[a] & adj[b]}
    return np.array(sorted(tris), dtype=np.int32)


def _loop_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


class FaceTopology:
    """Static mesh data: triangles + fans that fill the eye and mouth holes."""

    def __init__(self):
        self.base_tris = _triangles_from_tessellation()
        # Extra "centroid" vertices appended after the 468 mesh vertices close
        # the eye and mouth holes so an open mouth shows teeth, not a gap.
        # Which lip loop is the inner one is decided on the first frame.
        self._lip_loops = _loops(FLC.FACE_LANDMARKS_LIPS)
        self.fill_loops: list[list[int]] | None = None
        self.triangles: np.ndarray | None = None

        self.left_eye = _indices(FLC.FACE_LANDMARKS_LEFT_EYE)
        self.right_eye = _indices(FLC.FACE_LANDMARKS_RIGHT_EYE)
        self.left_brow = _indices(FLC.FACE_LANDMARKS_LEFT_EYEBROW)
        self.right_brow = _indices(FLC.FACE_LANDMARKS_RIGHT_EYEBROW)
        self.lips = _indices(FLC.FACE_LANDMARKS_LIPS)
        self.oval = _loops(FLC.FACE_LANDMARKS_FACE_OVAL)[0]
        self._eye_loops = _loops(FLC.FACE_LANDMARKS_LEFT_EYE) + _loops(FLC.FACE_LANDMARKS_RIGHT_EYE)

    def finalize(self, tex_uv: np.ndarray):
        if self.triangles is not None:
            return
        inner = min(self._lip_loops, key=lambda l: _loop_area(tex_uv[l]))
        self.fill_loops = [inner] + self._eye_loops
        fans = []
        for k, loop in enumerate(self.fill_loops):
            c = N_MESH + k
            for i in range(len(loop)):
                fans.append((c, loop[i], loop[(i + 1) % len(loop)]))
        self.triangles = np.concatenate([self.base_tris, np.array(fans, np.int32)])


@dataclass
class FaceState:
    img_uv: np.ndarray       # (N, 2) float32, webcam texture coords
    tex_uv: np.ndarray       # (N, 2) float32, face texture coords
    rotation: np.ndarray     # (3, 3) head rotation, image space
    center: np.ndarray       # (2,) normalized face center in the frame
    size: float              # eye-corner distance / frame width
    jaw_open: float
    mask_full: np.ndarray    # (MASK, MASK) uint8 face-oval mask in tex space
    mask_features: np.ndarray  # eyes + brows + mouth mask in tex space


def head_frame(p: np.ndarray) -> np.ndarray:
    """Orthonormal head axes (columns) from 3D landmarks in image space."""
    x = p[R_EYE_OUTER] - p[L_EYE_OUTER]
    x /= np.linalg.norm(x) + 1e-9
    y = p[CHIN] - p[FOREHEAD]
    y -= x * np.dot(x, y)
    y /= np.linalg.norm(y) + 1e-9
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1)


class FaceTracker:
    def __init__(self, model_path: str, smoothing: float = 0.5):
        opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=ensure_model(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
        self.topology = FaceTopology()
        self.smoothing = float(np.clip(smoothing, 0.0, 0.95))
        self._p3: np.ndarray | None = None
        self._last_ts = -1

    def close(self):
        self.landmarker.close()

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> FaceState | None:
        h, w = frame_bgr.shape[:2]
        ts = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = ts
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(image, ts)
        if not result.face_landmarks:
            self._p3 = None
            return None

        lms = result.face_landmarks[0][:N_MESH]
        p3 = np.array([(l.x * w, l.y * h, l.z * w) for l in lms], dtype=np.float64)
        # Temporal smoothing kills most landmark jitter.
        if self._p3 is not None and self.smoothing > 0:
            p3 = self.smoothing * self._p3 + (1.0 - self.smoothing) * p3
        self._p3 = p3

        jaw = 0.0
        if result.face_blendshapes:
            for cat in result.face_blendshapes[0]:
                if cat.category_name == "jawOpen":
                    jaw = float(cat.score)
                    break
        return self._build_state(p3, w, h, jaw)

    def _build_state(self, p3, w, h, jaw) -> FaceState:
        R = head_frame(p3)
        eye_dist = np.linalg.norm(p3[R_EYE_OUTER] - p3[L_EYE_OUTER])
        center = 0.5 * (p3[FOREHEAD] + p3[CHIN])
        # Undo the head rotation -> front-facing coordinates of every vertex.
        frontal = (p3 - center) @ R / eye_dist
        tex = 0.5 + TEX_SCALE * frontal[:, :2]

        topo = self.topology
        topo.finalize(tex)
        img = p3[:, :2] / np.array([w, h])

        # Centroid vertices for the hole-filling fans.
        tex_c = [tex[l].mean(axis=0) for l in topo.fill_loops]
        img_c = [img[l].mean(axis=0) for l in topo.fill_loops]
        tex_all = np.vstack([tex, tex_c]).astype(np.float32)
        img_all = np.vstack([img, img_c]).astype(np.float32)

        return FaceState(
            img_uv=img_all,
            tex_uv=tex_all,
            rotation=R,
            center=(center[:2] / np.array([w, h])).astype(np.float32),
            size=float(eye_dist / w),
            jaw_open=jaw,
            mask_full=self._mask(tex, [topo.oval], [], erode=0.035),
            mask_features=self._mask(
                tex,
                [],
                [topo.left_eye, topo.right_eye, topo.lips, topo.left_brow, topo.right_brow],
            ),
        )

    @staticmethod
    def _mask(tex, loops, hulls, erode=0.0) -> np.ndarray:
        S = MASK_TEX_SIZE
        m = np.zeros((S, S), np.uint8)
        px = lambda idx: np.round(tex[idx] * S).astype(np.int32)
        for loop in loops:
            cv2.fillPoly(m, [px(loop)], 255, lineType=cv2.LINE_AA)
        for idx in hulls:
            hull = cv2.convexHull(px(idx))
            cv2.fillConvexPoly(m, hull, 255, lineType=cv2.LINE_AA)
            # Pad features so a little skin frames each eye / the mouth.
            cv2.polylines(m, [hull], True, 255, thickness=max(2, S // 40), lineType=cv2.LINE_AA)
        if erode > 0:
            k = max(1, int(erode * S))
            m = cv2.erode(m, np.ones((k, k), np.uint8))
        blur = max(3, (S // 24) | 1)
        return cv2.GaussianBlur(m, (blur, blur), 0)
