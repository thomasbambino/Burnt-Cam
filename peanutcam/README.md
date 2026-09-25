# PeanutCam

A Snap Camera–style webcam filter for Windows. It puts **your real face** (live eyes, brows
and mouth, or your whole face) on a **3D character**, like TheBurntPeanut's peanut avatar,
and sends the result to a virtual webcam that you can pick in Zoom, Discord, Teams, OBS,
Google Meet, and other video apps.

```
webcam ──► MediaPipe face tracking ──► 3D peanut (rendered on your GPU) ──► "OBS Virtual Camera"
```

* The character's head follows your head: turn, nod and tilt, and the peanut does the same.
  The body follows more loosely.
* Your face is "unwrapped" into a front-facing texture before it's painted on, so it stays
  glued to the peanut even when you turn your head.
* Opening your mouth makes the head stretch a little (squash and stretch).
* It comes with four skins: **Burnt Peanut**, **Golden Peanut**, **Egg** and **Potato**.
  You can add more in `skins.py`.
* There are three backgrounds: studio gradient, green screen (for OBS chroma key), or your
  real room.

## Setup (once)

1. **Install Python 3.11 or 3.12** from <https://www.python.org/downloads/windows/>.
   During setup, tick **"Add python.exe to PATH"**.
2. **Install OBS Studio** from <https://obsproject.com>. PeanutCam uses the
   *OBS Virtual Camera* driver that comes with OBS. You don't need OBS open while you use
   PeanutCam.
3. Double-click **`install.bat`**. It creates a local `.venv` folder and installs the
   packages.

## Use it

1. Double-click **`run.bat`**. A preview window opens. The first run downloads the face
   model, which is about 4 MB.
2. In Zoom, Discord or another app, open the video settings and choose
   **OBS Virtual Camera** as your camera.
3. Look straight at the camera and press **C** to calibrate your neutral head pose.

### Keys (click the preview window first)

| Key       | Action                                                           |
|-----------|------------------------------------------------------------------|
| 1–4 / N   | Switch skin                                                      |
| M         | Face mask: `features` (eyes, brows, mouth) or `full` (whole face) |
| B         | Background: studio / green / webcam                              |
| C         | Calibrate: look straight ahead and press it                      |
| R         | Reset calibration                                                |
| P         | Show or hide the webcam picture-in-picture                       |
| H         | Show or hide the help text                                       |
| Q / Esc   | Quit                                                             |

The help text and the picture-in-picture appear only in the preview. The virtual camera
always gets a clean image.

### Options

Pass options to `run.bat`, for example `run.bat --skin egg --bg green --zoom 1.6`.

| Option              | Default    | What it does                                                          |
|---------------------|------------|-----------------------------------------------------------------------|
| `--camera N`        | 0          | Which webcam to use. Try 1 or 2 if the wrong one opens.               |
| `--skin`            | 0          | Skin index or name (`burntpeanut`, `goldenpeanut`, `egg`, `potato`)   |
| `--mask`            | features   | `features` or `full`                                                  |
| `--bg`              | studio     | `studio`, `green` or `webcam`                                         |
| `--zoom`            | 1.25       | 1 = full body with room around it, 2 = head and shoulders             |
| `--follow`          | 0.6        | How much the peanut moves around the frame with you (0 = stays put)   |
| `--width/--height`  | 1280×720   | Output resolution                                                     |
| `--supersample`     | 1.5        | Anti-aliasing. Use `1` on a laptop or integrated GPU for more FPS.    |
| `--smoothing`       | 0.5        | Tracking smoothing (0 to 0.95). Higher is steadier but adds lag.      |
| `--no-vcam`         |            | Preview only; don't send to the virtual camera                        |
| `--input FILE`      |            | Use an image or video file instead of the webcam                      |

## Make your own character

Every character in `skins.py` is built from the same parts: a **head lobe** and a
**body lobe** (squashable ellipsoids that blend together smoothly), optional cartoon arms
and legs, and a procedural shell texture with scorch marks and netting. Copy one of the
entries and change:

* `head` / `body`: center and radius. `head_scale` / `body_scale` stretch the lobes.
* `blend`: how soft the waist is. Small values give a pinched peanut; large values give an egg.
* `base`, `dark`, `line`, `burn`, `net`: the colors and how burnt the shell looks.
* `face_size`, `face_y`: how big your face is and where it sits on the head.
* `face_tint`, `tint_amount`: tint your skin toward the character's color so it blends in.

## Troubleshooting

* **"Could not start the virtual camera".** Install OBS Studio 28 or newer. If OBS is open
  and its own **Start Virtual Camera** is on, turn it off, because only one program can
  drive the device.
* **The camera shows up black in Zoom.** Start PeanutCam first, then select the camera in
  Zoom. If it's still black, restart Zoom.
* **Low FPS.** Use `--supersample 1`, or `--width 960 --height 540`.
* **The wrong webcam opens.** Use `--camera 1`.
* **The peanut looks up or down when you look straight.** Press **C** while looking at the
  camera.

## How it works

1. `tracker.py` runs the MediaPipe Face Landmarker, which finds 478 face points plus
   expression "blendshapes". It builds a head coordinate frame from your eye corners,
   forehead and chin, rotates every face point back to a front-facing layout, and makes
   soft masks for your eyes, brows and mouth (or your whole face).
2. `renderer.py` works in three GPU passes:
   * **Unwrap.** It draws the face mesh with your webcam as its texture into a
     512×512 front-facing face texture. Extra triangles fill the mouth, so an open mouth
     shows your teeth.
   * **Scene.** It ray-marches the character, which is a signed-distance field: two
     blended ellipsoids plus limbs. It projects the face texture onto the front of the head
     through the mask, then adds lighting, ambient occlusion, a rim light and a ground
     shadow.
   * **Resolve.** It downsamples the image for anti-aliasing.
3. `peanutcam.py` connects everything: it smooths the head pose, handles calibration and
   hotkeys, and sends frames to the virtual camera with `pyvirtualcam`.
