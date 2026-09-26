# Burnt Cam

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
* It comes with five skins: **Burnt Peanut**, **Golden Peanut**, **Egg**, **Potato** and
  **Lizard**. The lizard is a cute, upright cartoon inspired by Tom Lizard from Pixar's
  *Hoppers*. It has a soft green bean-shaped body, big googly eyes, a wide smile and a
  curly tail. By default it's a *puppet*: its mouth opens when you open yours, its tongue
  sticks out when you stick yours out, its eyes blink when you blink, and its head turns
  when you turn. Press **M** to put your real face on it instead. You can add more skins in
  `skins.py`.
* Backgrounds: studio gradient, green screen (for OBS chroma key), your real room, or any
  picture or video of your own (press **U** to upload one).
* Camera views: full body, waist up, chest up or close-up (press **V**), with fine zoom on
  **+ / −**.
* Burnt Cam remembers your character, view, background and zoom for next time.

## Setup (once)

1. **Install Python 3.11 or 3.12** from <https://www.python.org/downloads/windows/>.
   During setup, tick **"Add python.exe to PATH"**.
2. **Install OBS Studio** from <https://obsproject.com>. Burnt Cam uses the
   *OBS Virtual Camera* driver that comes with OBS. You don't need OBS open while you use
   Burnt Cam.
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
| 1–5 / N   | Switch character                                                 |
| V         | Camera view: full body / waist up / chest up / close-up          |
| + / −     | Zoom in / out                                                    |
| B         | Next background (studio, green, webcam, then your own)           |
| U         | Upload a background picture or video                             |
| M         | Your face on the character: `features` (eyes, brows, mouth), `full` (whole face) or `off` (puppet mode). Each character starts with its own default. |
| C         | Calibrate: look straight ahead and press it                      |
| R         | Reset calibration                                                |
| K         | Switch to the next webcam                                        |
| P         | Show or hide the webcam picture-in-picture                       |
| H         | Show or hide the help text                                       |
| Q / Esc   | Quit                                                             |

The help text and the picture-in-picture appear only in the preview. The virtual camera
always gets a clean image.

### Options

Pass options to `run.bat`, for example `run.bat --skin egg --bg green --view waist`.

| Option              | Default    | What it does                                                          |
|---------------------|------------|-----------------------------------------------------------------------|
| `--camera`          | auto       | Which webcam: `auto` (the first real one), a number, or part of its name, e.g. `--camera c920` |
| `--list-cameras`    |            | Show the webcams Burnt Cam can find, then exit                        |
| `--skin`            | last used  | Character number (1–5) or name (`burntpeanut`, `goldenpeanut`, `egg`, `potato`, `lizard`) |
| `--mask`            | per skin   | `features`, `full` or `off` (puppet mode)                             |
| `--bg`              | last used  | `studio`, `green`, `webcam`, or the path to a picture or video        |
| `--view`            | last used  | `full`, `waist`, `chest` or `face`                                    |
| `--zoom`            | last used  | Extra zoom on top of the view (1 = none, 1.2 = 20% closer)            |
| `--tongue-sensitivity` | 1.0     | Raise it (e.g. 1.5) if the lizard misses your tongue; lower it (e.g. 0.7) if it sticks its tongue out by itself |
| `--follow`          | 0.6        | How much the peanut moves around the frame with you (0 = stays put)   |
| `--width/--height`  | 1280×720   | Output resolution                                                     |
| `--supersample`     | 1.5        | Anti-aliasing. Use `1` on a laptop or integrated GPU for more FPS.    |
| `--smoothing`       | 0.5        | Tracking smoothing (0 to 0.95). Higher is steadier but adds lag.      |
| `--no-vcam`         |            | Preview only; don't send to the virtual camera                        |
| `--input FILE`      |            | Use an image or video file instead of the webcam                      |

## Custom backgrounds

Press **U** in the preview and choose a picture or video (JPG, PNG, WEBP, GIF, MP4, MOV,
and so on). Burnt Cam copies it into the `backgrounds` folder and switches to it. Videos
loop. You can also drop files straight into `backgrounds`. Press **B** to cycle through
them. Pictures are cropped to fill the frame, not stretched.

## Make your own character

Every character in `skins.py` is built from the same parts: a **head lobe** and a
**body lobe** (squashable ellipsoids that blend together smoothly), optional cartoon arms
and legs, and a procedural shell texture with scorch marks and netting. Copy one of the
entries and change:

* `head` / `body`: center and radius. `head_scale` / `body_scale` stretch the lobes.
* `blend`: how soft the waist is. Small values give a pinched peanut; large values give an egg.
* `base`, `dark`, `line`, `burn`, `net`: the colors and how burnt the shell looks.
* `face_size`, `face_y`: how big your face is and where it sits on the head.
* `top`: the highest point of the character, used to frame the camera views. It defaults
  to the top of the head.
* `face_tint`, `tint_amount`: tint your skin toward the character's color so it blends in.
* Optional extras: `belly` / `belly_amount`, `net_scale` (scale size), `mouth_only` (the
  features mode shows only your mouth) and `body_follow` (how much the body turns with
  your head).
* `default_mask`: the face mode the skin starts in (`features`, `full` or `off`).
* `"shape": "lizard"` switches to the cartoon lizard model, which has its own body. It's
  defined in `mapLizard` in `renderer.py`. `body_yaw` turns its body.

## Troubleshooting

* **"Could not start the virtual camera".** Install OBS Studio 28 or newer. If OBS is open
  and its own **Start Virtual Camera** is on, turn it off, because only one program can
  drive the device.
* **The camera shows up black in Zoom.** Start Burnt Cam first, then select the camera in
  Zoom. If it's still black, restart Zoom.
* **Low FPS.** Use `--supersample 1`, or `--width 960 --height 540`.
* **The preview doesn't use your webcam.** The picture-in-picture in the bottom-right corner
  shows what Burnt Cam sees, and the name of the webcam it's using is at the top. Press
  **K** to switch webcams, or run `run.bat --list-cameras` and pick one with
  `run.bat --camera <number or name>`. Burnt Cam skips virtual cameras such as OBS Virtual
  Camera automatically. Close other apps that might be using the webcam (Zoom, Teams, the
  Camera app, OBS), and check **Windows Settings > Privacy & security > Camera**: "Let
  desktop apps access your camera" must be on.
* **The peanut looks up or down when you look straight.** Press **C** while looking at the
  camera.
* **The lizard doesn't stick its tongue out (or does it by itself).** Burnt Cam spots your
  tongue by its pink color below your lower lip, so good, even lighting on your face helps.
  Keep your mouth closed for a couple of seconds after starting so it learns your normal
  look. Then use `--tongue-sensitivity 1.5` to make it easier to trigger, or `0.7` to make
  it harder.
* **In Zoom/Discord the camera is called "OBS Virtual Camera", not "Burnt Cam".** That's
  the name of the virtual camera driver that OBS installs, and Burnt Cam can't rename it.

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
3. `burntcam.py` connects everything: it smooths the head pose, handles calibration and
   hotkeys, and sends frames to the virtual camera with `pyvirtualcam`.
