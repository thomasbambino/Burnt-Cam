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

## Install (easiest)

1. Download **BurntCamSetup.exe** from the
   [latest release](https://github.com/thomasbambino/zoom/releases/tag/burntcam-latest).
2. Run it. If Windows says "Windows protected your PC", click **More info → Run anyway**.
   The installer isn't code-signed, so Windows shows this for it.
3. Click through the setup and say **Yes** when Windows asks for permission. That's the
   only permission prompt. The setup also installs OBS Studio (for the virtual camera) and
   the Microsoft Visual C++ runtime if your PC needs them.
4. Start **Burnt Cam** from the desktop or the Start menu. There's no console window;
   it opens straight to the preview.

Your settings and uploaded backgrounds are kept in `%LOCALAPPDATA%\BurntCam`, so they
survive updates. To update, just run a newer `BurntCamSetup.exe`. To uninstall, use
Windows **Settings → Apps**.

## Or run it from the ZIP

1. Download and extract Burnt Cam, then double-click **`Burnt Cam.bat`**. That's the only
   file you ever need to run.
2. **The first time**, it sets everything up for you. This takes a few minutes and
   downloads about 300 MB:
   * its own private copy of Python, which doesn't affect any other Python on your PC
   * the Microsoft Visual C++ runtime, if your PC doesn't have it
   * **OBS Studio**, which provides the "OBS Virtual Camera" that Zoom and Discord see.
     You never need to open OBS itself.
   * Burnt Cam's packages, plus a **Burnt Cam** shortcut on your desktop

   Windows asks "Do you want to allow this app to make changes?" once or twice. That's for
   OBS and the Visual C++ runtime, so click **Yes**.
3. **After that**, double-click **Burnt Cam** on your desktop and it starts in a few
   seconds.
4. In Zoom, Discord or another app, open the video settings and choose
   **OBS Virtual Camera** as your camera.
5. Look straight at the camera and press **C** to calibrate your neutral head pose.

If Windows shows "Windows protected your PC" the first time, click **More info →
Run anyway**. Windows shows this for any script downloaded from the internet.

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

Most people never need these. To use them, open Command Prompt in the Burnt Cam folder and
add them after the file name, for example `"Burnt Cam.bat" --skin egg --bg green --view waist`.

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

* **Setup stopped with an error.** Usually the internet connection dropped. Run
  `Burnt Cam.bat` again, and it continues where it left off. Everything it installs lives
  in `%LOCALAPPDATA%\BurntCam`. Delete that folder to start setup from scratch.
* **You said No to installing OBS.** Burnt Cam then runs in preview-only mode and won't
  ask again. Install OBS from <https://obsproject.com> yourself, or delete
  `%LOCALAPPDATA%\BurntCam\skip-obs` to have Burnt Cam offer again.
* **"Could not start the virtual camera".** Install OBS Studio 28 or newer. If OBS is open
  and its own **Start Virtual Camera** is on, turn it off, because only one program can
  drive the device.
* **The camera shows up black in Zoom.** Start Burnt Cam first, then select the camera in
  Zoom. If it's still black, restart Zoom.
* **Low FPS.** Use `--supersample 1`, or `--width 960 --height 540`.
* **The preview doesn't use your webcam.** The picture-in-picture in the bottom-right corner
  shows what Burnt Cam sees, and the name of the webcam it's using is at the top. Press
  **K** to switch webcams, or run `"Burnt Cam.bat" --list-cameras` and pick one
  with `"Burnt Cam.bat" --camera <number or name>`. Burnt Cam skips virtual cameras such as OBS Virtual
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

## Building the installer

`.github/workflows/burntcam-installer.yml` builds `BurntCamSetup.exe` on a Windows
GitHub Actions runner whenever `burntcam/` changes, and publishes it as the
`burntcam-latest` release. `installer/build.ps1` prepares the app folder: the embeddable
Python with all packages, the code and the face model, plus a smoke test.
`installer/burntcam.iss` is the Inno Setup script that packs it, downloads and installs OBS
and the Visual C++ runtime, and creates the shortcuts.

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
