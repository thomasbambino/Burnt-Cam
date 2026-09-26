"""Character skins.

Every character is built from the same signed-distance "rig": a head lobe and
a body lobe (ellipsoids) smoothly blended together, optional cartoon limbs,
and a procedural shell material.  Your face is projected onto the front of
the head lobe.  Add a new dict to SKINS to create a new character.

Shape units: the character stands roughly 2.8 units tall, feet at y = -1.6.
Colors are 0..1 RGB.

Optional keys: "net_scale" (size of the scale/netting cells), "belly" +
"belly_amount", "limb_radius", "body_follow" (how much the body turns with
your head), "top" (highest point, for camera framing; defaults to the top of
the head), "mouth_only" (features mask shows only your mouth), "default_mask"
(features / full / off).
"shape": "lizard" switches to the cartoon lizard model, which has its own
body (see mapLizard in renderer.py) and uses "body_yaw".
"""

SKINS = [
    {
        "name": "Burnt Peanut",
        "head": (0.0, 0.50, 0.0, 0.60),        # center xyz, radius
        "head_scale": (1.00, 1.05, 0.95),      # ellipsoid stretch
        "body": (0.0, -0.50, 0.0, 0.64),
        "body_scale": (1.00, 1.08, 0.95),
        "blend": 0.14,                         # how soft the waist is
        "bumps": 0.010,                        # surface lumpiness
        "base": (0.78, 0.56, 0.33),            # shell color
        "dark": (0.16, 0.08, 0.04),            # scorch color
        "line": (0.45, 0.28, 0.14),            # shell netting color
        "burn": 0.85,                          # 0 = raw, 1 = charcoal
        "net": 0.55,                           # netting strength
        "gloss": 0.25,
        "limbs": True,
        "limb_color": (0.07, 0.06, 0.06),
        "glove_color": (0.95, 0.95, 0.92),
        "face_size": 1.45,                     # width of the face decal
        "face_y": 0.46,                        # height of the face on the head
        "face_tint": (0.92, 0.78, 0.62),       # warms your skin toward the shell
        "tint_amount": 0.35,
    },
    {
        "name": "Golden Peanut",
        "head": (0.0, 0.50, 0.0, 0.60),
        "head_scale": (1.00, 1.05, 0.95),
        "body": (0.0, -0.50, 0.0, 0.64),
        "body_scale": (1.00, 1.08, 0.95),
        "blend": 0.14,
        "bumps": 0.010,
        "base": (0.90, 0.72, 0.45),
        "dark": (0.55, 0.36, 0.18),
        "line": (0.62, 0.45, 0.25),
        "burn": 0.25,
        "net": 0.6,
        "gloss": 0.2,
        "limbs": True,
        "limb_color": (0.07, 0.06, 0.06),
        "glove_color": (0.95, 0.95, 0.92),
        "face_size": 1.45,
        "face_y": 0.46,
        "face_tint": (1.0, 0.88, 0.72),
        "tint_amount": 0.25,
    },
    {
        "name": "Egg",
        "head": (0.0, 0.05, 0.0, 0.95),
        "head_scale": (0.85, 1.20, 0.85),
        "body": (0.0, -0.45, 0.0, 0.95),
        "body_scale": (0.95, 1.00, 0.95),
        "blend": 0.60,
        "bumps": 0.0,
        "base": (0.97, 0.94, 0.88),
        "dark": (0.85, 0.80, 0.70),
        "line": (0.97, 0.94, 0.88),
        "burn": 0.25,
        "net": 0.0,
        "gloss": 0.7,
        "limbs": True,
        "limb_color": (0.07, 0.06, 0.06),
        "glove_color": (0.98, 0.85, 0.20),
        "face_size": 1.45,
        "face_y": 0.40,
        "face_tint": (1.0, 1.0, 1.0),
        "tint_amount": 0.0,
    },
    {
        "name": "Potato",
        "head": (0.0, 0.15, 0.0, 0.90),
        "head_scale": (1.05, 1.10, 0.85),
        "body": (0.0, -0.55, 0.0, 0.80),
        "body_scale": (1.10, 1.00, 0.85),
        "blend": 0.50,
        "bumps": 0.016,
        "base": (0.72, 0.55, 0.34),
        "dark": (0.42, 0.29, 0.16),
        "line": (0.60, 0.44, 0.26),
        "burn": 0.55,
        "net": 0.0,
        "gloss": 0.1,
        "limbs": True,
        "limb_color": (0.07, 0.06, 0.06),
        "glove_color": (0.95, 0.95, 0.92),
        "face_size": 1.40,
        "face_y": 0.40,
        "face_tint": (0.95, 0.82, 0.66),
        "tint_amount": 0.30,
    },
    {
        # A cute cartoon lizard in the spirit of Tom Lizard from Pixar's
        # "Hoppers": a soft, bean-shaped green body, big googly eyes on top of
        # the head, a wide smile, thin arms, short bow legs and a curly tail.
        # By default it's a puppet: your face drives it (mouth opens with
        # yours, eyes blink with yours, head turns with yours). Press M to put
        # your real face on it instead.
        "name": "Lizard",
        "shape": "lizard",                     # custom lizard model (see renderer.py)
        "default_mask": "off",
        "head": (0.0, 0.40, 0.0, 0.30),        # used to place the face decal (M)
        "head_scale": (1.0, 1.0, 1.0),
        "body": (0.0, -0.74, 0.0, 0.40),
        "body_scale": (1.0, 1.0, 1.0),
        "blend": 0.15,
        "bumps": 0.0,
        "body_yaw": 12,                        # a slight three-quarter turn
        "body_follow": 0.25,
        "mouth_only": True,                    # "features" mask shows just your mouth
        "base": (0.47, 0.70, 0.43),            # soft minty green
        "dark": (0.12, 0.22, 0.12),            # nostrils
        "line": (0.38, 0.60, 0.36),            # tail scales
        "burn": 0.0,
        "net": 0.45,
        "net_scale": 1.0,
        "gloss": 0.22,
        "belly": (0.68, 0.82, 0.55),           # paler belly and throat
        "belly_amount": 0.8,
        "limbs": False,
        "limb_color": (0.47, 0.70, 0.43),
        "glove_color": (0.47, 0.70, 0.43),
        "face_size": 0.95,
        "face_y": 0.42,
        "face_tint": (0.85, 0.95, 0.80),
        "tint_amount": 0.25,
        "top": 0.86,                           # top of the eyes (for camera framing)
    },
    {
        # A sorcerer in the spirit of Yu-Gi-Oh's Dark Magician: tall spiral
        # hat, winged pauldrons, chrome-purple armor with lilac trim, an open
        # blue-violet robe and a teal staff with a glowing orb. Your whole
        # face shows in the hat's opening.
        "name": "Dark Mage",
        "shape": "mage",                       # custom model (see mapMage in renderer.py)
        "default_mask": "full",
        "head": (0.0, 0.39, 0.0, 0.25),
        "head_scale": (1.0, 1.0, 1.0),
        "body": (0.0, -0.28, 0.0, 0.40),
        "body_scale": (1.0, 1.0, 1.0),
        "top": 1.62,                           # hat tip, for camera framing
        "blend": 0.1,
        "bumps": 0.0,
        "body_follow": 0.2,
        "base": (0.33, 0.14, 0.55),
        "dark": (0.20, 0.08, 0.35),
        "line": (0.95, 0.72, 0.30),
        "burn": 0.0,
        "net": 0.0,
        "gloss": 0.6,
        "limbs": False,
        "limb_color": (0.33, 0.14, 0.55),
        "glove_color": (0.86, 0.80, 0.86),
        "face_size": 0.84,
        "face_y": 0.37,
        "face_tint": (0.92, 0.90, 1.0),        # a touch of the mage's pale, cool skin
        "tint_amount": 0.15,
    },
]
