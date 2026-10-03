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
the head), "head_follow" (how much the head turns with yours), "mouth_only"
(features mask shows only your mouth), "no_brows" (features mask shows your
eyes and mouth but not your eyebrows), "default_mask" (features / full / off),
"head_only" (a floating peanut: both lobes turn with your head, no limbs),
"hat" (none / propeller / cowboy, cycled with T), "cracks" (dry cracks in the
shell), "eye_scale" / "mouth_scale" / "eye_spread" (blow up and spread the
features on the shell), "lip_tint" + "lip_amount" (tint your lips).
"shape": "lizard" switches to the cartoon lizard model, which has its own
body (see mapLizard in renderer.py) and uses "body_yaw".
"""

SKINS = [
    {
        # TheBurntPeanut's look: a single roasted peanut in its shell that
        # floats where your head is, with your real eyes and mouth blown up
        # on the front (no brows), pink lips, a dry cracked tan shell and a
        # propeller beanie. Press T to swap the hat.
        "name": "Burnt Peanut",
        "head_only": True,                     # both lobes turn with your head; no body or limbs
        "hat": "propeller",
        "head": (0.0, 0.62, 0.0, 0.66),        # upper lobe: center xyz, radius
        "head_scale": (1.00, 1.06, 0.92),      # ellipsoid stretch
        "body": (0.0, -0.30, 0.0, 0.64),       # lower lobe
        "body_scale": (1.02, 1.00, 0.92),
        "blend": 0.42,                         # a gentle waist, not a pinched one
        "bumps": 0.008,                        # surface lumpiness
        "base": (0.80, 0.60, 0.36),            # tan roasted shell
        "dark": (0.30, 0.16, 0.07),            # over-roasted patches and cracks
        "line": (0.50, 0.33, 0.17),            # shell netting color
        "burn": 0.45,                          # 0 = raw, 1 = charcoal
        "net": 0.5,                            # netting strength
        "cracks": 0.8,                         # dry cracks in the shell
        "gloss": 0.12,                         # matte
        "limbs": False,
        "limb_color": (0.07, 0.06, 0.06),
        "glove_color": (0.95, 0.95, 0.92),
        "face_size": 1.90,                     # width of the face decal
        "face_y": 0.22,                        # eyes on the top lobe, mouth on the bottom one
        "no_brows": True,                      # eyes and mouth only, like the real peanut
        "eye_scale": 1.65,                     # blow up the eyes and the mouth
        "mouth_scale": 1.45,
        "eye_spread": 0.025,
        "lip_tint": (0.95, 0.42, 0.58),        # pink lips
        "lip_amount": 0.6,
        "face_tint": (0.92, 0.78, 0.62),       # warms your skin toward the shell
        "tint_amount": 0.30,
    },
    {
        # The same floating peanut, gold.
        "name": "Golden Peanut",
        "head_only": True,
        "hat": "none",
        "head": (0.0, 0.62, 0.0, 0.66),
        "head_scale": (1.00, 1.06, 0.92),
        "body": (0.0, -0.30, 0.0, 0.64),
        "body_scale": (1.02, 1.00, 0.92),
        "blend": 0.42,
        "bumps": 0.008,
        "base": (0.92, 0.74, 0.40),
        "dark": (0.60, 0.42, 0.16),
        "line": (0.66, 0.50, 0.24),
        "burn": 0.20,
        "net": 0.55,
        "cracks": 0.5,
        "gloss": 0.5,
        "limbs": False,
        "limb_color": (0.07, 0.06, 0.06),
        "glove_color": (0.95, 0.95, 0.92),
        "face_size": 1.90,
        "face_y": 0.22,
        "no_brows": True,
        "eye_scale": 1.65,
        "mouth_scale": 1.45,
        "eye_spread": 0.025,
        "lip_tint": (0.95, 0.42, 0.58),
        "lip_amount": 0.6,
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
    {
        # A card duelist in the spirit of Yu-Gi-Oh's Yugi: huge spiky black hair
        # with magenta tips and blonde bangs, a blue school jacket, a choker,
        # a gold pyramid pendant, and a card held up. Your face is in the middle.
        "name": "Duelist",
        "shape": "duelist",                    # custom model (see mapDuelist in renderer.py)
        "default_mask": "full",
        "head": (0.0, 0.39, 0.0, 0.25),
        "head_scale": (1.0, 1.0, 1.0),
        "body": (0.0, -0.40, 0.0, 0.40),
        "body_scale": (1.0, 1.0, 1.0),
        "top": 1.28,                           # hair tips, for camera framing
        "blend": 0.1,
        "bumps": 0.0,
        "body_follow": 0.2,
        "base": (0.16, 0.26, 0.68),
        "dark": (0.07, 0.07, 0.09),
        "line": (0.98, 0.82, 0.32),
        "burn": 0.0,
        "net": 0.0,
        "gloss": 0.4,
        "limbs": False,
        "limb_color": (0.16, 0.26, 0.68),
        "glove_color": (0.96, 0.82, 0.70),
        "face_size": 0.84,
        "face_y": 0.37,
        "face_tint": (1.0, 1.0, 1.0),
        "tint_amount": 0.0,
    },
    {
        # A cartoon sea sponge in the SpongeBob style: yellow, porous and
        # square, with a white shirt and red tie, brown square pants, and
        # striped socks. Shows your eyes and mouth; with your face off (M) it
        # uses its own big eyes and buck-toothed grin, driven by your face.
        "name": "Sea Sponge",
        "shape": "sponge",                     # custom model (see mapSponge in renderer.py)
        "default_mask": "features",
        "head": (0.0, 0.30, 0.0, 0.55),
        "head_scale": (1.0, 1.0, 1.0),
        "body": (0.0, -0.82, 0.0, 0.40),
        "body_scale": (1.0, 1.0, 1.0),
        "top": 0.92,
        "head_follow": 0.6,                    # the whole sponge turns, so keep it gentle
        "body_follow": 0.15,
        "blend": 0.1,
        "bumps": 0.0,
        "base": (1.0, 0.90, 0.25),
        "dark": (0.70, 0.70, 0.12),
        "line": (0.05, 0.04, 0.03),
        "burn": 0.0,
        "net": 0.0,
        "gloss": 0.2,
        "limbs": False,
        "limb_color": (1.0, 0.90, 0.25),
        "glove_color": (1.0, 0.90, 0.25),
        "face_size": 1.50,                     # puts your eyes where its eyes are
        "face_y": 0.205,
        "face_tint": (1.0, 0.92, 0.45),        # warms your skin toward sponge-yellow
        "tint_amount": 0.45,
    },
    {
        # The original full-body peanut with stick arms and legs.
        "name": "Peanut Pal",
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
]
