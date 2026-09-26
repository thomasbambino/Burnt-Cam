"""Character skins.

Every character is built from the same signed-distance "rig": a head lobe and
a body lobe (ellipsoids) smoothly blended together, optional cartoon limbs,
and a procedural shell material.  Your face is projected onto the front of
the head lobe.  Add a new dict to SKINS to create a new character.

Shape units: the character stands roughly 2.8 units tall, feet at y = -1.6.
Colors are 0..1 RGB.

Optional keys (default off): "eye_domes" (radius of bulging eyes placed where
your eyes land on the face), "tail" (tail + back crest), "toes", "net_scale" (size of the scale/netting
cells), "belly" + "belly_amount", "mouth_line" (strength of a drawn-on grin),
"limb_radius".
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
        # An original cartoon lizard in the spirit of Tom Lizard from Pixar's
        # "Hoppers": sea-green scales and big eyes that drift apart.
        "name": "Lizard",
        "head": (0.0, 0.42, 0.12, 0.60),
        "head_scale": (1.20, 0.70, 1.25),      # wide, flat head with a long snout
        "body": (0.0, -0.62, 0.0, 0.50),
        "body_scale": (0.95, 1.30, 0.85),      # slim, upright body
        "blend": 0.30,
        "bumps": 0.004,
        "base": (0.33, 0.66, 0.52),            # sea green
        "dark": (0.12, 0.36, 0.30),            # darker blotches
        "line": (0.20, 0.45, 0.36),            # scale edges
        "burn": 0.45,
        "net": 0.45,
        "net_scale": 2.4,
        "gloss": 0.45,
        "limbs": True,
        "limb_radius": 0.075,
        "limb_color": (0.28, 0.58, 0.45),
        "glove_color": (0.30, 0.62, 0.48),     # hands
        "belly": (0.86, 0.88, 0.62),
        "belly_amount": 0.8,
        "tail": True,                          # also adds a spiky crest
        "toes": True,
        "eye_domes": 0.19,
        "mouth_line": 0.45,
        "face_size": 1.60,
        "face_y": 0.57,
        "face_tint": (0.80, 0.95, 0.82),
        "tint_amount": 0.30,
    },
]
