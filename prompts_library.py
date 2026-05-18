"""
Scene + pose + outfit library for the AI image generator.
Used by cloud_swap.py to build random varied prompts.

NSFW levels:
  "off"      - clothed, swimwear, lingerie
  "tasteful" - artistic nudity, topless
  "explicit" - full nudity
"""

import random

OUTFITS = {
    "off": [
        "in a white string bikini",
        "in a tiny black bikini",
        "in a pink halter bikini and bottoms",
        "in lacy black lingerie",
        "in a sheer white nightgown",
        "in a silk slip dress, low cut",
        "in tight jean shorts and a crop top",
        "in a sports bra and yoga pants",
        "in a thin oversized t-shirt and panties",
        "in a fluffy white bath towel wrapped around her",
        "in a tight black bodycon dress",
        "in a wet white t-shirt",
        "in a fitted gym crop top and tiny shorts",
        "in just a man's button-down shirt unbuttoned",
        "in a halter neck mini-dress",
        "in a fitted denim jacket and bra",
        "in delicate lace underwear set",
        "in a sheer satin robe loosely tied",
        "in skin-tight latex shorts and crop top",
        "in a wet sheer dress clinging to her body",
    ],
    "tasteful": [
        "topless, hands modestly covering her chest",
        "wearing only panties, topless, side-on",
        "wrapped in a thin sheet showing shoulder",
        "wearing an open bath robe revealing cleavage",
        "in lingerie set, top strap fallen off shoulder",
        "topless with arms crossed over chest",
        "in a tiny string bikini, bottoms only, back to camera",
        "wearing only thigh-high socks, otherwise nude",
        "in an open silk robe, partial nudity",
        "topless, hair covering her chest",
        "wet and topless after a shower, towel at waist",
        "implied nudity behind a white bedsheet",
        "tasteful artistic nude, soft lighting",
        "sitting in bath, only shoulders and knees visible",
        "topless under a thin gauzy fabric",
    ],
    "explicit": [
        "fully nude, natural pose",
        "completely nude in bed, sheet half covering her",
        "nude in shower, water running over her",
        "topless, panties pulled down slightly",
        "fully nude lying on her back",
        "nude from behind, full body",
        "nude, sitting on the edge of the bed",
        "fully nude on the beach at sunset",
        "nude, knees pulled to chest",
        "fully nude in front of a mirror",
        "nude with a thin silver necklace and nothing else",
        "completely nude, kneeling pose",
        "nude lying on her stomach",
        "fully nude in a pool, water at her hips",
    ],
}

LOCATIONS = [
    "in a sunlit bedroom with white sheets",
    "in a hotel room with city view through window",
    "in a marble bathroom",
    "leaning against a kitchen counter, morning light",
    "on a tropical beach at golden hour",
    "by a luxury pool",
    "in a luxury walk-in closet",
    "in front of a full-length bedroom mirror",
    "on a balcony overlooking the ocean",
    "in a hot tub with steam rising",
    "in a rustic Airbnb cabin with wood walls",
    "on a velvet sofa in a dim apartment",
    "in a luxury sports car interior",
    "in a private gym with mirrors",
    "in a sauna, wood-lined",
    "in a Mediterranean villa garden",
    "lying on a sun lounger by the pool",
    "in a steamy bathroom after a shower",
    "in a fitness studio with floor-to-ceiling windows",
    "in a candle-lit boudoir",
    "in front of a marble fireplace",
    "in the back of a private jet",
    "in a luxury yacht cabin",
    "in a Tokyo hotel room with neon lights through the window",
    "on a king-sized hotel bed with white linen",
]

POSES = [
    "looking back over her shoulder at the camera",
    "lying on her stomach, propped up on elbows",
    "lying on her back, one knee bent up",
    "sitting cross-legged",
    "standing with weight on one hip",
    "running her fingers through her hair, eyes closed",
    "biting her lower lip, looking down",
    "leaning forward toward the camera",
    "stretching arms overhead, eyes closed",
    "playing with the bottom of her shirt",
    "adjusting her bikini strap",
    "looking up at the camera from below",
    "sitting on the edge of the bed, legs crossed",
    "leaning against the wall, one foot up",
    "arching her back, head tilted back",
    "playful candid laugh, hair in the air",
    "applying lipstick in the mirror",
    "drinking from a wine glass, looking at camera",
    "pulling her hair to one side",
    "kneeling on the bed, looking back",
    "selfie pose, mirror selfie, phone in hand",
    "yoga pose, downward dog",
    "stepping out of the pool, water dripping",
    "stretching after a workout, glowing skin",
    "casual posing, foot on a chair",
    "lying on her side, head propped on hand",
    "hand on hip, sassy expression",
    "post-shower, towel-drying hair",
]

LIGHTING = [
    "soft natural daylight from a window",
    "golden hour warm light",
    "harsh direct sunlight",
    "dim bedroom lighting",
    "candlelight, warm and intimate",
    "neon-pink glow from a sign",
    "morning light coming through sheer curtains",
    "blue hour soft light",
    "studio softbox lighting",
    "harsh on-camera flash, vivid colors",
    "moody low-key lighting",
    "bright midday sun by the pool",
    "warm tungsten room lighting",
]

CAMERA_STYLE = [
    "iPhone selfie",
    "iPhone front camera, candid",
    "DSLR portrait, shallow depth of field",
    "professional fashion shoot",
    "boyfriend shot, casual photo",
    "instagram story selfie",
    "phone held in hand, mirror reflection",
    "tilted phone angle, lifestyle vlog",
    "polaroid style, slight grain",
    "amateur snapshot, slightly off-center",
]

QUALITY_TAGS = [
    "ultra high resolution, sharp focus, photorealistic, fine skin pores",
    "raw photo, unedited, natural skin texture",
    "candid photograph, sharp focus, film grain",
    "high resolution photo, natural lighting, detailed",
]


def make_random_prompt(nsfw_level="off", trigger_name=None):
    nsfw_level = nsfw_level if nsfw_level in OUTFITS else "off"
    outfit   = random.choice(OUTFITS[nsfw_level])
    location = random.choice(LOCATIONS)
    pose     = random.choice(POSES)
    lighting = random.choice(LIGHTING)
    camera   = random.choice(CAMERA_STYLE)
    quality  = random.choice(QUALITY_TAGS)
    parts = []
    if trigger_name:
        parts.append(f"a photo of {trigger_name} woman")
    parts.extend([outfit, location, pose, lighting, camera, quality])
    return ", ".join(parts)


def make_n_prompts(n, nsfw_level="off", trigger_name=None):
    seen = set()
    out = []
    attempts = 0
    while len(out) < n and attempts < n * 5:
        p = make_random_prompt(nsfw_level=nsfw_level, trigger_name=trigger_name)
        if p not in seen:
            seen.add(p)
            out.append(p)
        attempts += 1
    return out
