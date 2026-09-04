from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from facenet_student.errors import FacenetStudentError

PALETTES = (
    ((236, 194, 154), (39, 67, 95), (89, 52, 37), (214, 225, 236)),
    ((158, 104, 74), (83, 42, 71), (28, 24, 30), (235, 213, 188)),
    ((217, 166, 128), (36, 112, 92), (205, 151, 76), (214, 235, 225)),
    ((113, 74, 55), (103, 76, 150), (38, 28, 25), (231, 220, 242)),
    ((242, 205, 176), (173, 72, 65), (112, 68, 36), (241, 226, 203)),
)


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def _draw_synthetic_identity(
    identity_index: int,
    sample_index: int,
    *,
    image_size: int,
    seed: int,
) -> Image.Image:
    """Draw a non-person, cartoon-like pattern for integration tests."""

    rng = random.Random(seed + identity_index * 10_007 + sample_index * 97)
    skin, accent, hair, background = PALETTES[identity_index % len(PALETTES)]
    canvas = Image.new("RGB", (image_size, image_size), background)
    draw = ImageDraw.Draw(canvas)

    jitter_x = rng.randint(-3, 3)
    jitter_y = rng.randint(-3, 3)
    face_left = image_size * 0.20 + jitter_x
    face_top = image_size * 0.12 + jitter_y
    face_right = image_size * 0.80 + jitter_x
    face_bottom = image_size * 0.91 + jitter_y
    draw.ellipse((face_left, face_top, face_right, face_bottom), fill=skin, outline=accent, width=3)

    hair_height = image_size * (0.22 + 0.02 * (identity_index % 3))
    draw.pieslice(
        (face_left - 1, face_top - 2, face_right + 1, face_top + hair_height * 2),
        180,
        360,
        fill=hair,
    )

    eye_y = image_size * (0.43 + rng.uniform(-0.015, 0.015))
    eye_spacing = image_size * (0.14 + 0.012 * identity_index)
    eye_radius = max(3, round(image_size * (0.035 + 0.003 * (identity_index % 2))))
    center_x = image_size / 2 + jitter_x
    for eye_x in (center_x - eye_spacing, center_x + eye_spacing):
        box = (
            eye_x - eye_radius,
            eye_y - eye_radius,
            eye_x + eye_radius,
            eye_y + eye_radius,
        )
        draw.ellipse(box, fill=(250, 250, 245), outline=accent, width=2)
        pupil_radius = max(1, eye_radius // 3)
        draw.ellipse(
            (
                eye_x - pupil_radius,
                eye_y - pupil_radius,
                eye_x + pupil_radius,
                eye_y + pupil_radius,
            ),
            fill=accent,
        )

    nose_x = center_x + (identity_index - 2) * image_size * 0.01
    draw.line(
        (
            nose_x,
            image_size * 0.50,
            nose_x - image_size * 0.035,
            image_size * 0.64,
            nose_x + image_size * 0.035,
            image_size * 0.64,
        ),
        fill=accent,
        width=2,
    )

    mouth_width = image_size * (0.20 + identity_index * 0.015)
    mouth_y = image_size * (0.72 + rng.uniform(-0.015, 0.015))
    mouth_box = (
        center_x - mouth_width / 2,
        mouth_y - image_size * 0.04,
        center_x + mouth_width / 2,
        mouth_y + image_size * 0.05,
    )
    start_angle = 5 if identity_index % 2 == 0 else 180
    end_angle = 175 if identity_index % 2 == 0 else 355
    draw.arc(mouth_box, start=start_angle, end=end_angle, fill=accent, width=3)

    if identity_index % 3 == 1:
        glasses_y = eye_y
        glasses_half = image_size * 0.075
        for eye_x in (center_x - eye_spacing, center_x + eye_spacing):
            draw.rectangle(
                (
                    eye_x - glasses_half,
                    glasses_y - glasses_half * 0.7,
                    eye_x + glasses_half,
                    glasses_y + glasses_half * 0.7,
                ),
                outline=hair,
                width=2,
            )
        draw.line(
            (
                center_x - eye_spacing + glasses_half,
                eye_y,
                center_x + eye_spacing - glasses_half,
                eye_y,
            ),
            fill=hair,
            width=2,
        )
    elif identity_index % 3 == 2:
        mole_x = center_x + image_size * 0.13
        mole_y = image_size * 0.66
        draw.ellipse((mole_x - 2, mole_y - 2, mole_x + 2, mole_y + 2), fill=accent)

    angle = rng.uniform(-4.0, 4.0)
    canvas = canvas.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=background)
    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 0.35)))

    pixels = np.asarray(canvas, dtype=np.int16)
    noise = np.random.default_rng(seed + identity_index * 503 + sample_index).normal(
        0.0, 2.2, pixels.shape
    )
    pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
    brightness = rng.randint(-5, 5)
    pixels = np.clip(pixels.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def generate_demo_dataset(
    output_dir: Path | str,
    *,
    identities: int = 3,
    train_per_identity: int = 18,
    test_per_identity: int = 6,
    image_size: int = 128,
    seed: int = 7,
    overwrite: bool = False,
) -> dict[str, object]:
    """Generate deterministic images for a dependency-light pipeline demo."""

    output_dir = Path(output_dir)
    if identities < 2:
        raise FacenetStudentError("The demo needs at least 2 synthetic identities")
    if train_per_identity < 2 or test_per_identity < 1:
        raise FacenetStudentError("Use at least 2 train and 1 test image per identity")
    if image_size < 48:
        raise FacenetStudentError("image_size must be at least 48 pixels")
    if not overwrite and output_dir.exists() and any(output_dir.rglob("*.jpg")):
        raise FacenetStudentError(
            f"Demo images already exist below {output_dir}; pass --overwrite to replace them"
        )

    counts: dict[str, dict[str, int]] = {"train": {}, "test": {}}
    for split, count in (("train", train_per_identity), ("test", test_per_identity)):
        for identity_index in range(identities):
            class_name = f"synthetic_identity_{identity_index + 1:02d}"
            class_dir = output_dir / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            counts[split][class_name] = count
            split_offset = 100_000 if split == "test" else 0
            for sample_index in range(count):
                image = _draw_synthetic_identity(
                    identity_index,
                    sample_index + split_offset,
                    image_size=image_size,
                    seed=seed,
                )
                destination = class_dir / f"image_{sample_index + 1:03d}.jpg"
                image.save(destination, format="JPEG", quality=94, optimize=True)

    return {
        "output_dir": str(output_dir),
        "synthetic_only": True,
        "seed": seed,
        "image_size": image_size,
        "identities": identities,
        "counts": counts,
        "total_images": identities * (train_per_identity + test_per_identity),
        "rotation_note": f"Each image varies by up to {math.ceil(4.0)} degrees.",
    }
