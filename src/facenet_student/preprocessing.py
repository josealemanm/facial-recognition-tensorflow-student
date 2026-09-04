from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from facenet_student.data import discover_dataset
from facenet_student.errors import FacenetStudentError

NoFacePolicy = Literal["skip", "use-full", "error"]


@dataclass(frozen=True, slots=True)
class PreprocessRecord:
    source: str
    output: str | None
    class_name: str
    status: str
    detail: str


def center_crop_resize(image: Image.Image, size: int) -> Image.Image:
    """Return an EXIF-corrected RGB square using a high-quality center crop."""

    if size < 32:
        raise FacenetStudentError("Output size must be at least 32 pixels")
    image = ImageOps.exif_transpose(image).convert("RGB")
    return ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS)


def _load_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise FacenetStudentError(
            "OpenCV is required for face detection. Install with "
            "`python -m pip install -e '.[vision]'`, or use --assume-cropped."
        ) from exc
    return cv2


def _largest_box(boxes: np.ndarray) -> tuple[int, int, int, int] | None:
    if len(boxes) == 0:
        return None
    x, y, width, height = max(boxes, key=lambda box: int(box[2]) * int(box[3]))
    return int(x), int(y), int(width), int(height)


def _align_detected_face(rgb: np.ndarray, size: int, margin: float) -> Image.Image | None:
    cv2 = _load_cv2()
    if not 0.0 <= margin <= 1.0:
        raise FacenetStudentError("margin must be between 0 and 1")

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade_root = Path(cv2.data.haarcascades)
    face_detector = cv2.CascadeClassifier(str(cascade_root / "haarcascade_frontalface_default.xml"))
    eye_detector = cv2.CascadeClassifier(str(cascade_root / "haarcascade_eye.xml"))
    if face_detector.empty() or eye_detector.empty():
        raise FacenetStudentError("OpenCV's bundled Haar cascade files could not be loaded")

    min_side = max(24, min(rgb.shape[:2]) // 10)
    boxes = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_side, min_side),
    )
    box = _largest_box(boxes)
    if box is None:
        return None

    x, y, width, height = box
    margin_x = round(width * margin)
    margin_y = round(height * margin)
    left = max(0, x - margin_x)
    top = max(0, y - margin_y)
    right = min(rgb.shape[1], x + width + margin_x)
    bottom = min(rgb.shape[0], y + height + margin_y)
    face_rgb = rgb[top:bottom, left:right]
    face_gray = gray[top:bottom, left:right]

    upper_gray = face_gray[: max(1, round(face_gray.shape[0] * 0.65)), :]
    eye_boxes = eye_detector.detectMultiScale(
        upper_gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(max(8, width // 12), max(8, height // 12)),
    )
    if len(eye_boxes) >= 2:
        candidates = sorted(
            eye_boxes,
            key=lambda eye: int(eye[2]) * int(eye[3]),
            reverse=True,
        )[:4]
        best_pair = max(
            (
                (first, second)
                for index, first in enumerate(candidates)
                for second in candidates[index + 1 :]
            ),
            key=lambda pair: abs(
                (float(pair[0][0]) + float(pair[0][2]) / 2)
                - (float(pair[1][0]) + float(pair[1][2]) / 2)
            ),
        )
        centers = sorted(
            (
                (float(eye[0]) + float(eye[2]) / 2, float(eye[1]) + float(eye[3]) / 2)
                for eye in best_pair
            ),
            key=lambda point: point[0],
        )
        delta_x = centers[1][0] - centers[0][0]
        delta_y = centers[1][1] - centers[0][1]
        if abs(delta_x) > 1.0:
            angle = math.degrees(math.atan2(delta_y, delta_x))
            rotation = cv2.getRotationMatrix2D(
                (face_rgb.shape[1] / 2, face_rgb.shape[0] / 2),
                angle,
                1.0,
            )
            face_rgb = cv2.warpAffine(
                face_rgb,
                rotation,
                (face_rgb.shape[1], face_rgb.shape[0]),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REFLECT_101,
            )

    return center_crop_resize(Image.fromarray(face_rgb, mode="RGB"), size)


def preprocess_dataset(
    input_dir: Path | str,
    output_dir: Path | str,
    *,
    size: int = 160,
    assume_cropped: bool = False,
    on_no_face: NoFacePolicy = "skip",
    margin: float = 0.18,
    overwrite: bool = False,
) -> list[PreprocessRecord]:
    """Preprocess an identity-labelled directory and write a JSONL manifest."""

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if on_no_face not in {"skip", "use-full", "error"}:
        raise FacenetStudentError(f"Unsupported no-face policy: {on_no_face}")

    samples, _ = discover_dataset(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[PreprocessRecord] = []

    for sample in samples:
        relative_source = sample.path.relative_to(input_dir).as_posix()
        relative_output = Path(sample.class_name) / f"{sample.path.stem}.jpg"
        destination = output_dir / relative_output
        if destination.exists() and not overwrite:
            records.append(
                PreprocessRecord(
                    source=relative_source,
                    output=relative_output.as_posix(),
                    class_name=sample.class_name,
                    status="existing",
                    detail="Output already exists; left unchanged.",
                )
            )
            continue

        try:
            with Image.open(sample.path) as opened:
                source_image = ImageOps.exif_transpose(opened).convert("RGB")
                if assume_cropped:
                    processed = center_crop_resize(source_image, size)
                    detail = "Centered and resized (--assume-cropped)."
                else:
                    processed = _align_detected_face(np.asarray(source_image), size, margin)
                    detail = "Largest detected face cropped; eye rotation used when available."
                    if processed is None and on_no_face == "use-full":
                        processed = center_crop_resize(source_image, size)
                        detail = "No face detected; full image used by policy."
                    elif processed is None and on_no_face == "error":
                        raise FacenetStudentError(f"No face detected in {relative_source}")

            if processed is None:
                records.append(
                    PreprocessRecord(
                        source=relative_source,
                        output=None,
                        class_name=sample.class_name,
                        status="skipped",
                        detail="No face detected.",
                    )
                )
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp")
            processed.save(temporary, format="JPEG", quality=95, optimize=True)
            temporary.replace(destination)
            records.append(
                PreprocessRecord(
                    source=relative_source,
                    output=relative_output.as_posix(),
                    class_name=sample.class_name,
                    status="written",
                    detail=detail,
                )
            )
        except (OSError, UnidentifiedImageError) as exc:
            records.append(
                PreprocessRecord(
                    source=relative_source,
                    output=None,
                    class_name=sample.class_name,
                    status="error",
                    detail=str(exc),
                )
            )

    manifest_path = output_dir / "preprocess_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for record in records:
            manifest.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    return records
