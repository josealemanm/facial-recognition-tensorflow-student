from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from facenet_student.errors import FacenetStudentError

IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})


@dataclass(frozen=True, slots=True)
class ImageSample:
    """One labelled image discovered below an identity directory."""

    path: Path
    class_name: str
    label: int


def discover_dataset(
    root: Path | str,
    *,
    min_images_per_class: int = 1,
) -> tuple[list[ImageSample], list[str]]:
    """Discover a deterministic ``identity/image`` directory dataset."""

    root = Path(root)
    if min_images_per_class < 1:
        raise FacenetStudentError("min_images_per_class must be at least 1")
    if not root.is_dir():
        raise FacenetStudentError(f"Dataset directory does not exist: {root}")

    eligible: list[tuple[str, list[Path]]] = []
    rejected: list[str] = []
    for class_dir in sorted(root.iterdir(), key=lambda path: path.name.casefold()):
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue
        images = sorted(
            (
                path
                for path in class_dir.rglob("*")
                if path.is_file()
                and not path.name.startswith(".")
                and path.suffix.casefold() in IMAGE_EXTENSIONS
            ),
            key=lambda path: path.as_posix().casefold(),
        )
        if len(images) < min_images_per_class:
            rejected.append(f"{class_dir.name} ({len(images)})")
            continue
        eligible.append((class_dir.name, images))

    if not eligible:
        detail = f" Classes below the minimum: {', '.join(rejected)}." if rejected else ""
        raise FacenetStudentError(
            f"No eligible identity folders found in {root}.{detail} "
            "Expected root/identity_name/image.jpg."
        )

    class_names = [name for name, _ in eligible]
    samples = [
        ImageSample(path=image_path, class_name=class_name, label=label)
        for label, (class_name, image_paths) in enumerate(eligible)
        for image_path in image_paths
    ]
    return samples, class_names


def stratified_split(
    samples: list[ImageSample],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[ImageSample], list[ImageSample]]:
    """Split each class while keeping at least one sample on either side."""

    if not 0.0 < validation_fraction < 1.0:
        raise FacenetStudentError("validation_fraction must be between 0 and 1")

    by_label: dict[int, list[ImageSample]] = defaultdict(list)
    for sample in samples:
        by_label[sample.label].append(sample)

    rng = random.Random(seed)
    train: list[ImageSample] = []
    validation: list[ImageSample] = []
    for label in sorted(by_label):
        group = sorted(by_label[label], key=lambda sample: sample.path.as_posix())
        if len(group) < 2:
            raise FacenetStudentError(
                f"Class {group[0].class_name!r} needs at least 2 images for validation"
            )
        rng.shuffle(group)
        validation_count = min(
            len(group) - 1,
            max(1, round(len(group) * validation_fraction)),
        )
        validation.extend(group[:validation_count])
        train.extend(group[validation_count:])

    train.sort(key=lambda sample: (sample.label, sample.path.as_posix()))
    validation.sort(key=lambda sample: (sample.label, sample.path.as_posix()))
    return train, validation


def dataset_summary(root: Path | str, *, min_images_per_class: int = 1) -> dict[str, object]:
    samples, class_names = discover_dataset(root, min_images_per_class=min_images_per_class)
    counts = Counter(sample.class_name for sample in samples)
    return {
        "root": str(Path(root)),
        "classes": class_names,
        "class_counts": {name: counts[name] for name in class_names},
        "num_classes": len(class_names),
        "num_images": len(samples),
    }
