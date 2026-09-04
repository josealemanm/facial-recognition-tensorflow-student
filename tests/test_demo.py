import hashlib
from pathlib import Path

from facenet_student.demo import generate_demo_dataset


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_demo_data_is_reproducible_and_separates_splits(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    summary = generate_demo_dataset(
        first,
        identities=2,
        train_per_identity=2,
        test_per_identity=1,
        image_size=64,
        seed=11,
    )
    generate_demo_dataset(
        second,
        identities=2,
        train_per_identity=2,
        test_per_identity=1,
        image_size=64,
        seed=11,
    )

    first_image = first / "train" / "synthetic_identity_01" / "image_001.jpg"
    second_image = second / "train" / "synthetic_identity_01" / "image_001.jpg"
    test_image = first / "test" / "synthetic_identity_01" / "image_001.jpg"
    assert summary["synthetic_only"] is True
    assert summary["total_images"] == 6
    assert _sha256(first_image) == _sha256(second_image)
    assert _sha256(first_image) != _sha256(test_image)
