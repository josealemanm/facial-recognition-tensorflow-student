import json
from pathlib import Path

from PIL import Image

from facenet_student.preprocessing import center_crop_resize, preprocess_dataset
from tests.conftest import write_image


def test_center_crop_resize_returns_rgb_square() -> None:
    image = Image.new("RGBA", (120, 60), (20, 30, 40, 128))

    processed = center_crop_resize(image, 48)

    assert processed.mode == "RGB"
    assert processed.size == (48, 48)


def test_assume_cropped_preprocessing_preserves_labels_and_manifest(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    output = tmp_path / "processed"
    write_image(source / "identity_a" / "wide.png", color=(20, 80, 140), size=(120, 60))
    write_image(source / "identity_b" / "tall.jpg", color=(180, 80, 40), size=(60, 120))

    records = preprocess_dataset(source, output, size=64, assume_cropped=True)

    assert [record.status for record in records] == ["written", "written"]
    with Image.open(output / "identity_a" / "wide.jpg") as image:
        assert image.size == (64, 64)
        assert image.mode == "RGB"
    manifest_lines = (output / "preprocess_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(manifest_lines) == 2
    assert json.loads(manifest_lines[0])["class_name"] == "identity_a"
