from pathlib import Path

from PIL import Image


def write_image(
    path: Path,
    *,
    color: tuple[int, int, int],
    size: tuple[int, int] = (80, 60),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
