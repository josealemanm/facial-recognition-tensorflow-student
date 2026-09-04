from pathlib import Path

import pytest

from facenet_student.data import discover_dataset, stratified_split
from facenet_student.errors import FacenetStudentError
from tests.conftest import write_image


def _dataset(root: Path) -> None:
    for class_name, color in (("zebra", (20, 30, 40)), ("alpha", (180, 120, 80))):
        for index in range(4):
            write_image(root / class_name / f"{index}.jpg", color=color)


def test_discover_dataset_assigns_stable_sorted_labels(tmp_path: Path) -> None:
    _dataset(tmp_path)
    (tmp_path / "alpha" / "notes.txt").write_text("ignored", encoding="utf-8")

    samples, class_names = discover_dataset(tmp_path)

    assert class_names == ["alpha", "zebra"]
    assert len(samples) == 8
    assert {sample.label for sample in samples if sample.class_name == "alpha"} == {0}
    assert {sample.label for sample in samples if sample.class_name == "zebra"} == {1}


def test_stratified_split_is_deterministic_and_keeps_every_class(tmp_path: Path) -> None:
    _dataset(tmp_path)
    samples, _ = discover_dataset(tmp_path)

    first_train, first_validation = stratified_split(
        samples,
        validation_fraction=0.25,
        seed=19,
    )
    second_train, second_validation = stratified_split(
        samples,
        validation_fraction=0.25,
        seed=19,
    )

    assert first_train == second_train
    assert first_validation == second_validation
    assert {sample.label for sample in first_train} == {0, 1}
    assert {sample.label for sample in first_validation} == {0, 1}
    assert {sample.path for sample in first_train}.isdisjoint(
        sample.path for sample in first_validation
    )


def test_discover_dataset_reports_empty_input(tmp_path: Path) -> None:
    with pytest.raises(FacenetStudentError, match="No eligible identity folders"):
        discover_dataset(tmp_path)
