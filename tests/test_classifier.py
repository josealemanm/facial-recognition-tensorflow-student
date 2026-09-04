import json
from pathlib import Path

import numpy as np
import pytest

from facenet_student.classifier import (
    evaluate_classifier,
    load_embedding_archive,
    train_classifier,
)
from facenet_student.errors import FacenetStudentError


def _write_archive(path: Path, embeddings: np.ndarray, labels: np.ndarray) -> None:
    class_names = np.asarray(["left", "right"], dtype=np.str_)
    paths = np.asarray([f"sample_{index}.jpg" for index in range(len(labels))], dtype=np.str_)
    np.savez_compressed(
        path,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        class_names=class_names,
        paths=paths,
        schema_version=np.asarray(1, dtype=np.int64),
    )


def test_classifier_round_trip_and_json_report(tmp_path: Path) -> None:
    train_path = tmp_path / "train.npz"
    test_path = tmp_path / "test.npz"
    classifier_path = tmp_path / "classifier.joblib"
    report_path = tmp_path / "report.json"
    train_vectors = np.asarray(
        [[-2.0, -1.0], [-1.8, -1.1], [-1.5, -0.8], [1.5, 0.8], [1.8, 1.1], [2.0, 1.0]]
    )
    train_labels = np.asarray([0, 0, 0, 1, 1, 1])
    test_vectors = np.asarray([[-1.9, -0.9], [-1.4, -1.0], [1.4, 0.9], [1.9, 1.0]])
    test_labels = np.asarray([0, 0, 1, 1])
    _write_archive(train_path, train_vectors, train_labels)
    _write_archive(test_path, test_vectors, test_labels)

    training = train_classifier(train_path, classifier_path, seed=5)
    report = evaluate_classifier(test_path, classifier_path, report_path)

    assert training["num_classes"] == 2
    assert report["accuracy"] == pytest.approx(1.0)
    assert report["confusion_matrix"] == [[2, 0], [0, 2]]
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["num_images"] == 4
    assert all(item["confidence"] <= 1.0 for item in persisted["predictions"])


def test_archive_loader_rejects_missing_fields(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.npz"
    np.savez_compressed(invalid, embeddings=np.zeros((2, 2), dtype=np.float32))

    with pytest.raises(FacenetStudentError, match="missing fields"):
        load_embedding_archive(invalid)
