from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from facenet_student.errors import FacenetStudentError

CLASSIFIER_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class EmbeddingArchive:
    embeddings: np.ndarray
    labels: np.ndarray
    class_names: list[str]
    paths: list[str]


def load_embedding_archive(path: Path | str) -> EmbeddingArchive:
    """Load and validate an embedding archive without pickle support."""

    path = Path(path)
    if not path.is_file():
        raise FacenetStudentError(f"Embedding archive does not exist: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {"embeddings", "labels", "class_names", "paths", "schema_version"}
            missing = required.difference(archive.files)
            if missing:
                raise FacenetStudentError(
                    f"Embedding archive is missing fields: {', '.join(sorted(missing))}"
                )
            schema_version = int(np.asarray(archive["schema_version"]).item())
            if schema_version != 1:
                raise FacenetStudentError(
                    f"Unsupported embedding schema {schema_version}; expected version 1"
                )
            embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
            labels = np.asarray(archive["labels"], dtype=np.int64)
            class_names = [str(value) for value in archive["class_names"].tolist()]
            paths = [str(value) for value in archive["paths"].tolist()]
    except FacenetStudentError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise FacenetStudentError(f"Could not read embedding archive {path}: {exc}") from exc

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise FacenetStudentError("embeddings must be a non-empty two-dimensional matrix")
    if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
        raise FacenetStudentError("labels must contain one value per embedding")
    if len(paths) != embeddings.shape[0]:
        raise FacenetStudentError("paths must contain one value per embedding")
    if len(class_names) < 1:
        raise FacenetStudentError("class_names cannot be empty")
    if labels.min() < 0 or labels.max() >= len(class_names):
        raise FacenetStudentError("labels reference a missing class name")
    if not np.all(np.isfinite(embeddings)):
        raise FacenetStudentError("embeddings contain non-finite values")
    return EmbeddingArchive(embeddings, labels, class_names, paths)


def _load_classifier(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FacenetStudentError(f"Classifier does not exist: {path}")
    try:
        artifact = joblib.load(path)
    except Exception as exc:
        raise FacenetStudentError(
            f"Could not load classifier {path}. Load only trusted joblib files."
        ) from exc
    if not isinstance(artifact, dict):
        raise FacenetStudentError("Classifier artifact has an invalid structure")
    if artifact.get("schema_version") != CLASSIFIER_SCHEMA_VERSION:
        raise FacenetStudentError("Classifier artifact has an unsupported schema")
    required = {"model", "class_names", "embedding_dim"}
    missing = required.difference(artifact)
    if missing:
        raise FacenetStudentError(
            f"Classifier artifact is missing fields: {', '.join(sorted(missing))}"
        )
    return artifact


def train_classifier(
    embeddings_path: Path | str,
    classifier_out: Path | str,
    *,
    c_value: float = 1.0,
    seed: int = 42,
) -> dict[str, object]:
    """Fit and save a standardization + linear-SVM pipeline."""

    if c_value <= 0:
        raise FacenetStudentError("c_value must be positive")
    archive = load_embedding_archive(embeddings_path)
    unique_labels = np.unique(archive.labels)
    if unique_labels.size < 2:
        raise FacenetStudentError("Classifier training requires at least 2 classes")
    missing_labels = set(range(len(archive.class_names))).difference(int(x) for x in unique_labels)
    if missing_labels:
        raise FacenetStudentError("Training embeddings do not contain every declared class")

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "svc",
                SVC(
                    C=c_value,
                    kernel="linear",
                    probability=True,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(archive.embeddings, archive.labels)
    training_accuracy = float(model.score(archive.embeddings, archive.labels))

    classifier_out = Path(classifier_out)
    if classifier_out.suffix != ".joblib":
        raise FacenetStudentError("Classifier output must use the .joblib extension")
    classifier_out.parent.mkdir(parents=True, exist_ok=True)
    artifact: dict[str, Any] = {
        "schema_version": CLASSIFIER_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "class_names": archive.class_names,
        "embedding_dim": int(archive.embeddings.shape[1]),
        "seed": seed,
        "c_value": c_value,
    }
    joblib.dump(artifact, classifier_out, compress=3)
    return {
        "output": str(classifier_out),
        "num_images": int(archive.embeddings.shape[0]),
        "num_classes": len(archive.class_names),
        "embedding_dim": int(archive.embeddings.shape[1]),
        "training_accuracy": training_accuracy,
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def evaluate_classifier(
    embeddings_path: Path | str,
    classifier_path: Path | str,
    report_out: Path | str,
) -> dict[str, object]:
    """Evaluate a saved classifier on an independently embedded test directory."""

    archive = load_embedding_archive(embeddings_path)
    artifact = _load_classifier(classifier_path)
    class_names = [str(value) for value in artifact["class_names"]]
    if class_names != archive.class_names:
        raise FacenetStudentError(
            "Test class names do not exactly match the classifier's training labels"
        )
    if int(artifact["embedding_dim"]) != archive.embeddings.shape[1]:
        raise FacenetStudentError("Embedding dimensions do not match the classifier")

    model = artifact["model"]
    predicted = np.asarray(model.predict(archive.embeddings), dtype=np.int64)
    probabilities = np.asarray(model.predict_proba(archive.embeddings), dtype=np.float64)
    probability_labels = np.asarray(model.named_steps["svc"].classes_, dtype=np.int64)
    best_columns = np.argmax(probabilities, axis=1)
    confidence = probabilities[np.arange(probabilities.shape[0]), best_columns]
    confidence_labels = probability_labels[best_columns]
    if not np.array_equal(predicted, confidence_labels):
        raise FacenetStudentError("Classifier probability labels are inconsistent")

    labels = list(range(len(class_names)))
    accuracy = float(accuracy_score(archive.labels, predicted))
    metrics = classification_report(
        archive.labels,
        predicted,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(archive.labels, predicted, labels=labels)
    predictions = [
        {
            "path": archive.paths[index],
            "actual": class_names[int(archive.labels[index])],
            "predicted": class_names[int(predicted[index])],
            "confidence": float(confidence[index]),
            "correct": bool(predicted[index] == archive.labels[index]),
        }
        for index in range(len(predicted))
    ]
    report: dict[str, object] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "accuracy": accuracy,
        "num_images": len(predictions),
        "num_classes": len(class_names),
        "class_names": class_names,
        "confusion_matrix": matrix.tolist(),
        "classification_report": _json_value(metrics),
        "predictions": predictions,
        "interpretation": (
            "Scores are estimated closed-set classification probabilities, not proof of identity."
        ),
    }

    report_out = Path(report_out)
    if report_out.suffix != ".json":
        raise FacenetStudentError("Evaluation report must use the .json extension")
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
