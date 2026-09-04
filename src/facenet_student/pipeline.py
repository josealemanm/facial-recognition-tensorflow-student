from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from facenet_student.classifier import evaluate_classifier, train_classifier
from facenet_student.demo import generate_demo_dataset
from facenet_student.errors import FacenetStudentError
from facenet_student.model import export_embeddings, train_embedding_model
from facenet_student.preprocessing import preprocess_dataset


def run_demo_pipeline(
    work_dir: Path | str,
    *,
    epochs: int = 8,
    seed: int = 7,
    image_size: int = 96,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run every pipeline stage with generated non-person images."""

    work_dir = Path(work_dir)
    known_outputs = (
        work_dir / "artifacts" / "embedder.keras",
        work_dir / "artifacts" / "classifier.joblib",
        work_dir / "artifacts" / "evaluation.json",
    )
    if not overwrite and any(path.exists() for path in known_outputs):
        raise FacenetStudentError(
            f"Demo artifacts already exist below {work_dir}; pass --overwrite to replace them"
        )

    raw_dir = work_dir / "data" / "raw"
    processed_dir = work_dir / "data" / "processed"
    artifacts_dir = work_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    generation = generate_demo_dataset(
        raw_dir,
        identities=3,
        train_per_identity=18,
        test_per_identity=6,
        image_size=image_size,
        seed=seed,
        overwrite=overwrite,
    )
    train_records = preprocess_dataset(
        raw_dir / "train",
        processed_dir / "train",
        size=image_size,
        assume_cropped=True,
        overwrite=overwrite,
    )
    test_records = preprocess_dataset(
        raw_dir / "test",
        processed_dir / "test",
        size=image_size,
        assume_cropped=True,
        overwrite=overwrite,
    )
    model_metadata = train_embedding_model(
        processed_dir / "train",
        artifacts_dir / "embedder.keras",
        artifacts_dir / "embedder.json",
        backbone="tiny",
        image_size=image_size,
        embedding_dim=128,
        batch_size=12,
        epochs=epochs,
        validation_fraction=0.2,
        learning_rate=2e-3,
        seed=seed,
        pretrained=False,
    )
    train_embedding_summary = export_embeddings(
        processed_dir / "train",
        artifacts_dir / "embedder.keras",
        artifacts_dir / "train_embeddings.npz",
        batch_size=24,
        seed=seed,
    )
    classifier_summary = train_classifier(
        artifacts_dir / "train_embeddings.npz",
        artifacts_dir / "classifier.joblib",
        seed=seed,
    )
    test_embedding_summary = export_embeddings(
        processed_dir / "test",
        artifacts_dir / "embedder.keras",
        artifacts_dir / "test_embeddings.npz",
        batch_size=24,
        seed=seed,
    )
    evaluation = evaluate_classifier(
        artifacts_dir / "test_embeddings.npz",
        artifacts_dir / "classifier.joblib",
        artifacts_dir / "evaluation.json",
    )

    summary: dict[str, object] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "work_dir": str(work_dir),
        "synthetic_only": True,
        "warning": (
            "The synthetic demo validates software integration only; it is not a human-face "
            "accuracy benchmark."
        ),
        "generation": generation,
        "preprocessing": {
            "train_written": sum(record.status == "written" for record in train_records),
            "test_written": sum(record.status == "written" for record in test_records),
        },
        "model": model_metadata,
        "train_embeddings": train_embedding_summary,
        "classifier": classifier_summary,
        "test_embeddings": test_embedding_summary,
        "evaluation": {
            "accuracy": evaluation["accuracy"],
            "num_images": evaluation["num_images"],
            "report": str(artifacts_dir / "evaluation.json"),
        },
    }
    (work_dir / "demo_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
