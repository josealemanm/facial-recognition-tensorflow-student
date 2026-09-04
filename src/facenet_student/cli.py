from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from facenet_student import __version__
from facenet_student.classifier import evaluate_classifier, train_classifier
from facenet_student.data import dataset_summary
from facenet_student.demo import generate_demo_dataset
from facenet_student.errors import FacenetStudentError
from facenet_student.model import export_embeddings, train_embedding_model
from facenet_student.pipeline import run_demo_pipeline
from facenet_student.preprocessing import preprocess_dataset


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _fraction(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="facenet-student",
        description=(
            "Educational TensorFlow image-embedding and SVM pipeline. Use only authorized data."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    scan = commands.add_parser("scan", help="Summarize an identity-folder dataset")
    scan.add_argument("--input-dir", type=Path, required=True)
    scan.add_argument("--min-images-per-class", type=_positive_int, default=1)

    demo_data = commands.add_parser(
        "demo-data",
        help="Generate deterministic, non-person images for testing",
    )
    demo_data.add_argument("--output-dir", type=Path, required=True)
    demo_data.add_argument("--identities", type=_positive_int, default=3)
    demo_data.add_argument("--train-per-identity", type=_positive_int, default=18)
    demo_data.add_argument("--test-per-identity", type=_positive_int, default=6)
    demo_data.add_argument("--image-size", type=_positive_int, default=128)
    demo_data.add_argument("--seed", type=int, default=7)
    demo_data.add_argument("--overwrite", action="store_true")

    preprocess = commands.add_parser(
        "preprocess",
        help="Detect/align faces or normalize already cropped inputs",
    )
    preprocess.add_argument("--input-dir", type=Path, required=True)
    preprocess.add_argument("--output-dir", type=Path, required=True)
    preprocess.add_argument("--size", type=_positive_int, default=160)
    preprocess.add_argument("--assume-cropped", action="store_true")
    preprocess.add_argument(
        "--on-no-face",
        choices=("skip", "use-full", "error"),
        default="skip",
    )
    preprocess.add_argument("--margin", type=float, default=0.18)
    preprocess.add_argument("--overwrite", action="store_true")

    train_embedder = commands.add_parser(
        "train-embedder",
        help="Train a Keras identity model with a normalized embedding layer",
    )
    train_embedder.add_argument("--input-dir", type=Path, required=True)
    train_embedder.add_argument("--model-out", type=Path, required=True)
    train_embedder.add_argument("--metadata-out", type=Path, required=True)
    train_embedder.add_argument(
        "--backbone",
        choices=("tiny", "mobilenet_v2"),
        default="mobilenet_v2",
    )
    train_embedder.add_argument("--image-size", type=_positive_int, default=160)
    train_embedder.add_argument("--embedding-dim", type=_positive_int, default=128)
    train_embedder.add_argument("--batch-size", type=_positive_int, default=16)
    train_embedder.add_argument("--epochs", type=_positive_int, default=20)
    train_embedder.add_argument("--validation-fraction", type=_fraction, default=0.2)
    train_embedder.add_argument("--learning-rate", type=float, default=1e-3)
    train_embedder.add_argument("--seed", type=int, default=42)
    train_embedder.add_argument(
        "--no-pretrained",
        action="store_false",
        dest="pretrained",
        help="Do not download ImageNet weights for MobileNetV2",
    )
    train_embedder.set_defaults(pretrained=True)

    embed = commands.add_parser("embed", help="Export normalized vectors from a saved model")
    embed.add_argument("--input-dir", type=Path, required=True)
    embed.add_argument("--model", type=Path, required=True)
    embed.add_argument("--output", type=Path, required=True)
    embed.add_argument("--batch-size", type=_positive_int, default=32)
    embed.add_argument("--seed", type=int, default=42)

    train_svm = commands.add_parser(
        "train-classifier",
        help="Train a probability-enabled linear SVM",
    )
    train_svm.add_argument("--embeddings", type=Path, required=True)
    train_svm.add_argument("--classifier-out", type=Path, required=True)
    train_svm.add_argument("--c-value", type=float, default=1.0)
    train_svm.add_argument("--seed", type=int, default=42)

    evaluate = commands.add_parser(
        "evaluate",
        help="Evaluate a classifier on held-out embeddings",
    )
    evaluate.add_argument("--embeddings", type=Path, required=True)
    evaluate.add_argument("--classifier", type=Path, required=True)
    evaluate.add_argument("--report-out", type=Path, required=True)

    demo = commands.add_parser(
        "demo",
        help="Run every stage with generated non-person images",
    )
    demo.add_argument("--work-dir", type=Path, required=True)
    demo.add_argument("--epochs", type=_positive_int, default=8)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--image-size", type=_positive_int, default=96)
    demo.add_argument("--overwrite", action="store_true")
    return parser


def _dispatch(args: argparse.Namespace) -> dict[str, object]:
    if args.command == "scan":
        return dataset_summary(
            args.input_dir,
            min_images_per_class=args.min_images_per_class,
        )
    if args.command == "demo-data":
        return generate_demo_dataset(
            args.output_dir,
            identities=args.identities,
            train_per_identity=args.train_per_identity,
            test_per_identity=args.test_per_identity,
            image_size=args.image_size,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    if args.command == "preprocess":
        records = preprocess_dataset(
            args.input_dir,
            args.output_dir,
            size=args.size,
            assume_cropped=args.assume_cropped,
            on_no_face=args.on_no_face,
            margin=args.margin,
            overwrite=args.overwrite,
        )
        counts = Counter(record.status for record in records)
        return {
            "input_dir": str(args.input_dir),
            "output_dir": str(args.output_dir),
            "total": len(records),
            "status_counts": dict(sorted(counts.items())),
            "manifest": str(args.output_dir / "preprocess_manifest.jsonl"),
        }
    if args.command == "train-embedder":
        return train_embedding_model(
            args.input_dir,
            args.model_out,
            args.metadata_out,
            backbone=args.backbone,
            image_size=args.image_size,
            embedding_dim=args.embedding_dim,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_fraction=args.validation_fraction,
            learning_rate=args.learning_rate,
            seed=args.seed,
            pretrained=args.pretrained,
        )
    if args.command == "embed":
        return export_embeddings(
            args.input_dir,
            args.model,
            args.output,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    if args.command == "train-classifier":
        return train_classifier(
            args.embeddings,
            args.classifier_out,
            c_value=args.c_value,
            seed=args.seed,
        )
    if args.command == "evaluate":
        return evaluate_classifier(
            args.embeddings,
            args.classifier,
            args.report_out,
        )
    if args.command == "demo":
        return run_demo_pipeline(
            args.work_dir,
            epochs=args.epochs,
            seed=args.seed,
            image_size=args.image_size,
            overwrite=args.overwrite,
        )
    raise FacenetStudentError(f"Unknown command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = _dispatch(args)
    except FacenetStudentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _print_json(result)
    return 0
