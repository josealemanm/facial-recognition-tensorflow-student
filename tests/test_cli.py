from facenet_student.cli import build_parser


def test_parser_accepts_scan_command() -> None:
    args = build_parser().parse_args(["scan", "--input-dir", "data/raw/train"])

    assert args.command == "scan"
    assert str(args.input_dir) == "data/raw/train"


def test_parser_configures_tiny_embedder() -> None:
    args = build_parser().parse_args(
        [
            "train-embedder",
            "--input-dir",
            "processed",
            "--model-out",
            "embedder.keras",
            "--metadata-out",
            "embedder.json",
            "--backbone",
            "tiny",
        ]
    )

    assert args.backbone == "tiny"
    assert args.embedding_dim == 128
