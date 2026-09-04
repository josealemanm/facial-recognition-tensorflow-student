# TensorFlow Face-Embedding Pipeline — Student Edition

An independently written, modern student implementation of the learning
pipeline described in Cole Murray's 2017 tutorial: prepare identity-labelled
images, learn 128-dimensional embeddings with TensorFlow/Keras, train a linear
SVM, and evaluate it on held-out images.

> **Original-project credit:** This exercise was inspired by Cole Murray's
> [HackerNoon tutorial](https://hackernoon.com/building-a-facial-recognition-pipeline-with-deep-learning-in-tensorflow-66e7645015b8)
> and [companion repository](https://github.com/ColeMurray/medium-facenet-tutorial).
> See [ATTRIBUTION.md](ATTRIBUTION.md) for full provenance. No upstream source
> code or model files are included.

## What this project demonstrates

1. Optional face detection, eye-based rotation, cropping, and resizing with
   OpenCV.
2. A TensorFlow 2/Keras embedding model with either a small classroom CNN or a
   frozen MobileNetV2 transfer-learning backbone.
3. Export of normalized embedding vectors in a portable `.npz` archive.
4. A probability-enabled linear SVM trained with scikit-learn.
5. Evaluation with accuracy, per-class metrics, a confusion matrix, and
   per-image predictions saved as JSON.

The synthetic demo contains generated cartoon-like patterns only. It proves
that the software stages connect correctly; it is **not** a facial-recognition
accuracy benchmark.

## Quick start

Use Python 3.10–3.12. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[tensorflow,dev]"
pytest
python -m facenet_student demo --work-dir runs/demo --epochs 8
```

The demo writes its final report to `runs/demo/artifacts/evaluation.json`.
OpenCV is not needed for the generated demo because those inputs are already
cropped. Install the complete real-image environment with:

```bash
python -m pip install -e ".[tensorflow,vision,dev]"
```

The committed `uv.lock` provides a fully resolved alternative:

```bash
uv sync --extra tensorflow --extra vision --extra dev
uv run facenet-student demo --work-dir runs/demo --epochs 8
```

## Run the stages on consented data

Use only images you are authorized to process. Keep train and test subjects in
the same label set but use different images:

```text
data/raw/
├── train/
│   ├── identity_a/
│   │   ├── image_001.jpg
│   │   └── image_002.jpg
│   └── identity_b/
└── test/
    ├── identity_a/
    └── identity_b/
```

Preprocess both splits:

```bash
python -m facenet_student preprocess \
  --input-dir data/raw/train \
  --output-dir data/processed/train

python -m facenet_student preprocess \
  --input-dir data/raw/test \
  --output-dir data/processed/test
```

Train a 128-dimensional embedder. The default MobileNetV2 backbone downloads
ImageNet weights on first use:

```bash
python -m facenet_student train-embedder \
  --input-dir data/processed/train \
  --model-out artifacts/embedder.keras \
  --metadata-out artifacts/embedder.json \
  --backbone mobilenet_v2 \
  --epochs 20
```

Generate vectors, train the SVM, and evaluate:

```bash
python -m facenet_student embed \
  --input-dir data/processed/train \
  --model artifacts/embedder.keras \
  --output artifacts/train_embeddings.npz

python -m facenet_student train-classifier \
  --embeddings artifacts/train_embeddings.npz \
  --classifier-out artifacts/classifier.joblib

python -m facenet_student embed \
  --input-dir data/processed/test \
  --model artifacts/embedder.keras \
  --output artifacts/test_embeddings.npz

python -m facenet_student evaluate \
  --embeddings artifacts/test_embeddings.npz \
  --classifier artifacts/classifier.joblib \
  --report-out artifacts/evaluation.json
```

If every input is already a tightly cropped, upright face, add
`--assume-cropped` to `preprocess`. For a quick classroom model that does not
download ImageNet weights, select `--backbone tiny`.

## Useful commands

```bash
python -m facenet_student --help
python -m facenet_student scan --input-dir data/raw/train
python -m facenet_student demo-data --output-dir runs/generated-data
make test
make demo
```

## Reproducibility and GitHub hygiene

- Seeds are recorded and applied to Python, NumPy, and TensorFlow.
- Dataset discovery and label assignment are deterministic.
- `uv.lock` records an exact cross-platform dependency resolution.
- Raw biometric data, processed images, model weights, and generated artifacts
  are ignored by Git.
- CI runs the lightweight unit suite without downloading TensorFlow or model
  weights.
- The Dockerfile provides a CPU-focused Linux environment for the full demo.
- Generated `joblib` files are Python pickles internally. Load only artifacts
  you created or obtained from a trusted source.

## Important limitations

This is a small educational closed-set classifier, not a production biometric
system. It does not establish identity, liveness, fairness, robustness, or
security. Do not use it for surveillance, authentication, law enforcement,
employment, housing, credit, healthcare, or other consequential decisions.
Performance can vary sharply across lighting, pose, cameras, and demographic
groups. Read [MODEL_CARD.md](MODEL_CARD.md) and [docs/ETHICS.md](docs/ETHICS.md)
before using real images.

## Documentation

- [Student report](docs/STUDENT_REPORT.md)
- [Architecture and data flow](docs/ARCHITECTURE.md)
- [Ethics and data handling](docs/ETHICS.md)
- [Verification record](docs/VERIFICATION.md)
- [Model card](MODEL_CARD.md)
- [Attribution](ATTRIBUTION.md)
- [GitHub export instructions](EXPORTING.md)

## License

The independently written code in this repository is available under the MIT
License. External packages, pretrained weights, datasets, and the credited
upstream tutorial retain their own terms.
