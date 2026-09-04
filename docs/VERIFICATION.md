# Verification record

This repository was validated on September 4, 2026, on Apple Silicon with
Python 3.10.20.

## Checks completed

- Ruff static checks passed.
- All 11 tests passed with TensorFlow installed.
- A wheel built successfully from the repository.
- The OpenCV preprocessing path loaded its bundled face and eye cascades and
  processed all 18 smoke-test images using the documented no-face fallback.
- The full synthetic pipeline ran on TensorFlow 2.21.0.

## End-to-end demo result

- Generated images: 72 total, depicting no real people
- Identities: 3 synthetic classes
- Training images exported to the SVM: 54
- Held-out test images: 18
- Embedding dimensions: 128
- Embedding norms: approximately 1.0
- SVM training accuracy: 1.0
- Held-out synthetic accuracy: 1.0

These values confirm that preprocessing, Keras training, model serialization,
embedding export, SVM training, and evaluation work together. The synthetic
accuracy is deliberately easy and is not evidence of performance on human
faces.
