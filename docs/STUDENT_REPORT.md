# Student report: a modern face-embedding pipeline

## Objective

The goal was to reproduce the learning outcomes of Cole Murray's 2017 facial
recognition tutorial with a current, reproducible Python project. The original
flow detects and aligns a face, converts it to a 128-dimensional FaceNet
embedding, trains a linear SVM, and evaluates predictions on held-out images.

## What I changed and why

The reference implementation targets TensorFlow 1.1 and uses sessions, queue
runners, a frozen `.pb` graph, Dlib 19.4, and OpenCV 3.2. Those versions are no
longer a practical default. I wrote a clean TensorFlow 2/Keras version using a
`tf.data` input pipeline and the current `.keras` model format.

The project provides two backbones:

- `tiny`: a small convolutional network for the offline classroom demo.
- `mobilenet_v2`: a frozen ImageNet feature extractor followed by a trainable
  128-dimensional embedding and identity head.

The original Dlib alignment asset is not redistributed. OpenCV Haar cascades
detect the largest face and eyes. The estimated eye angle rotates the crop
before it is resized. Already aligned datasets can bypass detection with
`--assume-cropped`.

## Method

### 1. Dataset organization

Every immediate child folder is an identity label. Sorting the folder names
makes the integer label mapping deterministic. Train and test images live in
separate directory trees.

### 2. Preprocessing

Images are corrected for EXIF orientation and converted to RGB. In detection
mode, the largest frontal face is selected. A margin is added, two eyes are
used to estimate roll, and the result is center-cropped to a square. Every
attempt is recorded in `preprocess_manifest.jsonl` so skipped inputs are not
silent.

### 3. Embedding model

The training model ends in a softmax identity head. Immediately before that
head, a dense layer maps features into 128 values and a unit-normalization
layer maps them onto a hypersphere. Training minimizes sparse categorical
cross-entropy. This is simpler than FaceNet's triplet loss, but it preserves the
central idea that an image becomes a compact vector that can be reused.

### 4. Linear SVM

The normalized vectors are standardized and passed to a probability-enabled
linear support-vector classifier. The TensorFlow model and SVM are saved
separately so their roles are explicit.

### 5. Evaluation

The test archive is embedded independently. The report includes accuracy,
precision, recall, F1, support, a confusion matrix, and each image's predicted
label and confidence. This is more informative than accuracy alone.

## Reproducibility checks

- Synthetic identities and train/test variations use a fixed random seed.
- Unit tests verify dataset ordering, stratified splitting, preprocessing,
  serialization, classifier evaluation, and command parsing.
- GitHub Actions runs linting and tests without fetching large ML packages.
- The one-command demo exercises TensorFlow training and all file artifacts on
  a machine with the optional TensorFlow extra installed.

## Interpretation

A high result on the synthetic demo only shows that the stages are wired
together and that simple generated patterns are separable. It says nothing
about performance on human faces. Real evaluation would require consented,
representative data; open-set rejection; liveness protections; subgroup
analysis; and a much more suitable face-specific model.

## Credit

The project structure and learning sequence were inspired by Cole Murray's
article, “Building a Facial Recognition Pipeline with Deep Learning in
Tensorflow,” published July 1, 2017, and its companion repository. The code in
this repository was independently written. Full links and provenance appear in
`ATTRIBUTION.md`.
