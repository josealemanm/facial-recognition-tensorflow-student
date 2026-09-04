# Architecture and data flow

```text
identity folders
      │
      ▼
preprocess ──► aligned/cropped JPEGs + JSONL manifest
      │
      ▼
train-embedder ──► TensorFlow `.keras` model + metadata JSON
      │
      ▼
embed ──► normalized vectors + integer labels + relative paths (`.npz`)
      │
      ▼
train-classifier ──► scaled linear SVM (`.joblib`)
      │
      ▼
evaluate ──► accuracy + class metrics + confusion matrix + predictions (`.json`)
```

## Package modules

- `data.py` discovers class directories, assigns stable labels, and creates
  stratified train/validation partitions.
- `preprocessing.py` handles EXIF orientation, optional Haar face detection,
  eye-based rotation, center cropping, and manifest output.
- `model.py` builds/trains the Keras model and exports normalized embeddings.
- `classifier.py` trains, serializes, and evaluates the linear SVM.
- `demo.py` creates deterministic synthetic images that do not depict people.
- `pipeline.py` composes all stages for the one-command demo.
- `cli.py` exposes each stage as a command suitable for scripts and CI.

## Main design decisions

### TensorFlow 2 instead of TensorFlow 1

The 2017 tutorial uses sessions, queue runners, and a frozen protobuf graph.
This implementation uses current `tf.data` and Keras model saving. Each stage
still produces an explicit artifact, which makes the data flow inspectable.

### Independent preprocessing

The original project relies on a Dlib landmark file. This project avoids
redistributing that file and uses OpenCV's bundled Haar cascades. Detected eyes
provide a simple rotation estimate; center cropping is used as a fallback after
a face is found. This is easier to install but less accurate than modern
landmark models.

### Supervised embedding head

The network learns an embedding through an identity-classification objective.
The named `embedding` layer is L2-normalized and becomes the input to a separate
SVM. This retains the tutorial's separation between deep features and the
downstream classifier, while keeping the classroom demo small.

### Separate train and test directories

The model never performs a random test split during evaluation. Users create a
held-out test directory before training, which makes accidental evaluation on
training images less likely. `train-embedder` creates a separate stratified
validation subset only for model-selection feedback.

## Artifact schemas

Embedding archives contain:

- `embeddings`: float32 matrix shaped `(n_samples, embedding_dim)`
- `labels`: int64 vector
- `class_names`: Unicode vector whose index defines the integer label
- `paths`: Unicode paths relative to the input directory
- `schema_version`: integer scalar

Classifier artifacts contain the fitted scaler/SVC pipeline, class names,
embedding dimension, and schema version. They are serialized with joblib and
must be treated as trusted-code artifacts.
