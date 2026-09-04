# Attribution and provenance

This repository is an independent educational reimplementation inspired by:

- Cole Murray, “Building a Facial Recognition Pipeline with Deep Learning in
  Tensorflow,” HackerNoon, July 1, 2017.
  <https://hackernoon.com/building-a-facial-recognition-pipeline-with-deep-learning-in-tensorflow-66e7645015b8>
- Cole Murray's companion repository:
  <https://github.com/ColeMurray/medium-facenet-tutorial>
- Florian Schroff, Dmitry Kalenichenko, and James Philbin, “FaceNet: A Unified
  Embedding for Face Recognition and Clustering,” 2015.
  <https://arxiv.org/abs/1503.03832>

Cole Murray deserves credit for the original tutorial's four-stage learning
flow: face preprocessing, neural embeddings, SVM training, and evaluation.

## Source-code provenance

The Python source in this repository was written independently for this
student exercise. It does not copy source files, model weights, or the Dlib
landmark data file from the companion repository. At the time this project was
prepared, that repository did not contain a repository-level license, so its
code is referenced for credit but not redistributed here.

The project uses TensorFlow, Pillow, NumPy, scikit-learn, joblib, and optionally
OpenCV under their respective licenses. MobileNetV2 weights, when requested,
are downloaded by TensorFlow and are not committed to this repository.
