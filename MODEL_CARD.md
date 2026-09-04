# Model card

## Model details

- **Name:** Student Face-Embedding Pipeline
- **Version:** 0.1.0
- **Type:** TensorFlow/Keras image embedder followed by a scikit-learn linear SVM
- **Embedding size:** 128 by default
- **Backbones:** a small classroom CNN or MobileNetV2 transfer learning
- **License:** MIT for this repository's code; dependency and weight licenses
  remain separate

## Intended use

The model is intended for classroom study of image preprocessing, feature
embeddings, supervised classification, evaluation, and ML project structure.
The included demonstration uses generated non-person imagery.

## Out-of-scope use

Do not use this project for surveillance, covert identification, biometric
authentication, access control, or decisions about employment, education,
housing, insurance, credit, healthcare, policing, or legal status. Do not
create datasets without the informed permission of the people depicted.

## Training data

No real-person training data or trained model is distributed. The user supplies
their own authorized dataset. `demo-data` creates three synthetic identities by
default and keeps train/test images separate.

## Evaluation

The evaluation command reports accuracy, a confusion matrix, per-class
precision/recall/F1, and per-image confidence. Results from the synthetic demo
only validate the pipeline and must not be represented as real-world accuracy.

## Limitations and risks

- The pipeline is closed-set: the SVM chooses among identities known during
  training and does not solve open-set recognition.
- SVM probabilities are calibrated on limited training data and should not be
  treated as certainty.
- Haar-cascade detection and eye alignment can fail with non-frontal faces,
  occlusion, unusual lighting, or low resolution.
- MobileNetV2 ImageNet features are not a substitute for a carefully validated
  face-specific model.
- Small or unbalanced datasets can overfit and hide performance disparities.
- Facial embeddings are biometric data and require careful access, retention,
  consent, and deletion practices.

## Recommended evaluation before any research use

Use a participant-independent holdout set, document consent and provenance,
report class counts, inspect false positives and false negatives, and evaluate
performance across relevant capture conditions and demographic groups. If the
system cannot safely abstain on an unknown person, it is not suitable for an
open-world setting.
