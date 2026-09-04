# Ethics and responsible data handling

Faces and facial embeddings can identify people and should be treated as
sensitive biometric data. A technically working classroom pipeline does not
make a deployment appropriate.

## Before collecting images

1. Define a narrow learning or research purpose.
2. Obtain informed, revocable permission from each person depicted.
3. Explain what will be trained, who can access it, how long it will be kept,
   and how deletion requests will be honored.
4. Collect only what is needed and avoid scraping images from the web.
5. Check applicable institutional, contractual, and legal requirements.

## During the project

- Keep raw images, processed images, embeddings, and trained models outside
  version control.
- Encrypt storage where possible and limit access to project participants.
- Use participant-independent test images and retain dataset provenance.
- Inspect error rates across relevant lighting, pose, camera, and demographic
  groups. Overall accuracy can conceal serious disparities.
- Do not treat SVM probabilities as verified identity or certainty.

## After the project

Delete images, embeddings, backups, and model artifacts according to the
consented retention plan. A face embedding is not anonymous merely because it
cannot be viewed like a photograph.

## Prohibited project uses

This repository is not designed or validated for surveillance, authentication,
access control, policing, border control, or consequential decisions involving
employment, education, housing, insurance, credit, or healthcare.
