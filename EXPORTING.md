# Exporting to GitHub

The repository intentionally excludes datasets, embeddings, trained models,
reports, and local environments. Review `git status` before publishing to make
sure no personal images or generated artifacts are staged.

## Publish with GitHub CLI

Replace the example repository name if desired:

```bash
git init -b main
git add .
git status
git commit -m "Initial student implementation"
gh repo create facial-recognition-tensorflow-student \
  --source=. \
  --private \
  --push
```

Starting with a private repository is recommended whenever work involves human
face images. The GitHub CLI will ask you to authenticate if needed.

## Export as an archive

From the parent directory:

```bash
zip -r facial-recognition-tensorflow-student.zip \
  facial-recognition-tensorflow-student \
  -x "*/.venv/*" "*/runs/*" "*/build/*" "*/dist/*" "*/.git/*" \
     "*/.pytest_cache/*" "*/.ruff_cache/*" "*/__pycache__/*" \
     "*/.coverage" "*/*.egg-info/*"
```
