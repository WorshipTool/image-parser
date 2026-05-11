---
name: ml-model-deployment
description: Use when working with ML model files (YOLO8, U-Net), updating trained models, or troubleshooting missing model errors in Docker builds or CI/CD pipeline
---

# ML Model Deployment

## Overview

Model weights are **not stored in git** (`.gitignore` excludes `*.pt` and `*.pth`). They live in **GitHub Releases** and are downloaded automatically during every Docker CI build.

## Production Model Files

| File | Size | Purpose | Location in repo |
|---|---|---|---|
| `yolo8best.pt` | 50 MB | YOLO8 — detects sheet/title/data regions | repo root |
| `paper_segmentation_unet.pth` | 153 MB | U-Net — detects paper edges | `parser/paper_detection/model/checkpoints/` |

Current release tag: **`v1-models`** (defined in `.github/workflows/production-deployment.yml` as `MODELS_RELEASE_TAG`)

## How CI Gets the Models

The workflow downloads both files before `docker build`:

```yaml
- name: Download model weights
  run: |
    gh release download ${{ env.MODELS_RELEASE_TAG }} \
      --pattern "yolo8best.pt" --dir .
    mkdir -p parser/paper_detection/model/checkpoints
    gh release download ${{ env.MODELS_RELEASE_TAG }} \
      --pattern "paper_segmentation_unet.pth" \
      --dir parser/paper_detection/model/checkpoints
  env:
    GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

## Updating Models (When You Train a New Version)

**Step 1** — Create a new release with the updated model files:

```bash
# Run from the image-parser root directory
gh release create v2-models \
  yolo8best.pt \
  parser/paper_detection/model/checkpoints/paper_segmentation_unet.pth \
  --title "ML Models v2" \
  --notes "What changed in this model version" \
  --repo WorshipTool/image-parser
```

**Step 2** — Update the tag in the workflow:

```yaml
# .github/workflows/production-deployment.yml
env:
    DOCKER_IMAGE_NAME: wt-image-parser
    MODELS_RELEASE_TAG: v2-models   # ← change this
```

Commit and push — the next CI build picks up the new models automatically.

## Local Development

Models are **not** downloaded automatically locally. You need them manually:

```bash
# YOLO model — download from existing Dropbox URL
python -m parser.sheet_detection.prepare

# U-Net model — copy from your training output or ask a teammate
# Expected path: parser/paper_detection/model/checkpoints/paper_segmentation_unet.pth
```

## Common Mistakes

**"Docker image works but parser fails at runtime"** — model files missing from the image. Check that the `Download model weights` step ran successfully in the CI log.

**"gh release download: release not found"** — `MODELS_RELEASE_TAG` in the workflow doesn't match an existing release. Check: `gh release list --repo WorshipTool/image-parser`

**"Uploading only one model"** — both files must be in the same release. CI downloads both from the same tag.
