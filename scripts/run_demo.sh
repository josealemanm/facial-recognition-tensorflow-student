#!/bin/sh
set -eu

python -m facenet_student demo --work-dir runs/demo "$@"
