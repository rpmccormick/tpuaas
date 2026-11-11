#!/usr/bin/env bash
set -euo pipefail

# Download official Google Coral EdgeTPU models, label maps, and sample images into ./models
MODELS_DIR="${1:-./models}"
mkdir -p "$MODELS_DIR"

BASE_URL="https://raw.githubusercontent.com/google-coral/test_data/master"

echo "==> Downloading Coral EdgeTPU models and label maps into ${MODELS_DIR}..."

declare -a FILES=(
  "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
  "ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite"
  "efficientdet_lite0_320_ptq_edgetpu.tflite"
  "ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite"
  "movenet_single_pose_lightning_ptq_edgetpu.tflite"
  "deeplabv3_mnv2_pascal_quant_edgetpu.tflite"
  "mobilenet_v2_1.0_224_quant_edgetpu.tflite"
  "mobilenet_v1_1.0_224_l2norm_quant_edgetpu.tflite"
  "coco_labels.txt"
  "imagenet_labels.txt"
  "parrot.jpg"
  "grace_hopper.bmp"
  "squat.bmp"
)

for f in "${FILES[@]}"; do
  if [ ! -f "${MODELS_DIR}/${f}" ]; then
    echo "  -> Fetching ${f}"
    curl -fsSL "${BASE_URL}/${f}" -o "${MODELS_DIR}/${f}"
  fi
done

cp -f "${MODELS_DIR}/parrot.jpg" "${MODELS_DIR}/sample_parrot.jpg"
cp -f "${MODELS_DIR}/grace_hopper.bmp" "${MODELS_DIR}/sample_face.bmp"
cp -f "${MODELS_DIR}/squat.bmp" "${MODELS_DIR}/sample_pose.bmp"

echo "==> EdgeTPU model pack ready in ${MODELS_DIR}."
