#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script is used to run local test on CITYSCAPE VOC 2012. Users could also
# modify from this script for their use case.
#
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   sh ./local_test.sh
#
#

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# restrict CUDA GPU
export CUDA_VISIBLE_DEVICES=$1

echo GPU $1
echo Batch size $2

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"

# Set up the working directories.
CITYSCAPE_FOLDER="cityscapes"
EXP_FOLDER="exp/train_on_trainval_set"
DATASET_DIR=datasets
# INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${CITYSCAPE_FOLDER}/${EXP_FOLDER}/train/model.ckpt-13447"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${CITYSCAPE_FOLDER}/${EXP_FOLDER}/train_vis/model.ckpt-9699"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${CITYSCAPE_FOLDER}/${EXP_FOLDER}/modify"
mkdir -p "${TRAIN_LOGDIR}"

CITYSCAPE_DATASET="${WORK_DIR}/${DATASET_DIR}/${CITYSCAPE_FOLDER}/tfrecord"

NUM_ITERATIONS=30000
ipython -i "${WORK_DIR}"/modify.py --\
  --train_split="train" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --train_crop_size=769 \
  --train_crop_size=769 \
  --train_batch_size=$2 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=true \
  --tf_initial_checkpoint="${INIT_FOLDER}" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset=${CITYSCAPE_FOLDER} \
  --dataset_dir="${CITYSCAPE_DATASET}"