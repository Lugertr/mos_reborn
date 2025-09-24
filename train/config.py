# coding: utf-8
"""
config.py — центральная конфигурация проекта TrOCR.
Редактируйте параметры по необходимости.
"""

import os
import tempfile
from typing import List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Папка с данными (изображения + json)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "handwritten")
TRAIN_IMAGES_SUBDIR = "train"
TEST_IMAGES_SUBDIR = "test"
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, TRAIN_IMAGES_SUBDIR)
TEST_IMAGES_DIR = os.path.join(DATA_DIR, TEST_IMAGES_SUBDIR)

TRAIN_JSON_FILENAME = "train.json"
TEST_JSON_FILENAME = "test.json"
TRAIN_JSON_PATH = os.path.join(DATA_DIR, TRAIN_JSON_FILENAME)
TEST_JSON_PATH = os.path.join(DATA_DIR, TEST_JSON_FILENAME)

# Временные файлы
TMP_DIR = tempfile.gettempdir()
TMP_CORPUS_FILENAME = "trocr_corpus.txt"
TMP_CORPUS_PATH = os.path.join(TMP_DIR, TMP_CORPUS_FILENAME)

IMAGE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"]

# Tokenizer / processor
TOKENIZER_OUT_DIR = os.path.join(PROJECT_ROOT, "trocr-processor")
TRAIN_TOKENIZER_FROM_TEST_SPLIT = False
TOKENIZER_VOCAB_SIZE = 8000
TOKENIZER_SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
TOKENIZER_IMAGE_SIZE: Tuple[int, int] = (384, 384)

# Training / checkpoints / logging
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "train", "checkpoints")
CHECKPOINT_PREFIX = "checkpoint-"
BEST_CHECKPOINT_NAME = "best"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "trocr-finetuned")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# Hyperparams
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1

DEBUG_TRAIN_SAMPLES = 0
DEBUG_EVAL_SAMPLES = 0

GENERATION_MAX_LENGTH = 128
LOGGING_STEPS = 20
TB_EXAMPLES_TO_LOG = 5

LEARNING_RATE = 5e-5
SEED = 42

# Model/runtime tweaks
GRAD_CLIP_NORM = 1.0
ENABLE_TF_GPU_MEMORY_GROWTH = True

# Mixed precision (optional)
ENABLE_MIXED_PRECISION = False

# Gradient accumulation: сколько шагов интегрировать перед apply_gradients
GRADIENT_ACCUMULATION_STEPS = 1

# Checkpointing behavior
SAVE_CHECKPOINT_EVERY_STEPS = 0  # 0 = отключено
CHECKPOINTS_KEEP_LAST = 5        # retention policy: сколько последних хранить
# Async accum vars options
SAVE_ACCUM_VARS = True                      # сохранять ли accum_vars (npz)
SAVE_ACCUM_VARS_ONLY_ON_EPOCH_END = False   # сохранять accum_vars только на конце эпохи/при сигнале
CHECKPOINTS_THREADPOOL_MAX_WORKERS = 2      # пул потоков для фоновых сохранений

# TFRecord pipeline (ускорение предобработки)
USE_TFRECORDS = False         # при True — создадим/прочитаем TFRecord с уже готовыми pixel_values
TFRECORDS_DIR = os.path.join(PROJECT_ROOT, "train", "tfrecords")

# Inference
DEFAULT_INFER_BATCH_SIZE = 4
DEFAULT_INFER_MAX_LENGTH = 128
DEFAULT_MODEL_DIR = OUTPUT_DIR
