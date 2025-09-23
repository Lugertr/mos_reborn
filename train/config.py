# config.py
"""
Конфигурация для пайплайна обучения TrOCR.
Отредактируй пути под свою структуру при необходимости.
"""

# --- Tokenizer / Processor ---
# Куда сохранить/загрузить созданный processor (tokenizer + feature_extractor)
TOKENIZER_OUT_DIR = "./trocr-processor"

# Если хочешь тренировать BPE на корпусе — включи True и укажи пути в CORPUS_PATHS
TRAIN_TOKENIZER_FROM_CORPUS = False
CORPUS_PATHS = []  # пример: ["../corpus/old1.txt", "../corpus/old2.txt"]

# Размер словаря при обучении BPE (если TRAIN_TOKENIZER_FROM_CORPUS=True)
TOKENIZER_VOCAB_SIZE = 8000

# Размер картинок для feature extractor (height, width)
TOKENIZER_IMAGE_SIZE = (384, 384)

# --- Данные ---
# DATA_DIR — папка, где лежат train.json и test.json (jsonl или массив/поле "data")
DATA_DIR = "./data/handwritten"
# Поддиректория с изображениями (от DATA_DIR)
IMAGES_SUBDIR = "images"

# --- Сохранение / чекпоинты / логирование ---
CHECKPOINTS_DIR = "./checkpoints"
OUTPUT_DIR = "./trocr-finetuned"
LOG_DIR = "./logs"

# --- Тренировочные параметры ---
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
DEBUG_TRAIN_SAMPLES = 0   # >0 — использовать маленький поднабор для отладки
DEBUG_EVAL_SAMPLES = 0

GENERATION_MAX_LENGTH = 128
LOGGING_STEPS = 20
TB_EXAMPLES_TO_LOG = 5

LEARNING_RATE = 5e-5
SEED = 42

# --- Примечания ---
# Формат JSON-файлов: предпочтительно JSONL (каждая строка — отдельный JSON объект):
# {"file_name": "img1.png", "text": "Текст"}
# {"file_name": "img2.png", "text": "Еще"}
#
# Если у вас JSON вида {"data": [ {...}, {...} ]}, загрузка автоматически попробует поле "data".
#
# Проверяйте, что images/ содержит файлы с именами, совпадающими с file_name в json.
