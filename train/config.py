# config.py
"""
Конфигурация проекта обучения TrOCR с использованием собственного токенизатора
(созданного через build_tokenizer.py).
"""

# Путь к директории, где сохранён свой processor (feature_extractor + tokenizer),
# созданный с помощью build_tokenizer.py
MODEL_NAME_OR_PATH = "./trocr-processor-old"

# Директория с данными
DATA_DIR = "data/handwritten"
IMAGES_SUBDIR = "images"

# Директории сохранения
CHECKPOINTS_DIR = "./checkpoints"
OUTPUT_DIR = "./trocr-finetuned"

# Тренировочные параметры
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1

# Количество примеров для отладки, если нужно
DEBUG_TRAIN_SAMPLES = 0   # 0 — означает использовать всё
DEBUG_EVAL_SAMPLES = 0

# Генерация / декодировка
GENERATION_MAX_LENGTH = 128

# Логирование
LOGGING_STEPS = 20
LOG_DIR = "./logs"
TB_EXAMPLES_TO_LOG = 5

# Случайное зерно для воспроизводимости
SEED = 42

# Скорость обучения
LEARNING_RATE = 5e-5
