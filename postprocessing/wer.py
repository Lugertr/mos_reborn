# postprocessing/wer.py
"""
Purpose
-------
Подсчёт WER (Word Error Rate) для оценивания качества распознавания.

Реализовано два сценария:
- Точный WER (`wer_exact`): классический Levenshtein по словам (вставки/удаления/замены).
- Прокси-WER:
    * `wer_proxy_from_conf` — простая эвристика из conf Tesseract (1 - conf/100).
    * `wer_proxy_between` — «несогласие» между гипотезами Tesseract и TrOCR как приближение WER.

Notes
-----
- Базовая функция `_levenshtein` работает на уровне токенов (список слов), а не символов.
- Сложность O(n*m) по времени и памяти — для строковой длины в наших задачах приемлемо.
"""

from typing import List

def _levenshtein(a: List[str], b: List[str]) -> int:
    """
    Классическое расстояние Левенштейна между последовательностями токенов.

    Args:
        a: Список токенов-слов «эталона».
        b: Список токенов-слов «гипотезы».

    Returns:
        Минимальное количество правок (вставка/удаление/замена), чтобы превратить `a` в `b`.

    Implementation details:
        Используем ДП-матрицу размера (len(a)+1) x (len(b)+1):
        - Первая строка/столбец — стоимость превращения пустой строки в префикс другой (чистые вставки/удаления).
        - Переход: min(удаление, вставка, замена/совпадение).
    """
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    # dp[i][j] — стоимость превращения первых i токенов a в первые j токенов b
    dp = [list(range(m + 1))] + [[i] + [0]*m for i in range(1, n + 1)]
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            dp[i][j] = min(dp[i-1][j]+1,        # удаление
                           dp[i][j-1]+1,        # вставка
                           dp[i-1][j-1]+cost)  # замена/совпадение
    return dp[n][m]

def wer_exact(ref: str, hyp: str) -> float:
    """
    Точный WER по словам: Levenshtein(ref_words, hyp_words) / max(1, len(ref_words)).

    Args:
        ref: Эталонный текст (строка).
        hyp: Гипотеза распознавания (строка).

    Returns:
        Значение WER в диапазоне [0..1]. Если `ref` пустой, то WER=1.0 при непустом `hyp`, иначе 0.0.
    """
    r = ref.split()
    h = hyp.split()
    if not r:
        return float(len(h) > 0)
    return _levenshtein(r, h) / max(1, len(r))

def wer_proxy_from_conf(conf: float) -> float:
    """
    Прокси-оценка WER из средней уверенности Tesseract.

    Args:
        conf: Средняя уверенность в процентах [0..100].

    Returns:
        Эвристика 1 - conf/100, ограниченная в [0..1].
    """
    # Простая эвристика: 1 - conf/100
    conf = max(0.0, min(100.0, float(conf)))
    return max(0.0, min(1.0, 1.0 - conf / 100.0))

def wer_proxy_between(tess_text: str, trocr_text: str) -> float:
    """
    Прокси-ошибка по «несогласию» двух гипотез (Tesseract vs TrOCR).

    Args:
        tess_text: Текст от Tesseract.
        trocr_text: Текст от TrOCR.

    Returns:
        WER между двумя гипотезами как приближённая оценка ошибки.
    """
    # «Несогласие» двух гипотез как прокси-ошибка
    return wer_exact(tess_text, trocr_text)
