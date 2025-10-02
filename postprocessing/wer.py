# postprocessing/wer.py
from typing import List

def _levenshtein(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = [list(range(m + 1))] + [[i] + [0]*m for i in range(1, n + 1)]
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[n][m]

def wer_exact(ref: str, hyp: str) -> float:
    r = ref.split()
    h = hyp.split()
    if not r:
        return float(len(h) > 0)
    return _levenshtein(r, h) / max(1, len(r))

def wer_proxy_from_conf(conf: float) -> float:
    # Простая эвристика: 1 - conf/100
    conf = max(0.0, min(100.0, float(conf)))
    return max(0.0, min(1.0, 1.0 - conf / 100.0))

def wer_proxy_between(tess_text: str, trocr_text: str) -> float:
    # «Несогласие» двух гипотез как прокси-ошибка
    return wer_exact(tess_text, trocr_text)
