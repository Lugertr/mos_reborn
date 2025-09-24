# coding: utf-8
"""
utils/metrics.py

Вычисление WER и нормализация текста.
"""
import unicodedata
import string
import re
from typing import List, Tuple, Dict, Optional

logger = __import__("logging").getLogger(__name__)


def simple_wer(ref: str, hyp: str) -> float:
    if ref is None:
        ref = ""
    if hyp is None:
        hyp = ""
    r = ref.split()
    h = hyp.split()
    n = len(r)
    m = len(h)
    if n == 0:
        return 0.0 if m == 0 else 1.0
    prev = list(range(m + 1))
    cur = [0] * (m + 1)
    for i in range(1, n + 1):
        cur[0] = i
        for j in range(1, m + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    distance = prev[m]
    return float(distance) / float(n)


def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = s.lower().strip()
    s = "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")
    rus_punct = "«»—–…„“‚‘’"
    punct = string.punctuation + rus_punct
    tr = str.maketrans({c: "" for c in punct})
    s = s.translate(tr)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def postprocess_text(pred_ids: List[List[int]],
                     label_ids: List[List[int]],
                     processor,
                     normalize: bool = True) -> Tuple[List[str], List[str]]:
    pred_texts = processor.batch_decode(pred_ids, skip_special_tokens=True)
    cleaned_labels = [[l for l in seq if l != -100] for seq in label_ids]
    label_texts = processor.batch_decode(cleaned_labels, skip_special_tokens=True)
    if normalize:
        pred_texts = [normalize_text(s) for s in pred_texts]
        label_texts = [normalize_text(s) for s in label_texts]
    return pred_texts, label_texts


def compute_metrics_from_processor(eval_pred: Tuple[List[List[int]], List[List[int]]],
                                   processor) -> Dict[str, float]:
    pred_ids, label_ids = eval_pred
    pred_texts, label_texts = postprocess_text(pred_ids, label_ids, processor, normalize=True)
    wers = [simple_wer(r, h) for r, h in zip(label_texts, pred_texts)]
    avg = sum(wers) / len(wers) if wers else 0.0
    return {"wer": avg}
