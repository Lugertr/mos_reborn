# postprocessing/text_norm.py
import unicodedata as ud
import re

_spaces_re = re.compile(r"\s+")

def nfc(s: str) -> str:
    return ud.normalize("NFC", s or "")

def clean_spaces(s: str) -> str:
    s = nfc(s)
    s = _spaces_re.sub(" ", s).strip()
    return s
