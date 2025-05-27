# -*- coding: utf-8 -*-
# This script is part of a Thai TTS system that converts English text to Thai script using ARPAbet phonemes.
# รายการนี้เป็นส่วนหนึ่งของระบบ TTS ภาษาไทยที่แปลงข้อความภาษาอังกฤษเป็นอักษรไทยโดยใช้ ARPAbet phonemes
# แต่ยังไม่เสร็จสมบูรณ์ ยังอ่านภษาอังกฤษไม่ออก เต็ม 100%

import os
import re
import csv
import nltk

# 1) Monkey-patch NLTK so g2p_en won’t try to fetch taggers
nltk.pos_tag = lambda tokens: [(t, "") for t in tokens]

from g2p_en import G2p

# 2) Initialize G2p, safely
try:
    g2p = G2p()
except Exception:
    g2p = None

# 3) Load exceptions mapping from CSV
def load_exceptions(csv_filename: str) -> dict:
    exc = {}
    base = os.path.dirname(__file__)
    path = os.path.join(base, csv_filename)
    if not os.path.exists(path):
        return exc
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            word = row[0].strip().lower()
            translit = row[1].strip()
            exc[word] = translit
    return exc

EXCEPTIONS = load_exceptions("exceptions.csv")

# 4) Simplified ARPABET → Thai map
ARPABET2TH = {
    # consonants
    "B": "บ",  "CH": "ช", "D": "ด",  "DH": "ด",
    "F": "ฟ",  "G": "ก",  "HH": "ฮ", "JH": "จ",
    "K": "ก",  "L": "ล",  "M": "ม",  "N": "น",
    "NG": "ง", "P": "ป",  "R": "ร",  "S": "ส",
    "SH": "ช", "T": "ต",  "TH": "ท", "V": "ว",
    "W": "ว",  "Y": "ย",  "Z": "ซ", "ZH": "ช",

    # vowels
    "AA": "อา", "AE": "แอ", "AH": "อะ", "AO": "ออ",
    "AW": "อาว","AY": "อาย","EH": "เอะ","ER": "เออร์",
    "EY": "เอย์","IH": "อิ","IY": "อี","OW": "โอะ",
    "OY": "ออย","UH": "อุ","UW": "อู",
}

def eng_to_thai_translit(text: str) -> str:
    """
    Split into ASCII-only runs vs Thai/other.
    If the run is pure ASCII letters and matches EXCEPTIONS → use that.
    Otherwise G2P→ARPABET→Thai.
    """
    if g2p is None:
        return text

    def _trans(word: str) -> str:
        # get phones and strip stress digits
        phones = [re.sub(r"\d", "", p) for p in g2p(word)]
        # map each ARPABET phone to Thai char
        return "".join(ARPABET2TH.get(p, "") for p in phones)

    parts = re.split(r"([A-Za-z]+)", text)
    out = []
    for p in parts:
        if re.fullmatch(r"[A-Za-z]+", p):
            low = p.lower()
            if low in EXCEPTIONS:
                out.append(EXCEPTIONS[low])
            else:
                out.append(_trans(p))
        else:
            out.append(p)
    return "".join(out)
