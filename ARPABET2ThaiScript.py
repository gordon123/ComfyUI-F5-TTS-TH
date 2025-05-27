# -*- coding: utf-8 -*-
# This script is part of a Thai TTS system that converts English text to Thai script using ARPAbet phonemes.
# รายการนี้เป็นส่วนหนึ่งของระบบ TTS ภาษาไทยที่แปลงข้อความภาษาอังกฤษเป็นอักษรไทยโดยใช้ ARPAbet phonemes
# ปรับปรุงให้ตัด ‘อ’ หน้า vowels เมื่ออยู่หลัง consonants เพื่อให้เสียงเป็นธรรมชาติยิ่งขึ้น

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

    # vowels (เก็บค่าเป็นการสะกดเต็มรูป)
    "AA": "อา", "AE": "แอ", "AH": "อะ", "AO": "ออ",
    "AW": "อาว","AY": "อาย","EH": "เอะ","ER": "เออร์",
    "EY": "เอย์","IH": "อิ","IY": "อี","OW": "โอะ",
    "OY": "ออย","UH": "อุ","UW": "อู",
}

# กำหนดรายการ ARPABET vowels เพื่อตรวจบริบท
VOWELS = {"AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW"}


def eng_to_thai_translit(text: str) -> str:
    """
    Split into ASCII-only runs vs Thai/other.
    If the run is pure ASCII letters and matches EXCEPTIONS → use that.
    Otherwise G2P→ARPABET→Thai.
    ปรับให้ตัด leading 'อ' ของ vowels เมื่อคำอยู่หลัง consonant.
    """
    if g2p is None:
        return text

    def _trans(word: str) -> str:
        # get phones and strip stress digits
        phones = [re.sub(r"\d", "", p) for p in g2p(word)]
        translits = []
        prev_was_consonant = False
        for p in phones:
            # แปลง ARPABET เป็น Thai
            t = ARPABET2TH.get(p, "")
            if not t:
                prev_was_consonant = False
                continue
            # ถ้าเป็น vowel และก่อนหน้าเป็น consonant และ translit ขึ้นต้นด้วย 'อ'
            if prev_was_consonant and p in VOWELS and t.startswith("อ"):
                t = t[1:]
            translits.append(t)
            # อัปเดตสถานะ prev_was_consonant
            prev_was_consonant = (p not in VOWELS)
        return "".join(translits)

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
