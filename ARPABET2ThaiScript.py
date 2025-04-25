# ARPABET2ThaiScript.py

import re
import nltk

# 1) Monkey-patch nltk so g2p_en won’t try to download a tagger
nltk.pos_tag = lambda tokens: [(t, "") for t in tokens]

from g2p_en import G2p

# 2) Initialize G2p, but safely
try:
    g2p = G2p()
except Exception:
    g2p = None

# 3) Simplified ARPABET→Thai map (we strip digits off phones first)
ARPABET2TH = {
    "AA": "อา", "AE": "แอ", "AH": "อะ", "AO": "ออ",
    "B": "บ",   "CH": "ช",   "D": "ด",   "DH": "ฺดฺ",
    "EH": "เอะ", "ER": "เออร์", "EY": "เอย์", "F": "ฟ",
    "G": "ก",   "HH": "ฮ",   "IH": "อิ",  "IY": "อี",
    "JH": "จ",  "K": "ก",   "L": "ล",   "M": "ม",
    "N": "น",   "NG": "ง",  "OW": "โอะ", "OY": "ออย",
    "P": "พ",   "R": "ร",   "S": "ส",   "SH": "ช",
    "T": "ท",   "TH": "ธ",  "UH": "อุ",  "UW": "อู",
    "V": "ว",   "W": "ว",   "Y": "ย",   "Z": "ซ", "ZH": "ช"
}

def eng_to_thai_translit(text: str) -> str:
    """
    Split the text into runs of ASCII letters vs other.
    Transliterate each ASCII word via G2P → ARPABET2TH,
    leave all non-ASCII (i.e. Thai) untouched.
    """
    if g2p is None:
        return text

    def _trans(word: str) -> str:
        phones = g2p(word)
        # strip digits (stress markers) from phones
        phones = [re.sub(r"\d", "", p) for p in phones]
        # map each phone to Thai char (skip unknown)
        return "".join(ARPABET2TH.get(p, "") for p in phones)

    parts = re.split(r"([A-Za-z]+)", text)
    result = []
    for part in parts:
        if re.fullmatch(r"[A-Za-z]+", part):
            result.append(_trans(part))
        else:
            result.append(part)
    return "".join(result)
