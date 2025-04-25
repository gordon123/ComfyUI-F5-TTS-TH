# ARPABET2ThaiScript.py

import re
import nltk

# 1) Monkey-patch nltk so g2p_en won’t try to download a tagger
nltk.pos_tag = lambda tokens: [(t, "") for t in tokens]

from g2p_en import G2p

# 2) Initialize G2p, safely
try:
    g2p = G2p()
except Exception:
    g2p = None

# 3) Full ARPABET→Thai map using your three tables
ARPABET2TH = {
    # — consonants (initial & final) —
    "B":   "บ",  "CH":  "ช",  "D":   "ด",  "DH":  "ด",
    "F":   "ฟ",  "G":   "ก",  "HH":  "ฮ",  "JH":  "จ",
    "K":   "ก",  "L":   "ล",  "M":   "ม",  "N":   "น",
    "NG":  "ง",  "P":   "ป",  "R":   "ร",  "S":   "ส",
    "SH":  "ช",  "T":   "ต",  "TH":  "ท",  "V":   "ว",
    "W":   "ว",  "Y":   "ย",  "Z":   "ซ",  "ZH":  "ช",

    # — vowels (‘แม่กก’, ‘แม่กง’, etc.) —
    "AA":  "อา",  "AE":  "แอ",  "AH":  "อะ",  "AO":  "ออ",
    "AW":  "อาว", "AY":  "อาย", "EH":  "เอะ", "ER":  "เออร์",
    "EY":  "เอย์","IH":  "อิ",  "IY":  "อี",  "OW":  "โอะ",
    "OY":  "ออย","UH":  "อุ",  "UW":  "อู",
}

def eng_to_thai_translit(text: str) -> str:
    """
    Split the text into ASCII-letter runs vs everything else.
    Transliterate each ASCII run via G2P→ARPABET2TH; leave Thai (and punctuation)
    untouched.
    """
    if g2p is None:
        return text

    def _trans(word: str) -> str:
        # get phones, strip digits (stress) off each
        phones = [re.sub(r"\d", "", p) for p in g2p(word)]
        # map each phone to Thai or skip
        return "".join(ARPABET2TH.get(p, "") for p in phones)

    parts = re.split(r"([A-Za-z]+)", text)
    out = []
    for p in parts:
        if re.fullmatch(r"[A-Za-z]+", p):
            out.append(_trans(p))
        else:
            out.append(p)
    return "".join(out)
