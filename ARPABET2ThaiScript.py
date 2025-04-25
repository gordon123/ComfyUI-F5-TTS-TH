import nltk
# Monkey-patch nltk.pos_tag to bypass missing tagger resource
nltk.pos_tag = lambda tokens: [(t, '') for t in tokens]

from g2p_en import G2p

# Initialize G2p object
try:
    g2p = G2p()
except Exception:
    # Fallback in case g2p fails to initialize
    g2p = None

# Simplified ARPABET → Thai mapping
ARPABET2TH = {
    "AA": "อา", "AE": "แอ", "AH": "อะ", "AO": "ออ",
    "B": "บ",  "CH": "ช",   "D": "ด",  "DH": "ฺดฺ",
    "EH": "เอะ","ER": "เออร์","EY": "เอย์","F": "ฟ",
    "G": "ก",  "HH": "ฮ",   "IH": "อิ","IY": "อี",
    "JH": "จ", "K": "ก",   "L": "ล",  "M": "ม",
    "N": "น",  "NG": "ง",  "OW": "โอะ","OY": "ออย",
    "P": "พ",  "R": "ร",   "S": "ส",  "SH": "ช",
    "T": "ท",  "TH": "ธ",  "UH": "อุ","UW": "อู",
    "V": "ว",  "W": "ว","Y": "ย","Z": "ซ","ZH": "ช"
}

def eng_to_thai_translit(eng_text: str) -> str:
    """
    Convert English text to a Thai transliteration by mapping ARPABET phonemes.
    Falls back to stripping non-mapped phonemes.
    """
    if g2p is None:
        return eng_text
    phonemes = g2p(eng_text)
    th_chars = []
    for p in phonemes:
        if p in ARPABET2TH:
            th_chars.append(ARPABET2TH[p])
        elif p.strip() == "":
            th_chars.append(" ")
        # skip any unknown symbols
    return "".join(th_chars)
