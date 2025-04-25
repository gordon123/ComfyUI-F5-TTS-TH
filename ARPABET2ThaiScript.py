from g2p_en import G2p

# 1. สร้าง G2p object
g2p = G2p()

# 2. ตาราง ARPABET → ภาษาไทย (simplified)
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

def eng_to_thai_translit(eng_text):
    phonemes = g2p(eng_text)    # e.g. ['M','AE','N','AH','JH','ER']
    th_chars = []
    for p in phonemes:
        t = ARPABET2TH.get(p)
        if t:
            th_chars.append(t)
        elif p == " " or p == "  ":
            th_chars.append(" ")
        # else skip unknown
    return "".join(th_chars)
