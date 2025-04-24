# __init__.py
# รวม node classes ทั้งหมดไว้ใน package นี้

from .F5TTS_Advance import F5TTS_Advance
from .F5TTS import F5TTSAudioInputs
from .save_audio_text_node import SaveAudioAndText

NODE_CLASS_MAPPINGS = {
    "F5TTS_Advance":     F5TTS_Advance,
    "F5TTSAudioInputs":  F5TTSAudioInputs,
    "SaveAudioAndText":   SaveAudioAndText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "F5TTS_Advance":     "🎤 F5‑TTS‑Advance TH",
    "F5TTSAudioInputs":  "🇹🇭 F5‑TTS‑TH รับข้อมูลจาก Inputs 🇹🇭",
    "SaveAudioAndText":  "💾 🇹🇭 Save Audio & Text TH",
}
