# __init__.py
from .F5TTS import F5TTSAudioInputs  # ลบ F5TTSThai ออก
from .save_audio_text_node import SaveAudioAndText

NODE_CLASS_MAPPINGS = {
    "F5TTSAudioInputs":   F5TTSAudioInputs,
    "SaveAudioAndText":  SaveAudioAndText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "F5TTSAudioInputs":   "🇹🇭 F5‑TTS‑TH รับข้อมูลจาก Inputs 🇹🇭",
    "SaveAudioAndText": "💾 🇹🇭 Save Audio & Text TH",
}

