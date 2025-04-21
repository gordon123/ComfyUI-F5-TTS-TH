# __init__.py
from .F5TTS import F5TTSAudioInputs  # ลบ F5TTSThai ออก

NODE_CLASS_MAPPINGS = {
    "F5TTSAudioInputs":   F5TTSAudioInputs,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "F5TTSAudioInputs":   "🇹🇭 F5‑TTS‑TH รับข้อมูลจาก Inputs 🇹🇭",
}
