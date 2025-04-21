# custom_nodes/ComfyUI-F5-TTS-TH/__init__.py

from .F5TTS import F5TTSThai, F5TTSAudioInputs

NODE_CLASS_MAPPINGS = {
    "F5TTSThai": F5TTSThai,
    "F5TTSAudioInputs": F5TTSAudioInputs,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "F5TTSThai": "🇹🇭 F5‑TTS‑TH (เสียงภาษาไทย) 🇹🇭",
    "F5TTSAudioInputs": "🇹🇭 F5‑TTS‑TH จาก Input เสียง/ข้อความ 🇹🇭",
}
