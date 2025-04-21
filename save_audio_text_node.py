import os
import torchaudio
import torch
from datetime import datetime

class SaveAudioAndText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "text": ("STRING", {"multiline": True}),
                "filename_prefix": ("STRING", {"default": "f5tts_output"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "save_both"
    CATEGORY = "🇹🇭 Thai TTS"

    def save_both(self, audio, text, filename_prefix):
        output_dir = "/workspace/ComfyUI/output/audio_output"
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{filename_prefix}_{timestamp}"

        # 🧱 ป้องกัน waveform ที่มีปัญหา
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        # 🔍 ตรวจสอบ dimension และจัดการทุกกรณี
        if waveform.ndim == 3:
            print(f"🔄 3D detected, squeezing: {waveform.shape}")
            waveform = waveform.squeeze()

        if waveform.ndim == 1:
            print(f"↪️ 1D detected, unsqueezing to mono: {waveform.shape}")
            waveform = waveform.unsqueeze(0)

        if waveform.ndim == 2 and waveform.shape[0] > waveform.shape[1]:
            print(f"🔁 Transposing (channels, samples): {waveform.shape}")
            waveform = waveform.transpose(0, 1)

        if waveform.ndim != 2:
            raise RuntimeError(f"❌ Final waveform shape invalid: {waveform.shape}")

        # 🔐 Ensure it's float32, on CPU, and contiguous
        waveform = waveform.float().cpu().contiguous()

        # 💾 Save WAV safely
        audio_path = os.path.join(output_dir, f"{base_name}.wav")
        try:
            torchaudio.save(audio_path, waveform, sample_rate, format="wav", encoding="PCM_S", bits_per_sample=16)
            print(f"✅ WAV saved at: {audio_path}")
        except Exception as e:
            print(f"❌ Failed to save WAV: {e}")
            raise

        # 📜 Save TXT
        text_path = os.path.join(output_dir, f"{base_name}.txt")
        try:
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text.strip())
            print(f"✅ TXT saved at: {text_path}")
        except Exception as e:
            print(f"❌ Failed to save TXT: {e}")
            raise

        return (audio,)
