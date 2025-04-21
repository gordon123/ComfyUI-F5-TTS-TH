import os
import torchaudio
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

    RETURN_TYPES = ()
    FUNCTION = "save_both"
    CATEGORY = "🇹🇭 Thai TTS"

    def save_both(self, audio, text, filename_prefix):
        # 🔐 ป้องกัน path
        output_dir = "/workspace/ComfyUI/output/audio_output"
        os.makedirs(output_dir, exist_ok=True)

        # 📅 ตั้งชื่อไฟล์ด้วย timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{filename_prefix}_{timestamp}"

        # 🎧 Save เป็น WAV (ปลอดภัยสุด)
        audio_path = os.path.join(output_dir, f"{base_name}.wav")
        waveform = audio["waveform"].float().cpu()  # ปรับให้เป็น float32
        sample_rate = audio["sample_rate"]
        torchaudio.save(audio_path, waveform, sample_rate, format="wav")
        print(f"✅ WAV saved at: {audio_path}")

        # ✍️ Save เป็น TXT
        text_path = os.path.join(output_dir, f"{base_name}.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text.strip())
        print(f"✅ TXT saved at: {text_path}")

        return ()
