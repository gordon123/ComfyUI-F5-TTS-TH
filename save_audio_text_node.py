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

    RETURN_TYPES = ("STRING",)  # เพื่อให้ node ไม่ถูกข้าม
    RETURN_NAMES = ("log",)
    FUNCTION = "save_both"
    CATEGORY = "🇹🇭 Thai / Audio"

    def save_both(self, audio, text, filename_prefix):
        output_dir = "/workspace/ComfyUI/output/audio_output"
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{filename_prefix}_{timestamp}"

        waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate")

        if waveform is None or waveform.numel() == 0:
            log = "❌ ไม่พบข้อมูล waveform หรือ waveform ว่างเปล่า"
            print(log)
            return (log,)

        print(f"🎧 Preparing to save waveform with shape: {waveform.shape} and sample rate: {sample_rate}")
        audio_path = os.path.join(output_dir, f"{base_name}.wav")
        torchaudio.save(audio_path, waveform.float().cpu(), sample_rate, format="wav")
        print(f"✅ WAV saved at: {audio_path}")

        text_path = os.path.join(output_dir, f"{base_name}.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text.strip())
        print(f"✅ TXT saved at: {text_path}")

        return (f"✅ Saved to: {audio_path} & {text_path}",)
