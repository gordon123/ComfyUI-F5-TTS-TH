from datetime import datetime
import os
import torchaudio

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

        audio_path = os.path.join(output_dir, f"{base_name}.wav")
        text_path = os.path.join(output_dir, f"{base_name}.txt")

        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        # ✅ บังคับ float32
        waveform = waveform.float().cpu().contiguous()

        # ✅ Force mono for compatibility
        if waveform.ndim == 2 and waveform.shape[0] > 1:
            print("🎧 Force mono: keeping only first channel.")
            waveform = waveform[:1, :]

        # ✅ ใช้ sox_io เพื่อแก้ปัญหา ffmpeg + flac + layout
        torchaudio.set_audio_backend("sox_io")

        try:
            torchaudio.save(audio_path, waveform, sample_rate, format="wav")
            print(f"✅ WAV saved at: {audio_path}")
        except Exception as e:
            print(f"❌ Failed to save audio: {e}")

        try:
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text.strip())
            print(f"✅ TXT saved at: {text_path}")
        except Exception as e:
            print(f"❌ Failed to save text: {e}")

        return (audio,)
