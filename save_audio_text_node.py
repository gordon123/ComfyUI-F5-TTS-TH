import os
import io
import torchaudio
from datetime import datetime

class SaveAudioAndText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "text": ("STRING", {"multiline": True}),
                "filename_prefix": ("STRING", {"default": "F5TTSTH"}),
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

        waveform = audio["waveform"].float().cpu()
        sample_rate = audio["sample_rate"]

        # Ensure waveform is 2D (channels, samples)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 3:
            waveform = waveform.squeeze()
        if waveform.shape[0] > 1:
            print("⚠️ Multi-channel detected. Saving only the first channel.")
            waveform = waveform[:1, :]

        # ✅ Use BytesIO for safety, then save to disk
        buff = io.BytesIO()
        torchaudio.save(buff, waveform, sample_rate, format="wav")
        buff.seek(0)

        with open(audio_path, "wb") as f:
            f.write(buff.getvalue())
        print(f"✅ WAV saved at: {audio_path}")

        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text.strip())
        print(f"✅ TXT saved at: {text_path}")

        return (audio,)
