import os
import torchaudio
from datetime import datetime
import io

class SaveAudioAndText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "text": ("STRING", {"multiline": True}),
                "filename_prefix": ("STRING", {"default": "F5TTSTH"}),
                "extension": (["wav", "flac", "mp3"], {"default": "wav"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "save_both"
    CATEGORY = "🇹🇭 Thai TTS"

    def save_both(self, audio, text, filename_prefix, extension):
        output_dir = "/workspace/ComfyUI/output/audio_output"
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{filename_prefix}_{timestamp}"
        file_name = f"{base_name}.{extension}"
        full_path = os.path.join(output_dir, file_name)

        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        # force mono
        if waveform.dim() == 3:
            waveform = waveform.squeeze()
        if waveform.shape[0] > 1:
            waveform = waveform[:1, :]
        
        # Save audio to BytesIO, then write it manually
        buff = io.BytesIO()
        torchaudio.save(buff, waveform.float().cpu(), sample_rate, format=extension.upper())
        buff.seek(0)

        with open(full_path, 'wb') as f:
            f.write(buff.getvalue())
        print(f"✅ Audio saved: {full_path}")

        # Save text
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text.strip())
        print(f"✅ Text saved: {txt_path}")

        return (audio,)
