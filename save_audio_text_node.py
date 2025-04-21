from datetime import datetime
import os
import torchaudio
import torch

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

        # 🎧 เตรียม waveform อย่างปลอดภัย
        waveform = audio["waveform"]
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform)

        waveform = waveform.float().contiguous().cpu()
        sample_rate = int(audio["sample_rate"])

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 3:
            waveform = waveform.squeeze()
        if waveform.ndim == 2 and waveform.shape[0] > waveform.shape[1]:
            waveform = waveform.transpose(0, 1)

        # ✅ ตัดเหลือแค่ 1 channel ถ้ามีหลายอัน
        if waveform.shape[0] > 1:
            waveform = waveform[:1, :]

        # 💾 Save WAV แบบปลอดภัย (sox_io supports PCM encoding)
        audio_path = os.path.join(output_dir, f"{base_name}.wav")
        torchaudio.save(audio_path, waveform, sample_rate, format="wav")
        print(f"✅ WAV saved at: {audio_path}")

        # ✍️ Save .txt
        text_path = os.path.join(output_dir, f"{base_name}.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text.strip())
        print(f"✅ TXT saved at: {text_path}")

        return ({"waveform": waveform, "sample_rate": sample_rate},)
