import os
import re
import io
import torchaudio
from datetime import datetime

class SaveAudioAndText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio":        ("AUDIO",),
                "text":         ("STRING", {"multiline": True}),
                "filename_prefix": ("STRING", {"default": "F5TTSTH"}),
                "extension":    (["wav", "flac", "mp3"], {"default": "wav"}),
            }
        }

    # ส่งกลับ (waveform, filename)
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "filename")
    FUNCTION = "save_both"
    CATEGORY = "🇹🇭 Thai TTS"

    def save_both(self, audio, text, filename_prefix, extension):
        output_dir = "/workspace/ComfyUI/output/audio_output"
        os.makedirs(output_dir, exist_ok=True)

        # หาไฟล์ทั้งหมดที่ตรงกับ prefix และนับเลขท้าย
        pattern = re.compile(rf"^{re.escape(filename_prefix)}_(\d{{3}})\.{re.escape(extension)}$")
        existing = []
        for fn in os.listdir(output_dir):
            m = pattern.match(fn)
            if m:
                existing.append(int(m.group(1)))
        next_idx = (max(existing) + 1) if existing else 1
        idx_str = f"{next_idx:03d}"

        # สร้างชื่อไฟล์
        base_name = f"{filename_prefix}_{idx_str}"
        file_name = f"{base_name}.{extension}"
        full_path = os.path.join(output_dir, file_name)

        # เตรียม waveform
        waveform   = audio["waveform"]
        sample_rate = audio["sample_rate"]

        # บังคับ mono
        if waveform.dim() == 3:
            waveform = waveform.squeeze()
        if waveform.shape[0] > 1:
            waveform = waveform[:1, :]

        # เซฟเสียงลงไฟล์
        buff = io.BytesIO()
        torchaudio.save(
            buff,
            waveform.float().cpu(),
            sample_rate,
            format=extension.upper()
        )
        buff.seek(0)
        with open(full_path, 'wb') as f:
            f.write(buff.getvalue())
        print(f"✅ Audio saved: {full_path}")

        # เซฟข้อความ
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text.strip())
        print(f"✅ Text saved: {txt_path}")

        # คืนค่า waveform พร้อมชื่อไฟล์
        return (audio, file_name)
