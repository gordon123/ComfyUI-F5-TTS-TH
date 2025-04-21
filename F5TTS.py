# ComfyUI-F5-TTS-TH: Thai-only TTS node
import os
import sys
import tempfile
import torch
import torchaudio
from pathlib import Path
from omegaconf import OmegaConf
import comfy
from comfy.utils import ProgressBar
from cached_path import cached_path
from .Install import Install

# Ensure the Thai submodule is initialized
Install.check_install()

# Add F5-TTS-THAI source to Python path
f5tts_src = os.path.join(Install.f5TTSPath, "src")
sys.path.insert(0, f5tts_src)
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    infer_process
)
sys.path.pop(0)

TOOLTIP_SEED = "Seed. -1 = random"
TOOLTIP_SPEED = "Speed. >1.0 slower. <1.0 faster"

class F5TTSThai:
    @staticmethod
    def get_available_models():
        model_dir = os.path.join(Install.f5TTSPath, "ckpts", "thai")
        return sorted([f.name for f in Path(model_dir).glob("*.pt")])

    @staticmethod
    def load_voice(ref_audio, ref_text):
        print(f"🔍 Preprocessing reference audio at: {ref_audio}")
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio, ref_text)
        return {"ref_audio": ref_audio, "ref_text": ref_text}

    def load_model_thai(self, model_name="model_475000_FP16.pt"):
        print(f"🌟 Loading model: {model_name}")
        model_path = os.path.join(Install.f5TTSPath, "ckpts", "thai", model_name)
        vocab_path = os.path.join(Install.f5TTSPath, "ckpts", "thai", "vocab.txt")

        cfg_candidates = [
            "F5TTS_Base.yaml",
            "F5TTS_Base_train.yaml"
        ]
        cfg_path = None
        for candidate in cfg_candidates:
            potential_path = os.path.join(Install.f5TTSPath, "src/f5_tts/configs", candidate)
            if os.path.exists(potential_path):
                cfg_path = potential_path
                print(f"✅ Found config: {cfg_path}")
                break

        if cfg_path is None:
            raise FileNotFoundError("❌ ไม่พบไฟล์ config ที่ต้องใช้สำหรับโหลดโมเดล F5-TTS")

        model_cfg = OmegaConf.load(cfg_path).model.arch

        model = load_model(DiT, model_cfg, model_path, vocab_file=vocab_path, mel_spec_type="vocos")
        vocoder = load_vocoder("vocos")

        device = comfy.model_management.get_torch_device()
        model = model.to(device)
        vocoder = vocoder.to(device)

        return model, vocoder, "vocos"

    def generate(self, voice, text, seed, speed, model_name="model_475000_FP16.pt"):
        model, vocoder, mel_spec = self.load_model_thai(model_name)
        if seed >= 0:
            torch.manual_seed(seed)
        audio, sample_rate, _ = infer_process(
            voice["ref_audio"], voice["ref_text"], text,
            model, vocoder=vocoder, mel_spec_type=mel_spec,
            device=comfy.model_management.get_torch_device()
        )
        waveform = torch.from_numpy(audio)
        print(f"📦 Generated waveform shape: {waveform.shape}")
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
            print(f"↪️ Reshaped to 2D: {waveform.shape}")
        return {"waveform": waveform, "sample_rate": sample_rate}


class F5TTSAudioInputs:
    @classmethod
    def INPUT_TYPES(cls):
        model_choices = F5TTSThai.get_available_models()
        return {"required": {
            "sample_audio": ("AUDIO",),
            "sample_text":  ("STRING", {"default": "Text of sample_audio"}),
            "speech":       ("STRING", {"multiline": True, "default": "สวัสดีครับ"}),
            "model_name":   (model_choices, {"default": "model_475000_FP16.pt"}),
            "seed":         ("INT",    {"default": -1, "min": -1, "tooltip": TOOLTIP_SEED}),
            "speed":        ("FLOAT",  {"default": 1.0, "step": 0.01, "tooltip": TOOLTIP_SPEED}),
        }}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "create"
    CATEGORY = "🇹🇭 Thai / Audio"

    def create(self, sample_audio, sample_text, speech, model_name="model_475000_FP16.pt", seed=-1, speed=1.0):
        waveform = sample_audio["waveform"]
        sample_rate = sample_audio["sample_rate"]
        print(f"📅 Received waveform shape: {waveform.shape}")

        if waveform.ndim == 3:
            print(f"🔀 Detected 3D waveform, squeezing...")
            waveform = waveform.squeeze()
            print(f"📂 After squeeze: {waveform.shape}")

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
            print(f"↪️ Converted 1D to 2D: {waveform.shape}")
        elif waveform.ndim == 2 and waveform.shape[0] > waveform.shape[1]:
            waveform = waveform.transpose(0, 1)
            print(f"🔀 Transposed waveform to (channels, samples): {waveform.shape}")

        if waveform.ndim != 2:
            raise RuntimeError(f"❌ Input waveform must be 2D (channels, samples). Got shape: {waveform.shape}")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            torchaudio.save(tmp.name, waveform, sample_rate)
            tmp_path = tmp.name
            print(f"📂 Saved temp WAV: {tmp_path}")

        voice = F5TTSThai.load_voice(tmp_path, sample_text)
        os.unlink(tmp_path)
        print("🧐 Voice reference loaded. Proceeding to generate speech...")
        audio = F5TTSThai().generate(voice, speech, seed, speed, model_name)

        return (audio,)
