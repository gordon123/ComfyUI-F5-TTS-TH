# ComfyUI-F5-TTS-TH: Thai-only TTS node
import os
import sys
import tempfile
import torch
import torchaudio
import numpy as np
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
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio, ref_text)
        return {"ref_audio": ref_audio, "ref_text": ref_text}

    def load_model_thai(self, model_name="model_475000_FP16.pt"):
        model_path = os.path.join(Install.f5TTSPath, "ckpts", "thai", model_name)
        vocab_path = os.path.join(Install.f5TTSPath, "ckpts", "thai", "vocab.txt")
        cfg_path = os.path.join(Install.f5TTSPath, "src/f5_tts/configs/F5TTS_Base.yaml")
        model_cfg = OmegaConf.load(cfg_path).model.arch

        model = load_model(DiT, model_cfg, model_path, vocab_file=vocab_path, mel_spec_type="vocos")
        vocoder = load_vocoder("vocos")

        device = comfy.model_management.get_torch_device()
        if torch.cuda.is_available():
            model = model.half().to(device)
            vocoder = vocoder.half().to(device)
        else:
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
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
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

        # 🛡 Ensure waveform is 2D: (channels, samples)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim > 2:
            waveform = waveform.squeeze()

        if waveform.ndim != 2:
            raise RuntimeError(f"❌ Input waveform must be 2D (channels, samples). Got shape: {waveform.shape}")

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        torchaudio.save(tmp.name, waveform, sample_rate)

        voice = F5TTSThai.load_voice(tmp.name, sample_text)
        os.unlink(tmp.name)
        audio = F5TTSThai().generate(voice, speech, seed, speed, model_name)

        return (audio,)
