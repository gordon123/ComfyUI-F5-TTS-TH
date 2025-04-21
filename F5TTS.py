# ComfyUI-F5-TTS-TH: Thai-only TTS node
from pathlib import Path
import os
import sys
import tempfile
import torch
import torchaudio
import numpy as np
import re
from omegaconf import OmegaConf
import comfy
from comfy.utils import ProgressBar
from cached_path import cached_path
from .Install import Install

# Ensure the Thai submodule is initialized (contains ckpts/thai/...)
Install.check_install()

# Add F5-TTS-THAI source to Python path for imports
f5tts_src = os.path.join(Install.f5TTSPath, "src")
sys.path.insert(0, f5tts_src)
from f5_tts.model import DiT                  # noqa: E402
from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    infer_process
)                                            # noqa: E402
sys.path.pop(0)

class F5TTSThai:
    tooltip_seed = "Seed. -1 = random"
    tooltip_speed = "Speed. >1.0 slower. <1.0 faster"

    @staticmethod
    def load_voice(ref_audio, ref_text):
        # Prepare reference audio and text
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio, ref_text)
        return {"ref_audio": ref_audio, "ref_text": ref_text}

    def load_model_thai(self):
        # Download or load checkpoint and vocab via cached_path
        ckpt_url  = "hf://VIZINTZOR/F5-TTS-THAI/model_475000_FP16.pt"
        vocab_url = "hf://VIZINTZOR/F5-TTS-THAI/vocab.txt"
        ckpt  = str(cached_path(ckpt_url))
        vocab = str(cached_path(vocab_url))
        # Load base config for F5-TTS
        cfg_path  = os.path.join(Install.f5TTSPath, "src/f5_tts/configs/F5TTS_Base.yaml")
        model_cfg = OmegaConf.load(cfg_path).model.arch
        # Instantiate model and vocoder
        model   = load_model(DiT, model_cfg, ckpt, vocab, mel_spec_type="vocos")
        vocoder = load_vocoder("vocos")
        device  = comfy.model_management.get_torch_device()
        # Move to half precision on GPU
        model   = model.half().to(device)
        vocoder = vocoder.half().to(device)
        return model, vocoder, "vocos"

    def generate(self, voice, text, seed, speed):
        # Load Thai-only model
        model, vocoder, mel_spec = self.load_model_thai()
        # Set seed if provided
        if seed >= 0:
            torch.manual_seed(seed)
        # Generate waveform
        audio, sample_rate, _ = infer_process(
            voice["ref_audio"], voice["ref_text"], text,
            model, vocoder=vocoder, mel_spec_type=mel_spec,
            device=comfy.model_management.get_torch_device()
        )
        waveform = torch.from_numpy(audio).unsqueeze(0)
        return {"waveform": waveform, "sample_rate": sample_rate}

class F5TTSAudioInputs:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "sample_audio": ("AUDIO",),
            "sample_text":  ("STRING", {"default": "Text of sample_audio"}),
            "speech":       ("STRING", {"multiline": True, "default": "สวัสดีครับ"}),
            "seed":         ("INT",    {"default": -1, "min": -1, "tooltip": F5TTSThai.tooltip_seed}),
            "speed":        ("FLOAT",  {"default": 1.0, "step": 0.01, "tooltip": F5TTSThai.tooltip_speed}),
        }}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "create"
    CATEGORY = "audio"

    def create(self, sample_audio, sample_text, speech, seed=-1, speed=1.0):
        # Save sample_audio to a temporary WAV
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        torchaudio.save(tmp.name, sample_audio["waveform"], sample_audio["sample_rate"])
        # Prepare voice dict
        voice = F5TTSThai.load_voice(tmp.name, sample_text)
        os.unlink(tmp.name)
        # Generate Thai speech
        audio = F5TTSThai().generate(voice, speech, seed, speed)
        return (audio,)
