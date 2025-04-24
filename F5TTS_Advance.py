import os
import sys
import tempfile
import torch
import torchaudio
import soundfile as sf
from omegaconf import OmegaConf
import comfy
from .Install import Install
import urllib.request

# Ensure the submodule is initialized
Install.check_install()

# Add submodule source to Python path for inference
f5tts_src = os.path.join(Install.base_path, "src")
sys.path.insert(0, f5tts_src)
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav
)
# Import Thai text cleaning
from f5_tts.cleantext.number_tha import replace_numbers_with_thai
from f5_tts.cleantext.th_repeat import process_thai_repeat
sys.path.pop(0)

class F5TTS_Advance:
    # Widget properties: expose parameters on the node
    PROPERTIES = {
        "remove_silence": ("BOOL", {"default": True}),
        "cross_fade_duration": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
        "nfe_step": ("INT", {"default": 32, "min": 1, "max": 128}),
        "cfg_strength": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
        "sway_sampling_coef": ("FLOAT", {"default": -1.0, "min": -5.0, "max": 5.0, "step": 0.1}),
        "fix_duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1}),
        "max_chars": ("INT", {"default": 250, "min": 1, "max": 1000}),
        "speed": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
    }

    @classmethod
    def INPUT_TYPES(cls):
        model_choices = [
            "model_100000.pt", "model_130000.pt", "model_150000.pt", "model_200000.pt",
            "model_250000.pt", "model_350000.pt", "model_430000.pt", "model_475000.pt", "model_500000.pt"
        ]
        print(f"[DEBUG] INPUT_TYPES model choices: {model_choices}")
        return {
            "required": {
                "sample_audio": ("AUDIO",),
                "sample_text": ("STRING", {"default": "Text of sample_audio"}),
                "text": ("STRING", {"multiline": True, "default": "สวัสดีครับ"}),
                "model_name": (model_choices, {"default": "model_500000.pt"}),
                "seed": ("INT", {"default": -1, "min": -1}),
                "remove_silence": ("BOOL", {"default": True}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "cross_fade_duration": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "nfe_step": ("INT", {"default": 32, "min": 1, "max": 128}),
                "cfg_strength": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "sway_sampling_coef": ("FLOAT", {"default": -1.0, "min": -5.0, "max": 5.0, "step": 0.1}),
                "fix_duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "max_chars": ("INT", {"default": 250, "min": 1, "max": 1000}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "text")
    FUNCTION = "synthesize"
    CATEGORY = "🎤 Thai TTS"

    def synthesize(
        self,
        sample_audio,
        sample_text,
        text,
        model_name="model_500000.pt",
        seed=-1,
        remove_silence=True,
        cross_fade_duration=0.15,
        nfe_step=32,
        cfg_strength=2.0,
        sway_sampling_coef=-1.0,
        fix_duration=0.0,
        max_chars=250,
        speed=1.0,
    ):
        # Clean input text
        cleaned_text = process_thai_repeat(replace_numbers_with_thai(text))

        # Prepare reference audio
        waveform = sample_audio["waveform"].float().contiguous()
        if waveform.ndim == 3:
            waveform = waveform.squeeze()
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        sr = sample_audio["sample_rate"]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, waveform.cpu().numpy().T, sr)
            ref_path = tmp.name
        ref_audio, ref_text = preprocess_ref_audio_text(ref_path, sample_text)
        os.unlink(ref_path)

        # Load config
        cfg_folder = os.path.join(Install.base_path, "src", "f5_tts", "configs")
        cfg_candidates = ["F5TTS_Base.yaml", "F5TTS_Base_train.yaml"]
        cfg_path = next((os.path.join(cfg_folder, c) for c in cfg_candidates if os.path.exists(os.path.join(cfg_folder, c))), None)
        if not cfg_path:
            raise FileNotFoundError("Config file not found in configs")
        model_cfg = OmegaConf.load(cfg_path).model.arch

        # Prepare model & vocab
        model_dir = os.path.join(Install.base_path, "model")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, model_name)
        vocab_dir = os.path.join(Install.base_path, "vocab")
        os.makedirs(vocab_dir, exist_ok=True)
        vocab_path = os.path.join(vocab_dir, "vocab.txt")
        if not os.path.exists(model_path):
            urllib.request.urlretrieve(f"https://huggingface.co/VIZINTZOR/F5-TTS-THAI/resolve/main/model/{model_name}", model_path)
        if not os.path.exists(vocab_path):
            urllib.request.urlretrieve("https://huggingface.co/VIZINTZOR/F5-TTS-THAI/resolve/main/vocab.txt", vocab_path)

        # Load model and vocoder
        model = load_model(DiT, model_cfg, model_path, vocab_file=vocab_path, mel_spec_type="vocos")
        vocoder = load_vocoder("vocos")
        device = comfy.model_management.get_torch_device()
        model.to(device)
        vocoder.to(device)

        # Seed
        if seed >= 0:
            torch.manual_seed(seed)

        # Inference
        audio_np, sr_out, _ = infer_process(
            ref_audio,
            ref_text,
            cleaned_text,
            model,
            vocoder=vocoder,
            speed=speed,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            fix_duration=fix_duration,
            set_max_chars=max_chars,
            mel_spec_type="vocos",
            device=device
        )
        audio_tensor = torch.from_numpy(audio_np)
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # Remove silence if requested
        if remove_silence:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as outf:
                sf.write(outf.name, audio_tensor.cpu().numpy().T, sr_out)
                remove_silence_for_generated_wav(outf.name)
                audio_tensor, sr_out = torchaudio.load(outf.name)
                os.unlink(outf.name)

        return {"waveform": audio_tensor, "sample_rate": sr_out}, cleaned_text
