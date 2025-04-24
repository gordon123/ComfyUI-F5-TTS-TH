import os
import sys
import tempfile
import torch
import torchaudio
import soundfile as sf
from omegaconf import OmegaConf
import comfy
from comfy.utils import ProgressBar
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
    infer_process
)
# Import Thai text cleaning functions
from f5_tts.cleantext.number_tha import replace_numbers_with_thai
from f5_tts.cleantext.th_repeat import process_thai_repeat
sys.path.pop(0)

class F5TTS_Advance:
    @classmethod
    def INPUT_TYPES(cls):
        model_choices = [
            "model_100000.pt",
            "model_130000.pt",
            "model_150000.pt",
            "model_200000.pt",
            "model_250000.pt",
            "model_350000.pt",
            "model_430000.pt",
            "model_475000.pt",
            "model_500000.pt",
        ]
        print(f"[DEBUG] INPUT_TYPES model choices: {model_choices}")
        return {"required": {
            "sample_audio": ("AUDIO",),
            "sample_text": ("STRING", {"default": "Text of sample_audio"}),
            "text": ("STRING", {"multiline": True, "default": "สวัสดีครับ"}),
            "model_name": (model_choices, {"default": "model_500000.pt"}),
            "seed": ("INT", {"default": -1, "min": -1, "tooltip": "Seed. -1 = random"}),
            "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1, "tooltip": "Speed. >1.0 slower, <1.0 faster"}),
        }}

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "text")
    FUNCTION = "synthesize"
    CATEGORY = "🎤 Thai TTS"

    def synthesize(self, sample_audio, sample_text, text, model_name="model_500000.pt", seed=-1, speed=1.0):
        print("[DEBUG] Starting synthesis pipeline")
        # Clean input text
        print(f"[DEBUG] Original text: {text}")
        cleaned_text = replace_numbers_with_thai(text)
        print(f"[DEBUG] After number conversion: {cleaned_text}")
        cleaned_text = process_thai_repeat(cleaned_text)
        print(f"[DEBUG] After repeat cleaning: {cleaned_text}")

        # Prepare reference audio
        waveform = sample_audio["waveform"].float().contiguous()
        print(f"[DEBUG] Raw waveform shape: {waveform.shape}")
        if waveform.ndim == 3:
            waveform = waveform.squeeze()
            print(f"[DEBUG] Squeezed to: {waveform.shape}")
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
            print(f"[DEBUG] Unsqueezed to: {waveform.shape}")
        elif waveform.ndim > 2:
            waveform = waveform.view(waveform.shape[0], -1)
            print(f"[DEBUG] Reshaped to 2D: {waveform.shape}")
        sr = sample_audio["sample_rate"]
        print(f"[DEBUG] Sample rate: {sr}")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            data = waveform.cpu().numpy().T  # shape (samples, channels)
            sf.write(tmp.name, data, sr)
            print(f"[DEBUG] Reference WAV saved to: {tmp.name} via soundfile")
            tmp_path = tmp.name

        # Preprocess reference
        try:
            ref_audio, ref_text = preprocess_ref_audio_text(tmp_path, sample_text)
            print(f"[DEBUG] Preprocessed ref_audio path: {ref_audio}, ref_text: {ref_text}")
        finally:
            os.unlink(tmp_path)

        # Load model config
        cfg_candidates = ["F5TTS_Base.yaml", "F5TTS_Base_train.yaml"]
        cfg_path = None
        for cfg in cfg_candidates:
            candidate = os.path.join(Install.base_path, "src", "f5_tts", "configs", cfg)
            if os.path.exists(candidate):
                cfg_path = candidate
                break
        print(f"[DEBUG] Using config path: {cfg_path}")
        if cfg_path is None:
            raise FileNotFoundError("Config file not found in submodule configs")
        model_cfg = OmegaConf.load(cfg_path).model.arch

        # Prepare model and vocab paths using 'model/' and 'vocab/' directories
        model_dir = os.path.join(Install.base_path, "model")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, model_name)
        vocab_dir = os.path.join(Install.base_path, "vocab")
        os.makedirs(vocab_dir, exist_ok=True)
        vocab_path = os.path.join(vocab_dir, "vocab.txt")
        print(f"[DEBUG] model_dir: {model_dir}, model_path: {model_path}, vocab_dir: {vocab_dir}, vocab_path: {vocab_path}")


        # Download if missing
        if not os.path.exists(model_path):
            print(f"[DEBUG] Downloading model {model_name}")
            urllib.request.urlretrieve(
                f"https://huggingface.co/VIZINTZOR/F5-TTS-THAI/resolve/main/model/{model_name}",
                model_path
            )
        if not os.path.exists(vocab_path):
            print(f"[DEBUG] Downloading vocab.txt")
            urllib.request.urlretrieve(
                "https://huggingface.co/VIZINTZOR/F5-TTS-THAI/resolve/main/vocab.txt",
                vocab_path
            )

        # Load model and vocoder
        print("[DEBUG] Loading model and vocoder")
        model = load_model(DiT, model_cfg, model_path, vocab_file=vocab_path, mel_spec_type="vocos")
        vocoder = load_vocoder("vocos")
        device = comfy.model_management.get_torch_device()
        model = model.to(device)
        vocoder = vocoder.to(device)
        print(f"[DEBUG] Model and vocoder moved to device: {device}")

        # Seed
        if seed >= 0:
            torch.manual_seed(seed)
            print(f"[DEBUG] Seed set to: {seed}")

        # Inference
        print(f"[DEBUG] Calling infer_process with speed: {speed}")
        audio_np, sample_rate, _ = infer_process(
            ref_audio, ref_text, cleaned_text,
            model, vocoder=vocoder, mel_spec_type="vocos",
            device=device,
            speed=speed
        )
        print(f"[DEBUG] infer_process returned audio_np shape: {audio_np.shape}, sample_rate: {sample_rate}")

        audio_tensor = torch.from_numpy(audio_np)
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
            print(f"[DEBUG] output tensor reshaped to 2D: {audio_tensor.shape}")

        print("[DEBUG] Synthesis pipeline complete")
        return {"waveform": audio_tensor, "sample_rate": sample_rate}, cleaned_text
