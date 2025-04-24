import os
import sys
import tempfile
import torch
import torchaudio
from omegaconf import OmegaConf
import comfy
from comfy.utils import ProgressBar
from .Install import Install
import urllib.request

# Ensure the submodule is initialized
Install.check_install()

# Use sox_io backend to avoid FFmpeg channel layout issues
torchaudio.set_audio_backend("sox_io")

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
        # Prepare and clean text
        cleaned_text = replace_numbers_with_thai(text)
        cleaned_text = process_thai_repeat(cleaned_text)

        # Save reference audio as temporary WAV
        waveform = sample_audio["waveform"].float().contiguous()
        # Ensure 2D tensor (channels, samples)
        if waveform.ndim == 3:
            waveform = waveform.squeeze()
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim > 2:
            waveform = waveform.view(waveform.shape[0], -1)
        print(f"[DEBUG] waveform shape before save: {waveform.shape}, dtype: {waveform.dtype}")
        sr = sample_audio["sample_rate"]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            try:
                import soundfile as sf
                data = waveform.cpu().numpy().T  # (samples, channels)
                sf.write(tmp.name, data, sr)
                print(f"[DEBUG] Saved WAV via soundfile to: {tmp.name}")
            except Exception as e:
                print(f"[ERROR] soundfile write failed: {e}")
                # Fallback to torchaudio.save
                torchaudio.set_audio_backend("soundfile")
                torchaudio.save(tmp.name, waveform, sr)
                print(f"[DEBUG] Saved WAV via torchaudio.save to: {tmp.name}")
            tmp_path = tmp.name

        # Preprocess reference(tmp_path, sample_text)
        os.unlink(tmp_path)

        # Load model configuration
        cfg_candidates = ["F5TTS_Base.yaml", "F5TTS_Base_train.yaml"]
        cfg_path = None
        for cfg in cfg_candidates:
            candidate = os.path.join(Install.base_path, "src", "f5_tts", "configs", cfg)
            if os.path.exists(candidate):
                cfg_path = candidate
                break
        if cfg_path is None:
            raise FileNotFoundError("Config file not found in submodule configs")
        model_cfg = OmegaConf.load(cfg_path).model.arch

        # Paths for model and vocab
        model_path = os.path.join(Install.base_path, "model", model_name)
        vocab_path = os.path.join(Install.base_path, "vocab.txt")

        # Download model/vocab if missing
        if not os.path.exists(model_path):
            print(f"⬇️ Downloading model {model_name}...")
            urllib.request.urlretrieve(
                f"https://huggingface.co/VIZINTZOR/F5-TTS-THAI/resolve/main/model/{model_name}",
                model_path
            )
            print(f"✅ Model downloaded: {model_name}")
        if not os.path.exists(vocab_path):
            print("⬇️ Downloading vocab.txt...")
            urllib.request.urlretrieve(
                "https://huggingface.co/VIZINTZOR/F5-TTS-THAI/resolve/main/vocab.txt",
                vocab_path
            )
            print("✅ vocab.txt downloaded.")

        # Load model and vocoder
        model = load_model(DiT, model_cfg, model_path, vocab_file=vocab_path, mel_spec_type="vocos")
        vocoder = load_vocoder("vocos")
        device = comfy.model_management.get_torch_device()
        model = model.to(device)
        vocoder = vocoder.to(device)

        # Seed for reproducibility
        if seed >= 0:
            torch.manual_seed(seed)

        # Generate speech
        audio_np, sample_rate, _ = infer_process(
            ref_audio, ref_text, cleaned_text,
            model, vocoder=vocoder, mel_spec_type="vocos",
            device=device,
            speed=speed
        )
        audio_tensor = torch.from_numpy(audio_np)
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        return (audio_tensor, sample_rate), cleaned_text
