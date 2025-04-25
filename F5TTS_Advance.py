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
# Import English transliteration
from f5_tts.cleantext.ARPABET2ThaiScript import eng_to_thai_translit
from f5_tts.cleantext.number_tha import replace_numbers_with_thai
from f5_tts.cleantext.th_repeat import process_thai_repeat
from f5_tts.cleantext.ARPABET2ThaiScript import eng_to_thai_translit  # เพิ่ม import ของ transliteration
sys.path.pop(0)

class F5TTS_Advance:
    @classmethod
    def INPUT_TYPES(cls):
        model_choices = [
            "model_50000.pt", "model_80000.pt",
            "model_100000.pt", "model_130000.pt", "model_150000.pt", "model_200000.pt",
            "model_250000.pt", "model_250000_FP16.pt",
            "model_350000.pt", "model_430000.pt",
            "model_475000.pt", "model_475000_FP16.pt",
            "model_500000.pt", "model_500000_FP16.pt"
        ]
        return {
            "required": {
                "sample_audio": ("AUDIO",),
                "sample_text": ("STRING", {"default": "Text of sample_audio"}),
                "text": ("STRING", {"multiline": True, "default": "สวัสดีครับ"}),
                "model_name": (model_choices, {"default": "model_500000.pt"}),
                "seed": ("INT", {"default": -1, "min": -1}),
            },
            "optional": {
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
        speed=1.0,
        cross_fade_duration=0.15,
        nfe_step=32,
        cfg_strength=2.0,
        sway_sampling_coef=-1.0,
        fix_duration=0.0,
        max_chars=250,
    ):
        # 1. Transliterate English to Thai script
        text_translit = eng_to_thai_translit(text)
        print(f"[DEBUG] after transliteration: {text_translit}")

        # 2. Clean input text: numbers and repeats
        cleaned_text = process_thai_repeat(replace_numbers_with_thai(text_translit))
        print(f"[DEBUG] cleaned_text: {cleaned_text}")

        # Prepare reference audio
        waveform = sample_audio["waveform"].float().contiguous()
        print(f"[DEBUG] ref raw waveform shape: {waveform.shape}")
        if waveform.ndim == 3:
            waveform = waveform.squeeze()
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        sr = sample_audio["sample_rate"]
        print(f"[DEBUG] ref sample_rate: {sr}")

        # save reference for debugging
        ref_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(ref_tmp.name, waveform.cpu().numpy().T, sr)
        print(f"[DEBUG] wrote reference WAV: {ref_tmp.name}")

        ref_audio, ref_text = preprocess_ref_audio_text(ref_tmp.name, sample_text)
        print(f"[DEBUG] preprocess_ref_audio_text -> ref_audio: {ref_audio}, ref_text: {ref_text}")
        os.unlink(ref_tmp.name)

        # Load model config
        cfg_folder = os.path.join(Install.base_path, "src", "f5_tts", "configs")
        cfg_candidates = ["F5TTS_Base.yaml", "F5TTS_Base_train.yaml"]
        cfg_path = next((os.path.join(cfg_folder, c) for c in cfg_candidates
                         if os.path.exists(os.path.join(cfg_folder, c))), None)
        if not cfg_path:
            raise FileNotFoundError("Config file not found in configs")
        model_cfg = OmegaConf.load(cfg_path).model.arch
        print(f"[DEBUG] Using cfg_path: {cfg_path}")

        # Prepare model & vocab paths
        model_dir = os.path.join(Install.base_path, "model")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, model_name)
        vocab_dir = os.path.join(Install.base_path, "vocab")
        os.makedirs(vocab_dir, exist_ok=True)
        vocab_path = os.path.join(vocab_dir, "vocab.txt")
        if not os.path.exists(model_path):
            print(f"[DEBUG] Downloading model: {model_name}")
            urllib.request.urlretrieve(
                f"https://huggingface.co/VIZINTZOR/F5-TTS-THAI/resolve/main/model/{model_name}",
                model_path
            )
        if not os.path.exists(vocab_path):
            print("[DEBUG] Downloading vocab.txt")
            urllib.request.urlretrieve(
                "https://huggingface.co/VIZINTZOR/F5-TTS-THAI/resolve/main/vocab.txt",
                vocab_path
            )

        # Load model and vocoder
        print(f"[DEBUG] Loading model from: {model_path}")
        model = load_model(DiT, model_cfg, model_path, vocab_file=vocab_path, mel_spec_type="vocos")
        vocoder = load_vocoder("vocos")
        device = comfy.model_management.get_torch_device()
        model.to(device)
        vocoder.to(device)
        print(f"[DEBUG] Model and vocoder on device: {device}")

        # Seed
        if seed >= 0:
            torch.manual_seed(seed)
            print(f"[DEBUG] seed set to: {seed}")

        # fix_duration: 0 -> None
        fd = None if fix_duration == 0.0 else fix_duration
        print(f"[DEBUG] Parameters: speed={speed}, cross_fade_duration={cross_fade_duration}, nfe_step={nfe_step}, cfg_strength={cfg_strength}, sway_sampling_coef={sway_sampling_coef}, fix_duration={fd}, max_chars={max_chars}")

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
            fix_duration=fd,
            set_max_chars=max_chars,
            mel_spec_type="vocos",
            device=device
        )
        print(f"[DEBUG] infer_process output: shape={audio_np.shape}, dtype={audio_np.dtype}, min={audio_np.min()}, max={audio_np.max()}")

        # write raw for debug
        raw_debug = "debug_raw.wav"
        sf.write(raw_debug, audio_np, sr_out)
        print(f"[DEBUG] wrote raw debug file: {raw_debug}")

        audio_tensor = torch.from_numpy(audio_np)
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # Remove silence if requested
        if remove_silence:
            print("[DEBUG] Removing silence from generated audio...")
            tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp_out.name, audio_tensor.cpu().numpy().T, sr_out)
            remove_silence_for_generated_wav(tmp_out.name)
            audio_tensor, sr_out = torchaudio.load(tmp_out.name)
            os.unlink(tmp_out.name)
            print(f"[DEBUG] Post-silence removal tensor shape: {audio_tensor.shape}")

        return {"waveform": audio_tensor, "sample_rate": sr_out}, cleaned_text
