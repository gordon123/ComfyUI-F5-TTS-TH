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

# ----- Import English-to-Thai transliteration (local file) -----
# Place ARPABET2ThaiScript.py alongside this F5TTS_Advance.py
from .ARPABET2ThaiScript import eng_to_thai_translit

# ----- Add F5-TTS source path for imports -----
f5tts_src = os.path.join(Install.base_path, "src")
sys.path.insert(0, f5tts_src)

# Main inference imports
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
)
from f5_tts.cleantext.number_tha import replace_numbers_with_thai
from f5_tts.cleantext.th_repeat import process_thai_repeat

# Clean up path
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
        # 1. Transliterate English segments into Thai
        translit = eng_to_thai_translit(text)
        print(f"[DEBUG] transliterated: {translit}")

        # 2. Clean numbers and Thai repeats
        cleaned = process_thai_repeat(replace_numbers_with_thai(translit))
        print(f"[DEBUG] cleaned_text: {cleaned}")

        # 3. Prepare reference audio
        wav = sample_audio["waveform"].float().contiguous()
        if wav.ndim == 3:
            wav = wav.squeeze()
        elif wav.ndim == 1:
            wav = wav.unsqueeze(0)
        sr = sample_audio["sample_rate"]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpf:
            sf.write(tmpf.name, wav.cpu().numpy().T, sr)
            ref_path = tmpf.name
        ref_audio, ref_text = preprocess_ref_audio_text(ref_path, sample_text)
        os.unlink(ref_path)
        print(f"[DEBUG] ref_text: {ref_text}")

        # 4. Load model config
        cfg_dir = os.path.join(Install.base_path, "src", "f5_tts", "configs")
        for fn in ["F5TTS_Base.yaml", "F5TTS_Base_train.yaml"]:
            p = os.path.join(cfg_dir, fn)
            if os.path.exists(p):
                model_cfg = OmegaConf.load(p).model.arch
                break
        else:
            raise FileNotFoundError("Config file not found")

        # 5. Ensure model & vocab
        mdir = os.path.join(Install.base_path, "model"); os.makedirs(mdir, exist_ok=True)
        mp = os.path.join(mdir, model_name)
        vdir = os.path.join(Install.base_path, "vocab"); os.makedirs(vdir, exist_ok=True)
        vp = os.path.join(vdir, "vocab.txt")
        if not os.path.exists(mp):
            urllib.request.urlretrieve(
                f"https://huggingface.co/VIZINTZOR/F5-TTS-THAI/resolve/main/model/{model_name}", mp
            )
        if not os.path.exists(vp):
            urllib.request.urlretrieve(
                "https://huggingface.co/VIZINTZOR/F5-TTS-THAI/resolve/main/vocab.txt", vp
            )

        # 6. Load model + vocoder
        model = load_model(DiT, model_cfg, mp, vocab_file=vp, mel_spec_type="vocos")
        vocoder = load_vocoder("vocos")
        device = comfy.model_management.get_torch_device()
        model.to(device)
        vocoder.to(device)
        if seed >= 0:
            torch.manual_seed(seed)

        # 7. fix_duration arg
        fd = None if fix_duration == 0.0 else fix_duration

        # 8. Generate audio
        audio_np, sr_out, _ = infer_process(
            ref_audio, ref_text, cleaned,
            model, vocoder=vocoder,
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
        print(f"[DEBUG] generated np: shape={audio_np.shape}, min={audio_np.min()}, max={audio_np.max()}")

        # 9. To tensor
        audio_tensor = torch.from_numpy(audio_np)
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # 10. Optional silence removal
        if remove_silence:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp2:
                sf.write(tmp2.name, audio_tensor.cpu().numpy().T, sr_out)
                remove_silence_for_generated_wav(tmp2.name)
                audio_tensor, sr_out = torchaudio.load(tmp2.name)
                os.unlink(tmp2.name)

        return {"waveform": audio_tensor, "sample_rate": sr_out}, cleaned
