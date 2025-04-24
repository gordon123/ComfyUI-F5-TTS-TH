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

# Ensure submodule
Install.check_install()

# add src path
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
from f5_tts.cleantext.number_tha import replace_numbers_with_thai
from f5_tts.cleantext.th_repeat import process_thai_repeat
sys.path.pop(0)


class F5TTS_Advance:
    @classmethod
    def INPUT_TYPES(cls):
        model_choices = [...]
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
                # ...
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "text")
    FUNCTION = "synthesize"
    CATEGORY = "🎤 Thai TTS"

    def synthesize(
        self,
        sample_audio, sample_text, text, model_name="model_500000.pt", seed=-1,
        remove_silence=True, speed=1.0, cross_fade_duration=0.15, nfe_step=32,
        cfg_strength=2.0, sway_sampling_coef=-1.0, fix_duration=0.0, max_chars=250
    ):
        # 1. clean text
        cleaned_text = process_thai_repeat(replace_numbers_with_thai(text))

        # 2. prepare ref audio
        wav = sample_audio["waveform"].float().contiguous()
        if wav.ndim == 3: wav = wav.squeeze()
        elif wav.ndim == 1: wav = wav.unsqueeze(0)
        sr = sample_audio["sample_rate"]
        print(f"[DEBUG] ref-waveform shape = {wav.shape}, sr = {sr}")

        # write out direct ref.wav for check
        tmp_ref = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp_ref.name, wav.cpu().numpy().T, sr)
        print(f"[DEBUG] wrote debug_ref.wav => play this: {tmp_ref.name}")

        ref_audio, ref_text = preprocess_ref_audio_text(tmp_ref.name, sample_text)
        os.unlink(tmp_ref.name)
        print(f"[DEBUG] ref_text = {ref_text}")

        # 3. load config, model, vocab as before...
        cfg_folder = os.path.join(Install.base_path, "src", "f5_tts", "configs")
        cfg_candidates = ["F5TTS_Base.yaml", "F5TTS_Base_train.yaml"]
        cfg_path = next((os.path.join(cfg_folder,c) for c in cfg_candidates if os.path.exists(os.path.join(cfg_folder, c))), None)
        model_cfg = OmegaConf.load(cfg_path).model.arch

        model_dir = os.path.join(Install.base_path, "model"); os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, model_name)
        vocab_dir = os.path.join(Install.base_path, "vocab"); os.makedirs(vocab_dir, exist_ok=True)
        vocab_path = os.path.join(vocab_dir, "vocab.txt")
        # ensure downloads...

        model = load_model(DiT, model_cfg, model_path, vocab_file=vocab_path, mel_spec_type="vocos")
        vocoder = load_vocoder("vocos")
        device = comfy.model_management.get_torch_device()
        model.to(device); vocoder.to(device)

        if seed >= 0:
            torch.manual_seed(seed)

        # 4. inference
        audio_np, sr_out, _ = infer_process(
            ref_audio, ref_text, cleaned_text,
            model, vocoder=vocoder,
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
        print(f"[DEBUG] infer_process => np shape={audio_np.shape}, dtype={audio_np.dtype}, min={audio_np.min()}, max={audio_np.max()}")

        # write raw output
        raw_out = "debug_raw.wav"
        sf.write(raw_out, audio_np, sr_out)
        print(f"[DEBUG] wrote debug_raw.wav => play this first")

        audio_tensor = torch.from_numpy(audio_np)
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # 5. optional silence removal (comment out to test)
        if False and remove_silence:
            tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp_out.name, audio_tensor.cpu().numpy().T, sr_out)
            remove_silence_for_generated_wav(tmp_out.name)
            audio_tensor, sr_out = torchaudio.load(tmp_out.name)
            os.unlink(tmp_out.name)
            print(f"[DEBUG] after silence removal: tensor shape={audio_tensor.shape}")

        return {"waveform": audio_tensor, "sample_rate": sr_out}, cleaned_text
