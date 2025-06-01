import os
import sys
import tempfile
import torch
import torchaudio
import soundfile as sf
from omegaconf import OmegaConf
import comfy
from .Install import Install
from huggingface_hub import HfApi, hf_hub_download

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
    """
    โค้ดนี้ปรับให้:
    1. ดึงชื่อไฟล์ .pt จากรีโป VIZINTZOR/F5-TTS-THAI ในโฟลเดอร์ model/ มาเป็น dropdown
    2. ให้ผู้ใช้พิมพ์ <repo_id>/model/<filename>.pt ได้เอง
    """

    # รีโปที่เราจะเฝ้ามองเพื่อดึงชื่อไฟล์ .pt จากโฟลเดอร์ model/
    WATCHED_REPOS = [
        "VIZINTZOR/F5-TTS-THAI",
        "Muscari/F5-TTS-TH_Finetuned",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        api = HfApi()
        model_choices = []

        # ดึงชื่อไฟล์ .pt จาก "model/" ทุกรีโปใน WATCHED_REPOS
        for repo in cls.WATCHED_REPOS:
            try:
                files = api.list_repo_files(repo_id=repo)
            except Exception:
                continue

            for fn in files:
                # แค่ไฟล์ที่อยู่ในโฟลเดอร์ model/ และลงท้าย .pt
                if fn.startswith("model/") and fn.endswith(".pt"):
                    model_choices.append(f"{repo}/{fn}")

        model_choices = sorted(model_choices)
        default_choice = model_choices[-1] if model_choices else ""

        return {
            "required": {
                "sample_audio": ("AUDIO",),
                "sample_text": ("STRING", {"default": "Text of sample_audio"}),
                "text": ("STRING", {"multiline": True, "default": "สวัสดีครับ"}),
                # model_path: dropdown + free-form ให้กรอก <repo_id>/model/<filename>.pt
                "model_path": (model_choices, {
                    "default": default_choice,
                    "description": "เลือกหรือพิมพ์ <repo_id>/model/<filename>.pt เช่น VIZINTZOR/F5-TTS-THAI/model/model_650000.pt"
                }),
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
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "text")
    FUNCTION = "synthesize"
    CATEGORY = "🇹🇭 Thai TTS"

    def synthesize(
        self,
        sample_audio,
        sample_text,
        text,
        model_path="",        # เป็น <repo_id>/model/<filename>.pt
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
        # ── 1. เตรียมข้อความ: transliterate + clean ──────────────────
        translit = eng_to_thai_translit(text)
        cleaned = process_thai_repeat(replace_numbers_with_thai(translit))

        # ── 2. เตรียม reference audio ───────────────────────────────
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

        # ── 3. โหลด config ของโมเดล ─────────────────────────────────
        cfg_dir = os.path.join(Install.base_path, "src", "f5_tts", "configs")
        for fn in ["F5TTS_Base.yaml", "F5TTS_Base_train.yaml"]:
            p = os.path.join(cfg_dir, fn)
            if os.path.exists(p):
                model_cfg = OmegaConf.load(p).model.arch
                break
        else:
            raise FileNotFoundError("Config file not found")

        # ── 4. แยก model_path เป็น repo_id กับ relative_path ─────────────────
        try:
            repo_id, relpath = model_path.strip().split("/", 1)
        except ValueError:
            raise ValueError(
                "กรุณากรอกเป็น <repo_id>/model/<filename>.pt เช่น VIZINTZOR/F5-TTS-THAI/model/model_650000.pt"
            )

        if not relpath.startswith("model/") or not relpath.endswith(".pt"):
            raise ValueError("ต้องระบุไฟล์จากโฟลเดอร์ 'model/' และลงท้ายด้วย .pt")

        # ── 5. ดาวน์โหลดไฟล์ .pt มาเก็บโฟลเดอร์ model/ ของ custom node ─────
        mdir = os.path.join(Install.base_path, "model")
        os.makedirs(mdir, exist_ok=True)
        local_model_path = os.path.join(mdir, os.path.basename(relpath))

        if not os.path.exists(local_model_path):
            try:
                local_model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=relpath,
                    cache_dir=mdir,
                    local_dir=mdir
                )
            except Exception as e:
                raise RuntimeError(f"❌ ไม่สามารถดาวน์โหลดโมเดลจาก {repo_id}/{relpath} ได้: {e}")

        # ── 6. เตรียม vocab.txt จาก repo เดียวกัน ────────────────────
        vdir = os.path.join(Install.base_path, "vocab")
        os.makedirs(vdir, exist_ok=True)
        vocab_path = os.path.join(vdir, "vocab.txt")

        if not os.path.exists(vocab_path):
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename="vocab.txt",
                    cache_dir=vdir,
                    local_dir=vdir
                )
            except Exception as e:
                raise RuntimeError(f"❌ ไม่สามารถดาวน์โหลด vocab.txt จาก {repo_id} ได้: {e}")

        # ── 7. โหลดโมเดล + vocoder ลง device ────────────────────────
        model = load_model(DiT, model_cfg, local_model_path, vocab_file=vocab_path, mel_spec_type="vocos")
        vocoder = load_vocoder("vocos")
        device = comfy.model_management.get_torch_device()
        model.to(device)
        vocoder.to(device)
        if seed >= 0:
            torch.manual_seed(seed)

        # ── 8. จัด fix_duration ────────────────────────────────────
        fd = None if fix_duration == 0.0 else fix_duration

        # ── 9. Inference: สร้างเสียง ────────────────────────────────
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

        # ── 10. แปลงเป็น Tensor ───────────────────────────────────
        audio_tensor = torch.from_numpy(audio_np)
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # ── 11. ตัด silence ถ้าต้องการ ───────────────────────────────
        if remove_silence:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp2:
                sf.write(tmp2.name, audio_tensor.cpu().numpy().T, sr_out)
                remove_silence_for_generated_wav(tmp2.name)
                audio_tensor, sr_out = torchaudio.load(tmp2.name)
                try:
                    os.unlink(tmp2.name)
                except PermissionError:
                    print(f"PermissionError: Unable to delete {tmp2.name}. Please deleteมัน manually.")
                except Exception as e:
                    print(f"Error deleting {tmp2.name}: {e}")

        return {"waveform": audio_tensor, "sample_rate": sr_out}, cleaned
