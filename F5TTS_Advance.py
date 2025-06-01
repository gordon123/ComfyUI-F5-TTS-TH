import os
import sys
import tempfile
import torch
import torchaudio
import soundfile as sf
from omegaconf import OmegaConf
import comfy
from .Install import Install
from huggingface_hub import HfApi, hf_hub_url
import requests
from tqdm.auto import tqdm

# Ensure the submodule is initialized
Install.check_install()

# ----- Import English-to-Thai transliteration (local file) -----
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


def download_with_progress(url: str, local_path: str):
    """
    ดาวน์โหลดไฟล์จาก URL แล้วแสดง progress bar ด้วย tqdm
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_bytes = int(response.headers.get("content-length", 0))
    chunk_size = 8192

    with tqdm(
        total=total_bytes,
        unit="iB",
        unit_scale=True,
        desc=os.path.basename(local_path),
        leave=False,
    ) as t:
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    t.update(len(chunk))


class F5TTS_Advance:
    """
    ปรับโค้ดให้อิง path ของ submodule F5TTS-on-Pod model folder โดยไม่แก้ core:
    แล้วปรับ logic ส่ง full text เข้า infer_process เพื่อให้ chunk_text แยก newline เอง
    """

    WATCHED_REPOS = [
        "VIZINTZOR/F5-TTS-THAI",
        "Muscari/F5-TTS-TH_Finetuned",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        api = HfApi()
        suggested = []

        for repo in cls.WATCHED_REPOS:
            try:
                files = api.list_repo_files(repo_id=repo)
            except Exception:
                continue
            for fn in files:
                if fn.startswith("model/") and fn.endswith(".pt"):
                    suggested.append(f"{repo}/{fn}")

        suggested = sorted(suggested)
        default_val = "VIZINTZOR/F5-TTS-THAI/model/model_700000.pt" if not suggested else suggested[-1]

        return {
            "required": {
                "sample_audio": ("AUDIO",),
                "sample_text": ("STRING", {"default": "Text of sample_audio"}),
                # ส่ง full cleaned text เข้า infer_process
                "text": ("STRING", {"multiline": True, "default": "สวัสดีครับ\nทำไมวันนี้อากาศดี"}),
                "model_path": ("STRING", {
                    "default": default_val,
                    "description": (
                        "พิมพ์ <namespace>/<repo_name>/<filename>.pt หรือ "
                        "<namespace>/<repo_name>/model/<filename>.pt\n"
                        "เช่น VIZINTZOR/F5-TTS-THAI/model_650000_FP16.pt\n\n"
                        "suggested:\n" + "\n".join(suggested)
                    )
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
                "max_chars": ("INT", {"default": 120, "min": 1, "max": 2000}),
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
        text,            # ข้อความหลายบรรทัด (มี \n)
        model_path="",
        seed=-1,
        remove_silence=True,
        speed=1.0,
        cross_fade_duration=0.15,
        nfe_step=32,
        cfg_strength=2.0,
        sway_sampling_coef=-1.0,
        fix_duration=0.0,
        max_chars=120,
    ):
        # 1. transliterate + clean (แปลงทั้งข้อความรวมทั้งบล็อก)
        translit_full = eng_to_thai_translit(text)
        cleaned_full = process_thai_repeat(replace_numbers_with_thai(translit_full))

        # 2. เตรียม reference audio (เหมือนเดิม)
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

        # 3. โหลด config ของโมเดล (เหมือนเดิม)
        cfg_dir = os.path.join(Install.base_path, "src", "f5_tts", "configs")
        for fn in ["F5TTS_Base.yaml", "F5TTS_Base_train.yaml"]:
            p = os.path.join(cfg_dir, fn)
            if os.path.exists(p):
                model_cfg = OmegaConf.load(p).model.arch
                break
        else:
            raise FileNotFoundError("Config file not found")

        # 4. แยก model_path → repo_id + relpath
        parts = model_path.strip().split("/")
        if len(parts) < 3:
            raise ValueError(
                "กรุณากรอกเป็น <namespace>/<repo_name>/<filename>.pt "
                "หรือ <namespace>/<repo_name>/model/<filename>.pt"
            )
        repo_id = f"{parts[0]}/{parts[1]}"
        rest = "/".join(parts[2:])
        if rest.endswith(".pt") and not rest.startswith("model/"):
            relpath = f"model/{rest}"
        else:
            relpath = rest
        if not relpath.startswith("model/") or not relpath.endswith(".pt"):
            raise ValueError("ต้องระบุไฟล์จากโฟลเดอร์ 'model/' และลงท้ายด้วย .pt")

        # 5. เลือกที่เก็บโมเดล: ถ้ามี submodule อยู่ ให้ใช้นั่น ถ้าไม่มีให้ fallback
        submod_model_dir = os.path.join(
            Install.base_path, "submodules", "F5TTS-on-Pod", "model", "model"
        )
        if os.path.isdir(submod_model_dir):
            mdir = submod_model_dir
        else:
            mdir = os.path.join(Install.base_path, "model")
        os.makedirs(mdir, exist_ok=True)
        filename = os.path.basename(relpath)
        local_model_path = os.path.join(mdir, filename)

        # 6. ดาวน์โหลดโมเดลถ้ายังไม่มี (พร้อม progress bar)
        if not os.path.exists(local_model_path):
            url = hf_hub_url(repo_id=repo_id, filename=relpath)
            try:
                download_with_progress(url, local_model_path)
            except Exception as e:
                raise RuntimeError(f"❌ ไม่สามารถดาวน์โหลดโมเดลจาก {repo_id}/{relpath} ได้: {e}")

        # 7. เตรียม vocab.txt (similar logic)
        submod_vocab_dir = os.path.join(
            Install.base_path, "submodules", "F5TTS-on-Pod", "vocab"
        )
        if os.path.isdir(submod_vocab_dir):
            vdir = submod_vocab_dir
        else:
            vdir = os.path.join(Install.base_path, "vocab")
        os.makedirs(vdir, exist_ok=True)
        vocab_path = os.path.join(vdir, "vocab.txt")

        if not os.path.exists(vocab_path):
            url_vocab = hf_hub_url(repo_id=repo_id, filename="vocab.txt")
            try:
                download_with_progress(url_vocab, vocab_path)
            except Exception as e:
                raise RuntimeError(f"❌ ไม่สามารถดาวน์โหลด vocab.txt จาก {repo_id} ได้: {e}")

        # 8. โหลดโมเดล + vocoder ลง device (เหมือนเดิม)
        model = load_model(DiT, model_cfg, local_model_path, vocab_file=vocab_path, mel_spec_type="vocos")
        vocoder = load_vocoder("vocos")
        device = comfy.model_management.get_torch_device()
        model.to(device)
        vocoder.to(device)
        if seed >= 0:
            torch.manual_seed(seed)

        # 9. fix_duration
        fd = None if fix_duration == 0.0 else fix_duration

        # 10. เรียก infer_process โดยส่ง cleaned_full (มี newline) ให้ chunk_text ทำงาน
        audio_np_full, sr_out, _ = infer_process(
            ref_audio, ref_text, cleaned_full,
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

        # 11. แปลงเป็น Tensor
        audio_tensor = torch.from_numpy(audio_np_full)
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # 12. ตัด silence ถ้าต้องการ
        if remove_silence:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp2:
                sf.write(tmp2.name, audio_tensor.cpu().numpy().T, sr_out)
                remove_silence_for_generated_wav(tmp2.name)
                audio_tensor, sr_out = torchaudio.load(tmp2.name)
                try:
                    os.unlink(tmp2.name)
                except PermissionError:
                    pass

        # 13. ส่งกลับพร้อมข้อความเต็ม (มี newline เหมือนเดิม)
        return {"waveform": audio_tensor, "sample_rate": sr_out}, cleaned_full