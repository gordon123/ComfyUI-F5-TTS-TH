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
    โค้ดนี้ปรับให้:
    1. ผู้ใช้พิมพ์ <repo_id>/model/<filename>.pt หรือ <repo_id>/<filename>.pt ได้เอง
    2. เมื่อดาวน์โหลดโมเดล จะพยายามดาวน์โหลด vocab.txt เสมอจาก repo_id นั้น
       ซึ่งแก้ปัญหาขนาด embedding mismatch ของโมเดล fine-tuned
    """

    WATCHED_REPOS = [
        "VIZINTZOR/F5-TTS-THAI",
        "Muscari/F5-TTS-TH_Finetuned",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        api = HfApi()
        model_suggestions = []

        # ดึงชื่อไฟล์ .pt จาก "model/" ในทุกรีโปที่กำหนด
        for repo in cls.WATCHED_REPOS:
            try:
                files = api.list_repo_files(repo_id=repo)
            except Exception:
                continue

            for fn in files:
                if fn.startswith("model/") and fn.endswith(".pt"):
                    model_suggestions.append(f"{repo}/{fn}")

        model_suggestions = sorted(model_suggestions)
        default_choice = model_suggestions[-1] if model_suggestions else ""

        description_text = (
            "พิมพ์ <namespace>/<repo_name>/model/<filename>.pt หรือ\n"
            "<namespace>/<repo_name>/<filename>.pt เช่น:\n"
            "VIZINTZOR/F5-TTS-THAI/model/model_700000.pt\n"
            "หรือ Muscari/F5-TTS-TH_Finetuned/model_62400.pt\n\n"
        )
        if model_suggestions:
            description_text += "หรือเลือกใช้ตัวอย่างด้านล่าง:\n" + "\n".join(model_suggestions)

        return {
            "required": {
                "sample_audio": ("AUDIO",),
                "sample_text": ("STRING", {"default": "Text of sample_audio"}),
                "text": ("STRING", {"multiline": True, "default": "สวัสดีครับ"}),
                "model_path": ("STRING", {
                    "default": default_choice,
                    "description": description_text
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
                "max_chars": ("INT", {"default": 250, "min": 1, "max": 2000}),
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
        model_path="",  # รูปแบบ <repo_id>/model/<filename>.pt หรือ <repo_id>/<filename>.pt
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
        # 1. Transliterate + Clean
        translit = eng_to_thai_translit(text)
        cleaned = process_thai_repeat(replace_numbers_with_thai(translit))

        # 2. Prepare reference audio
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

        # 3. Load model config
        cfg_dir = os.path.join(Install.base_path, "src", "f5_tts", "configs")
        for fn in ["F5TTS_Base.yaml", "F5TTS_Base_train.yaml"]:
            p = os.path.join(cfg_dir, fn)
            if os.path.exists(p):
                model_cfg = OmegaConf.load(p).model.arch
                break
        else:
            raise FileNotFoundError("Config file not found")

        # 4. Parse model_path → repo_id + filename
        parts = model_path.strip().split("/")
        if len(parts) < 2:
            raise ValueError(
                "กรุณากรอกเป็น <namespace>/<repo_name>/model/<filename>.pt "
                "หรือ <namespace>/<repo_name>/<filename>.pt"
            )
        repo_id = f"{parts[0]}/{parts[1]}"
        rest = "/".join(parts[2:])  # อาจเป็น "model_62400.pt" หรือ "model/model_62400.pt"
        filename = os.path.basename(rest)

        # 5. เลือกโฟลเดอร์เก็บโมเดล: submodule หรือ fallback
        submod_model_dir = os.path.join(
            Install.base_path, "submodules", "F5TTS-on-Pod", "model", "model"
        )
        if os.path.isdir(submod_model_dir):
            mdir = submod_model_dir
        else:
            mdir = os.path.join(Install.base_path, "model")
        os.makedirs(mdir, exist_ok=True)
        local_model_path = os.path.join(mdir, filename)

        # 6. ดาวน์โหลดโมเดลถ้ายังไม่มี (ลอง 2 ตำแหน่งก่อน)
        if not os.path.exists(local_model_path):
            # A) ลองจาก “model/<filename>.pt”
            relA = f"model/{filename}"
            urlA = hf_hub_url(repo_id=repo_id, filename=relA)
            try:
                download_with_progress(urlA, local_model_path)
            except Exception:
                # B) ถ้า A ล้มเหลว ให้ลองจาก root “<filename>.pt”
                relB = filename
                urlB = hf_hub_url(repo_id=repo_id, filename=relB)
                try:
                    download_with_progress(urlB, local_model_path)
                except Exception as e:
                    raise RuntimeError(f"❌ ไม่สามารถดาวน์โหลดโมเดลจาก {repo_id}/{filename} ได้: {e}")

        # 7. ดาวน์โหลด vocab.txt จาก repo_id เสมอ (ไม่สนว่ามีไฟล์เดิมหรือไม่)
        #    เก็บชื่อไฟล์เป็น <namespace>_<repo_name>_vocab.txt เพื่อแยกแต่ละ repo
        escaped = repo_id.replace("/", "_")
        vdir = os.path.join(Install.base_path, "vocab")
        os.makedirs(vdir, exist_ok=True)
        vocab_filename = f"{escaped}_vocab.txt"
        vocab_path = os.path.join(vdir, vocab_filename)

        # **Always** ดาวน์โหลด vocab.txt จาก repo_id
        try:
            url_vocab = hf_hub_url(repo_id=repo_id, filename="vocab.txt")
            download_with_progress(url_vocab, vocab_path)
        except Exception as e:
            raise RuntimeError(f"❌ ไม่สามารถดาวน์โหลด vocab.txt จาก {repo_id} ได้: {e}")

        # 8. Load model + vocoder
        model = load_model(DiT, model_cfg, local_model_path, vocab_file=vocab_path, mel_spec_type="vocos")
        vocoder = load_vocoder("vocos")
        device = comfy.model_management.get_torch_device()
        model.to(device)
        vocoder.to(device)
        if seed >= 0:
            torch.manual_seed(seed)

        # 9. fix_duration
        fd = None if fix_duration == 0.0 else fix_duration

        # 10. Inference (ส่ง cleaned เข้า chunk_text ภายใน infer_process)
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

        # 11. แปลงเป็น Tensor
        audio_tensor = torch.from_numpy(audio_np)
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

        return {"waveform": audio_tensor, "sample_rate": sr_out}, cleaned
