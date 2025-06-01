import re, json
try:
    import torch
    torch_imported = True
except ImportError:
    torch_imported = False

from .F5TTS_Advance import F5TTS_Advance
from huggingface_hub import HfApi, hf_hub_url
import os
import tempfile
import torch
import torchaudio
import soundfile as sf
from omegaconf import OmegaConf
import comfy
from .Install import Install
import requests
from tqdm.auto import tqdm

# Ensure the submodule is initialized
Install.check_install()

def download_with_progress(url: str, local_path: str):
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


class FairyTaleNarratorSwitcher:
    @classmethod
    def INPUT_TYPES(cls):
        # รายการรีโปที่เราจะเฝ้าดู (เหมือน F5TTS_Advance)
        WATCHED_REPOS = [
            "VIZINTZOR/F5-TTS-THAI",
            "Muscari/F5-TTS-TH_Finetuned",
        ]
        api = HfApi()
        model_choices = []

        # ดึงชื่อไฟล์ .pt จาก "model/" ทุกรีโปใน WATCHED_REPOS
        for repo in WATCHED_REPOS:
            try:
                files = api.list_repo_files(repo_id=repo)
            except Exception:
                continue
            for fn in files:
                if fn.startswith("model/") and fn.endswith(".pt"):
                    model_choices.append(f"{repo}/{fn}")

        model_choices = sorted(model_choices)
        default_choice = model_choices[-1] if model_choices else ""

        # สร้าง dropdown + free-form ให้กรอก <repo_id>/model/<filename>.pt
        required = {
            "text": ("STRING", {"multiline": True}),
            "sample_audio_narator": ("AUDIO",),
            "sample_text_narator": ("STRING", {"default": ""}),
            "model_path": (model_choices, {
                "default": default_choice,
                "description": (
                    "พิมพ์ <namespace>/<repo_name>/model/<filename>.pt เช่น\n"
                    "VIZINTZOR/F5-TTS-THAI/model/model_650000.pt\n\n"
                    "หรือเลือกจากรายการ suggested ด้านล่าง:\n" +
                    "\n".join(model_choices)
                )
            }),
            "seed": ("INT", {"default": -1, "min": -1}),
        }
        optional = {}
        for i in range(1, 6):
            optional[f"char_name_{i}"]   = ("STRING", {"default": f"Character{i}"})
            optional[f"sample_audio_{i}"] = ("AUDIO",)
            optional[f"sample_text_{i}"]  = ("STRING", {"default": ""})
        optional.update({
            "remove_silence":      ("BOOL",  {"default": True}),
            "speed":               ("FLOAT", {"default": 1.0,  "min": 0.1, "max": 5.0,  "step": 0.1}),
            "cross_fade_duration": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0,  "step": 0.01}),
            "nfe_step":            ("INT",   {"default": 32,   "min": 1,   "max": 128}),
            "cfg_strength":        ("FLOAT", {"default": 2.0,  "min": 0.0, "max": 10.0, "step": 0.1}),
            "sway_sampling_coef":  ("FLOAT", {"default": -1.0, "min": -5.0,"max": 5.0,  "step": 0.1}),
            "fix_duration":        ("FLOAT", {"default": 0.0,  "min": 0.0, "max": 30.0, "step": 0.1}),
            "max_chars":           ("INT",   {"default": 250,  "min": 1,   "max": 1000})
        })
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "text")
    FUNCTION = "run"
    CATEGORY = "🇹🇭 Thai TTS"

    def run(self, text, sample_audio_narator, sample_text_narator,
            model_path="", seed=-1, *args, **kwargs):

        def parse_line(line):
            if line.startswith('['):
                end = line.find(']')
                if end != -1:
                    spk = line[1:end]
                    utt = line[end+1:].strip().strip('“”" ')
                    return spk, utt
            return None, line

        # Build reference dictionary
        refs = {"narator": (sample_audio_narator, sample_text_narator)}
        for i in range(1,6):
            name = kwargs.get(f"char_name_{i}", "").strip()
            aud  = kwargs.get(f"sample_audio_{i}")
            txt  = kwargs.get(f"sample_text_{i}", "")
            if name and aud is not None:
                refs[name] = (aud, txt)

        # Split text into segments per line and map speaker tags
        segments = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            spk, utt = parse_line(line)
            speaker_key = "narator"
            if spk is not None:
                spk_clean = spk.strip().rstrip('：:').strip()
                for key in refs.keys():
                    if key.strip().lower() == spk_clean.lower():
                        speaker_key = key
                        break
            segments.append((speaker_key, utt))

        # เตรียมพารามิเตอร์สำหรับ F5TTS
        f5_params = {}
        for k in ["remove_silence", "speed", "cross_fade_duration",
                  "nfe_step", "cfg_strength", "sway_sampling_coef",
                  "fix_duration", "max_chars"]:
            if k in kwargs:
                f5_params[k] = kwargs[k]
        f5_params["model_path"] = model_path
        f5_params["seed"]       = seed

        # สร้างเสียงทีละเซกเมนต์
        tts = F5TTS_Advance()
        audio_chunks = []
        sr_out = None
        out_meta = []
        for spk_key, utt in segments:
            ref_aud, ref_txt = refs.get(spk_key, refs["narator"])
            audio_dict, _ = tts.synthesize(ref_aud, ref_txt, utt, **f5_params)
            wav  = audio_dict["waveform"]
            sr   = audio_dict["sample_rate"]
            sr_out = sr_out or sr
            if torch_imported and wav.dim() == 1:
                wav = wav.unsqueeze(0)
            audio_chunks.append(wav)
            out_meta.append({"speaker": spk_key, "text": utt})

        # ถ้ามีหลาย chunk ให้ concat (ถ้า chunk เดียวก็ return เดิม)
        if torch_imported and len(audio_chunks) > 1:
            full = torch.cat(audio_chunks, dim=1)
        else:
            full = audio_chunks[0]

        return ({"waveform": full, "sample_rate": sr_out},
                json.dumps(out_meta, ensure_ascii=False))
