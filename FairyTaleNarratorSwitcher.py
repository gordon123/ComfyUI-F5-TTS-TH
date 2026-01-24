# FairyTaleNarratorSwitcher.py

import re
import json
try:
    import torch

    torch_imported = True
except ImportError:
    torch_imported = False

from .F5TTS_Advance import F5TTS_Advance
from huggingface_hub import HfApi, hf_hub_url
import os
import tempfile
import torchaudio
import soundfile as sf
from omegaconf import OmegaConf
import comfy
from .install import Install
import requests
from tqdm.auto import tqdm

# Lazy install: run installer only when the node is actually used
_installed = False

def _ensure_install():
    global _installed
    if not _installed:
        Install.check_install()
        _installed = True


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
        _ensure_install()
        WATCHED_REPOS = [
            "VIZINTZOR/F5-TTS-THAI",
            "Muscari/F5-TTS-TH_Finetuned",
        ]
        api = HfApi()
        model_choices = []

        # Fetch all .pt filenames under "model/" from each watched repo
        for repo in WATCHED_REPOS:
            try:
                files = api.list_repo_files(repo_id=repo)
            except Exception:
                continue
            for fn in files:
                if fn.startswith("model/") and fn.endswith(".pt"):
                    model_choices.append(f"{repo}/{fn}")

        model_choices = sorted(model_choices)

        # Prefer 1000000 if available
        if "VIZINTZOR/F5-TTS-THAI/model/model_1000000.pt" in model_choices:
            default_choice = "VIZINTZOR/F5-TTS-THAI/model/model_1000000.pt"
        else:
            default_choice = "VIZINTZOR/F5-TTS-THAI/model/model_700000.pt" if "VIZINTZOR/F5-TTS-THAI/model/model_700000.pt" in model_choices else ""
            if default_choice == "" and model_choices:
                default_choice = model_choices[-1]

        description_text = (
            "พิมพ์ <namespace>/<repo_name>/model/<filename>.pt หรือ\n"
            "<namespace>/<repo_name>/<filename>.pt เช่น:\n"
            "  VIZINTZOR/F5-TTS-THAI/model/model_700000.pt\n"
            "  Muscari/F5-TTS-TH_Finetuned/model_62400.pt\n\n"
        )
        if model_choices:
            description_text += (
                "หรือเลือกดูตัวอย่างด้านล่าง (คัดลอกแล้ววางในช่อง text):\n" + "\n".join(model_choices)
            )

        required = {
            "text": ("STRING", {"multiline": True}),
            "sample_audio_narator": ("AUDIO",),
            "sample_text_narator": ("STRING", {"default": ""}),
            "model_path": (
                "STRING",
                {
                    "default": default_choice,
                    "description": description_text,
                },
            ),
            "seed": ("INT", {"default": -1, "min": -1}),
        }

        optional = {}
        for i in range(1, 6):
            optional[f"char_name_{i}"] = ("STRING", {"default": f"Character{i}"})
            optional[f"sample_audio_{i}"] = ("AUDIO",)
            optional[f"sample_text_{i}"] = ("STRING", {"default": ""})

        optional.update(
            {
                "remove_silence": ("BOOL", {"default": True}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "cross_fade_duration": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "nfe_step": ("INT", {"default": 32, "min": 1, "max": 128}),
                "cfg_strength": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "sway_sampling_coef": ("FLOAT", {"default": -1.0, "min": -5.0, "max": 5.0, "step": 0.1}),
                "fix_duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "max_chars": ("INT", {"default": 250, "min": 1, "max": 1000}),
            }
        )

        return {"required": required, "optional": optional}

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "text")
    FUNCTION = "run"
    CATEGORY = "🇹🇭 Thai TTS"

    def run(
        self,
        text: str,
        sample_audio_narator,
        sample_text_narator: str,
        model_path: str = "",
        seed: int = -1,
        *args,
        **kwargs,
    ):
        """
        Process the multi‐line dialogue. Each line optionally has a “[Speaker]Text” format.
        Runs each segment through F5TTS_Advance and concatenates the results.
        Returns:
          - waveform of all segments concatenated
          - JSON metadata (speaker + text) for each line
        """

        def parse_line(line: str) -> tuple[str | None, str]:
            """
            If the line starts with “[Speaker]”, extract that. Otherwise speaker=None.
            """
            if line.startswith("["):
                end = line.find("]")
                if end != -1:
                    spk = line[1:end]
                    utt = line[end + 1 :].strip().strip('“”" ')
                    return spk, utt
            return None, line

        # Build reference dictionary: “narator” + up to 5 character references
        refs = {"narator": (sample_audio_narator, sample_text_narator)}
        for i in range(1, 6):
            name = kwargs.get(f"char_name_{i}", "").strip()
            aud = kwargs.get(f"sample_audio_{i}")
            txt = kwargs.get(f"sample_text_{i}", "")
            if name and aud is not None:
                refs[name] = (aud, txt)

        # Split text into lines, map speaker tags
        segments: list[tuple[str, str]] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            spk, utt = parse_line(line)
            speaker_key = "narator"
            if spk is not None:
                spk_clean = spk.strip().rstrip("：:").strip().lower()
                for key in refs.keys():
                    if key.strip().lower() == spk_clean:
                        speaker_key = key
                        break
            segments.append((speaker_key, utt))

        # Gather F5TTS parameters
        f5_params: dict[str, object] = {}
        for k in [
            "remove_silence",
            "speed",
            "cross_fade_duration",
            "nfe_step",
            "cfg_strength",
            "sway_sampling_coef",
            "fix_duration",
            "max_chars",
        ]:
            if k in kwargs:
                f5_params[k] = kwargs[k]
        f5_params["model_path"] = model_path
        f5_params["seed"] = seed

        # Instantiate F5TTS_Advance
        tts = F5TTS_Advance()
        audio_chunks = []
        sr_out = None
        out_meta: list[dict[str, str]] = []

        # For each (speaker, utterance), run one pass of F5TTS_Advance
        for spk_key, utt in segments:
            ref_aud, ref_txt = refs.get(spk_key, refs["narator"])
            audio_dict, _ = tts.synthesize(ref_aud, ref_txt, utt, **f5_params)
            wav = audio_dict["waveform"]
            sr = audio_dict["sample_rate"]
            sr_out = sr_out or sr
            if torch_imported and wav.dim() == 1:
                wav = wav.unsqueeze(0)
            audio_chunks.append(wav)
            out_meta.append({"speaker": spk_key, "text": utt})

        # Concatenate all chunks into a single tensor
        if torch_imported and len(audio_chunks) > 1:
            full = torch.cat(audio_chunks, dim=1)
        else:
            full = audio_chunks[0]

        return {"waveform": full, "sample_rate": sr_out}, json.dumps(out_meta, ensure_ascii=False)
