import re
import json
torch_imported = False
try:
    import torch
    torch_imported = True
except ImportError:
    pass

from .F5TTS_Advance import F5TTS_Advance

class FairyTaleNarratorSwitcher:
    """
    🧚 FairyTale Narrator Switcher for ComfyUI with integrated F5-TTS
    - Inputs:
        • text                : STRING (multiline) — fairy tale script
        • sample_audio_narator: AUDIO — reference audio for narrator voice
        • sample_text_narator : STRING — textual prompt for narrator reference
        • model_name          : MODEL choice list
        • seed                : INT (default -1)
        • Optional char_i pairs and TTS params…
    - Outputs:
        • audio              : AUDIO — concatenated waveform
        • text               : STRING — JSON list of segments {speaker, text}
    """

    @classmethod
    def INPUT_TYPES(cls):
        # mirror F5TTS_Advance model choices
        model_choices = [
            "model_100000.pt", "model_130000.pt", "model_150000.pt",
            "model_200000.pt", "model_250000.pt", "model_350000.pt",
            "model_430000.pt", "model_475000.pt", "model_500000.pt"
        ]
        required = {
            "text": ("STRING", {"multiline": True, "default": ""}),
            "sample_audio_narator": ("AUDIO",),
            "sample_text_narator": ("STRING", {"default": ""}),
            # TTS model selection
            "model_name": (model_choices, {"default": "model_500000.pt"}),
            "seed": ("INT", {"default": -1, "min": -1}),
        }
        optional = {}
        # up to 5 character voices
        for i in range(1, 6):
            optional[f"char_name_{i}"]   = ("STRING", {"default": f"Character{i}"})
            optional[f"sample_audio_{i}"] = ("AUDIO",)
            optional[f"sample_text_{i}"]  = ("STRING", {"default": ""})
        # TTS numeric/bool params
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

    def run(
        self,
        text,
        sample_audio_narator,
        sample_text_narator,
        model_name="model_500000.pt",
        seed=-1,
        *args,
        **kwargs
    ):
        # strip tags helper
        def strip_tags(line): return re.sub(r'^\[[^\]]+\]\s*', '', line)

        # build speaker refs
        refs = {"narator": (sample_audio_narator, sample_text_narator)}
        for i in range(1, 6):
            raw = kwargs.get(f"char_name_{i}")
            aud = kwargs.get(f"sample_audio_{i}")
            txt = kwargs.get(f"sample_text_{i}")
            if raw and aud is not None:
                name = raw.strip().strip("[]").strip()
                refs[name] = (aud, txt or "")

        # parse script
        pattern = re.compile(r'^\[([^\]]+)\]\s*“(.+)”')
        segments = []
        for line in text.splitlines():
            line = line.strip()
            if not line: continue
            m = pattern.match(line)
            if m:
                spk, utt = m.groups()
                segments.append((spk if spk in refs else "narator", utt))
            else:
                segments.append(("narator", line))

        # collect TTS params
        f5_params = {k: kwargs[k] for k in [
            "remove_silence", "speed", "cross_fade_duration",
            "nfe_step", "cfg_strength", "sway_sampling_coef",
            "fix_duration", "max_chars"
        ] if k in kwargs}
        # include model & seed
        f5_params["model_name"] = model_name
        f5_params["seed"] = seed

        # synthesize each segment
        tts = F5TTS_Advance()
        audio_tensors, out_segments, sample_rate = [], [], None
        for spk, utt in segments:
            ref_aud, ref_txt = refs.get(spk, refs["narator"])
            audio_dict, _ = tts.synthesize(ref_aud, ref_txt, utt, **f5_params)
            wav, sr = audio_dict["waveform"], audio_dict["sample_rate"]
            sample_rate = sample_rate or sr
            if wav.dim() == 1: wav = wav.unsqueeze(0)
            audio_tensors.append(wav)
            out_segments.append({"speaker": spk, "text": utt})

        # concat
        full = torch.cat(audio_tensors, dim=1) if torch_imported else audio_tensors[0]
        return ({"waveform": full, "sample_rate": sample_rate},
                json.dumps(out_segments, ensure_ascii=False))
