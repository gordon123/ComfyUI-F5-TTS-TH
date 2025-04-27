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
        • Optional char_i pairs:
            - char_name_i      : STRING — tag name matching script (e.g. มะลิ)
            - sample_audio_i   : AUDIO — reference for TTS style
            - sample_text_i    : STRING — textual prompt for reference audio
        • TTS numeric/bool params only:
            - remove_silence, speed, cross_fade_duration, nfe_step,
              cfg_strength, sway_sampling_coef, fix_duration, max_chars
    - Outputs:
        • audio              : AUDIO — concatenated waveform of full story
        • text               : STRING — JSON list of segments {speaker, text}
    """

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "text":                  ("STRING", {"multiline": True, "default": ""}),
            "sample_audio_narator":  ("AUDIO",),
            "sample_text_narator":   ("STRING", {"default": ""}),
        }
        optional = {}

        # character references up to 5
        for i in range(1, 6):
            optional[f"char_name_{i}"]   = ("STRING", {"default": f"Character{i}"})
            optional[f"sample_audio_{i}"] = ("AUDIO",)
            optional[f"sample_text_{i}"]  = ("STRING", {"default": ""})

        # TTS numeric/bool params (no MODEL ports)
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
    CATEGORY = "🧚 FairyTale Tools"

    def run(
        self,
        text,
        sample_audio_narator,
        sample_text_narator,
        *args,
        **kwargs
    ):
        # helper to strip tags from fallback lines
        def strip_tags(line):
            return re.sub(r'^\[[^\]]+\]\s*', '', line)

        # build voice reference map: speaker -> (audio, text)
        refs = {"narator": (sample_audio_narator, sample_text_narator)}
        for i in range(1, 6):
            name = kwargs.get(f"char_name_{i}")
            audio = kwargs.get(f"sample_audio_{i}")
            txt   = kwargs.get(f"sample_text_{i}")
            if name and audio is not None:
                refs[name] = (audio, txt or "")

        # parse script into segments
        pattern = re.compile(r'^\[([^\]]+)\]\s*“(.+)”')
        segments = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            m = pattern.match(line)
            if m:
                spk, utt = m.groups()
                if spk in refs:
                    segments.append((spk, utt))
                else:
                    # unrecognized tag: fallback narrator, strip tag
                    segments.append(("narator", strip_tags(line)))
            else:
                segments.append(("narator", line))

        # gather TTS params
        f5_params = {
            k: kwargs[k] for k in [
                "remove_silence", "speed", "cross_fade_duration",
                "nfe_step", "cfg_strength", "sway_sampling_coef",
                "fix_duration", "max_chars"
            ] if k in kwargs
        }

        # synthesize segments
        tts = F5TTS_Advance()
        audio_tensors = []
        sample_rate = None
        out_segments = []

        for spk, utt in segments:
            ref_audio, ref_txt = refs.get(spk, refs["narator"])
            audio_dict, _ = tts.synthesize(
                ref_audio, ref_txt, utt, **f5_params
            )
            wav = audio_dict["waveform"]
            sr  = audio_dict["sample_rate"]
            sample_rate = sample_rate or sr
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            audio_tensors.append(wav)
            out_segments.append({"speaker": spk, "text": utt})

        # concatenate all audio
        if torch_imported:
            full = torch.cat(audio_tensors, dim=1)
        else:
            full = audio_tensors[0]

        return ({"waveform": full, "sample_rate": sample_rate},
                json.dumps(out_segments, ensure_ascii=False))
