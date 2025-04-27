import re
torch_imported = False
try:
    import torch
    torch_imported = True
except ImportError:
    pass

from .F5TTS_Advance import F5TTS_Advance
import json

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
        • TTS params same as F5TTS_Advance
    - Outputs:
        • audio              : AUDIO — concatenated waveform of full story
        • text               : STRING — JSON list of segments {speaker, text}
    """

    @classmethod
    def INPUT_TYPES(cls):
        # base required
        required = {
            "text": ("STRING", {"multiline": True, "default": ""}),
            "sample_audio_narator": ("AUDIO",),
            "sample_text_narator": ("STRING", {"default": ""}),
        }
        # optional character references up to 5
        optional = {}
        for i in range(1, 6):
            optional[f"char_name_{i}"] = ("STRING", {"default": f"Character{i}"})
            optional[f"sample_audio_{i}"] = ("AUDIO",)
            optional[f"sample_text_{i}"] = ("STRING", {"default": ""})
        # include F5TTS params
        f5_opts = F5TTS_Advance.INPUT_TYPES()["required"]
        for k, v in F5TTS_Advance.INPUT_TYPES()["optional"].items():
            f5_opts[k] = v
        return {"required": required, "optional": optional | f5_opts}

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
        # build voice reference map: speaker -> (audio, text)
        refs = {"narator": (sample_audio_narator, sample_text_narator)}
        # collect char refs
        for i in range(1, 6):
            name = kwargs.get(f"char_name_{i}")
            audio = kwargs.get(f"sample_audio_{i}")
            txt = kwargs.get(f"sample_text_{i}")
            if name and audio is not None:
                refs[name] = (audio, txt or "")
        # parse script into segments
        pattern = re.compile(r'^\[([^\]]+)\]\s*“(.+)”')
        segments = []
        for line in text.splitlines():
            line = line.strip()
            if not line: continue
            m = pattern.match(line)
            if m:
                spk, utt = m.groups()
                if spk in refs:
                    segments.append((spk, utt))
                else:
                    segments.append(("narator", line))
            else:
                segments.append(("narator", line))
        # synthesize each with F5TTS
        tts = F5TTS_Advance()
        audio_tensors = []
        sample_rate = None
        out_segments = []
        # extract F5 params
        f5_params = {}
        for key in F5TTS_Advance.INPUT_TYPES()["optional"]:
            if key in kwargs:
                f5_params[key] = kwargs[key]
        for spk, utt in segments:
            ref_audio, ref_txt = refs.get(spk, refs["narator"])
            # call TTS
            audio_dict, _ = tts.synthesize(
                ref_audio,
                ref_txt,
                utt,
                **f5_params
            )
            wav = audio_dict["waveform"]
            sr = audio_dict["sample_rate"]
            if sample_rate is None:
                sample_rate = sr
            # ensure shape
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            audio_tensors.append(wav)
            out_segments.append({"speaker": spk, "text": utt})
        # concatenate
        if torch_imported:
            full = torch.cat(audio_tensors, dim=1)
        else:
            full = audio_tensors[0]
        return ({"waveform": full, "sample_rate": sample_rate}, json.dumps(out_segments, ensure_ascii=False))