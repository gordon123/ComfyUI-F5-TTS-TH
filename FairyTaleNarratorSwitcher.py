import re, json
try:
    import torch
    torch_imported = True
except ImportError:
    torch_imported = False

from .F5TTS_Advance import F5TTS_Advance

class FairyTaleNarratorSwitcher:
    @classmethod
    def INPUT_TYPES(cls):
        model_choices = [
            "model_250000.pt", "model_250000_FP16.pt",
            "model_475000.pt", "model_475000_FP16.pt",
            "model_500000.pt", "model_500000_FP16.pt",
            "model_600000.pt", "model_600000_FP16.pt",
            "model_650000.pt", "model_650000_FP16.pt"
        ]
        required = {
            "text": ("STRING", {"multiline": True}),
            "sample_audio_narator": ("AUDIO",),
            "sample_text_narator": ("STRING", {"default": ""}),
            "model_name": (model_choices, {"default": "model_650000.pt"}),
            "seed": ("INT", {"default": -1, "min": -1}),
        }
        optional = {}
        for i in range(1,6):
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
            model_name="model_650000.pt", seed=-1, *args, **kwargs):

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

        # Split text into segments and map speakers robustly
        segments = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            spk, utt = parse_line(line)
            # Default to narrator
            speaker_key = "narator"
            if spk is not None:
                # Clean tag, remove trailing punctuation and spaces
                spk_clean = spk.strip().rstrip('：:').strip()
                # Case-insensitive name match
                for key in refs.keys():
                    if key.strip().lower() == spk_clean.lower():
                        speaker_key = key
                        break
            segments.append((speaker_key, utt))

        # Prepare parameters for F5TTS
        f5_params = {k: kwargs[k] for k in [
            "remove_silence", "speed", "cross_fade_duration",
            "nfe_step", "cfg_strength", "sway_sampling_coef",
            "fix_duration", "max_chars"
        ] if k in kwargs}
        f5_params["model_name"] = model_name
        f5_params["seed"]       = seed

        # Synthesize each segment
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

        # Concatenate or return single
        if torch_imported and len(audio_chunks) > 1:
            full = torch.cat(audio_chunks, dim=1)
        else:
            full = audio_chunks[0]

        return ({"waveform": full, "sample_rate": sr_out},
                json.dumps(out_meta, ensure_ascii=False))