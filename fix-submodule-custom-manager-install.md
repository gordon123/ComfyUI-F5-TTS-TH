⚠️ IMPORTANT

When installing via ComfyUI-Manager / Registry: <br>
Some environments (including RunPod and container-based setups)<br>
do NOT fully support git submodules during installation.<br>

If you see errors such as:<br>
- Submodule initialization failed<br>
- ModuleNotFoundError: No module named 'f5_tts'<br>

This is expected behavior and can be fixed manually.<br>

## 🛠 Manual Fix (Recommended if Manager install fails)

### 1. Go to ComfyUI directory
```
cd /workspace/ComfyUI/custom_nodes/ComfyUI-F5-TTS-TH
```

test git version
```git --version```

### 2. Initialize submodule
```
git config --global --add safe.directory /workspace/ComfyUI
git submodule update --init --recursive
```

### 3. install submodule
```
pip install -e submodules/F5TTS-on-Pod
```

### Restart ComfyUI


