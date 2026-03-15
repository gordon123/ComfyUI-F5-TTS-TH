# ComfyUI-F5-TTS-TH 🇹🇭
Thai Text-to-Speech (TTS) custom node for **ComfyUI**, powered by **F5-TTS (Thai)**

## Demo Video วิธี Install และ แนะนำการใช้คร่าวๆ

[![Demo Video](https://img.youtube.com/vi/Hpa2dnaRCmA/hqdefault.jpg)](https://www.youtube.com/watch?v=Hpa2dnaRCmA)

▶️ https://www.youtube.com/watch?v=Hpa2dnaRCmA


---

## ✨ Features
- Thai TTS คุณภาพสูง
- Auto-download model + vocab
- ใช้งานผ่าน ComfyUI node ได้ทันที
- รองรับ ComfyUI-Manager
- มี fallback อัตโนมัติ ปลอดภัยต่อ environment

---
### วิธีการ ดาวโหลด โมเดล
ให้ไปที่ https://huggingface.co/VIZINTZOR/F5-TTS-THAI/tree/main หรือ https://huggingface.co/VIZINTZOR/F5-TTS-THAI/tree/main/model <br>

สำหรับ ช่อง model_path นี้ ให้ ใส่ <br>

```VIZINTZOR/F5-TTS-THAI/model/model_1000000.pt ``` 

หรือกรณีที่ custom node ไม่ดาวโหลดโมเดลให้อัตโนมัติ ให้เราใช้ wget ดาวโหลด ไฟล์โมเดลไปไว้ในนี้ <br>
```
cd /workspace/ComfyUI/custom_nodes/ComfyUI-F5-TTS-TH/submodules/F5TTS-on-Pod/model/
```
---

## 🚀 Quick Start (แนะนำที่สุด)

### วิธีที่ 1: ติดตั้งผ่าน ComfyUI-Manager (ง่ายสุด)
1. เปิด ComfyUI
2. ไปที่ **Manager → Custom Nodes**
3. ค้นหา `ComfyUI-F5-TTS-TH`
4. กด **Install**
5. Restart ComfyUI

ระบบจะดาวน์โหลดทุกอย่างให้อัตโนมัติ

---

### วิธีที่ 2: ติดตั้งแบบ Manual
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/gordon123/ComfyUI-F5-TTS-TH.git
cd ComfyUI-F5-TTS-TH

# ⚠️ สำคัญมาก: ต้อง init submodule
git submodule update --init --recursive

apt-get update
apt-get install -y gfortran libopenblas-dev liblapack-dev ffmpeg

# ติดตั้ง dependency
pip install -r requirements.txt
```

```
ถ้าเจอ compile error ตอนลง requirement ให้ลอง พิมพ์ แล้ว install requirement ใหม่
apt-get update && sudo apt-get install -y gfortran pkg-config libopenblas-dev liblapack-dev

ถ้าเจอ error opencv-python-headless 4.13.0.92 requires numpy>=2; ให้ ลด เวอชั่นลง
pip install "opencv-python-headless<4.10"

```
จากนั้น restart ComfyUI

---

## 📦 โครงสร้างไฟล์ (อัตโนมัติ)
```
ComfyUI-F5-TTS-TH/
├─ install.py
├─ submodules/
│  └─ F5TTS-on-Pod/
│     ├─ ckpts/thai/model_1000000.pt
│     ├─ vocab/vocab.txt
│     └─ src/f5_tts/...
```

---

## 🧠 วิธีใช้งาน
- เปิด ComfyUI
- คลิกขวา → Search node
- เลือกหมวด **F5-TTS / Thai TTS**
- ใส่ข้อความภาษาไทย → Generate

---

## ⚡ Optional: เร่งความเร็วโหลดโมเดล
```bash
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
```

---

## 🧩 Compatibility
- Python 3.10–3.12
- CUDA 11.8 – 12.6
- RTX 20xx–40xx
- RunPod / Docker

---

## 📜 License
MIT

---

## ❤️ Credits
F5-TTS / Hugging Face / ComfyUI Community

https://huggingface.co/VIZINTZOR/F5-TTS-THAI

https://github.com/niknah/ComfyUI-F5-TTS

https://github.com/VYNCX/F5-TTS-THAI

https://github.com/VrchStudio/comfyui-web-viewer
