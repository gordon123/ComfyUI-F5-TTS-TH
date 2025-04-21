# 🛠️ ComfyUI-F5-TTS-TH 🇹🇭  
**Custom Node สำหรับแปลงข้อความเป็นเสียงภาษาไทย บน ComfyUI**  

---

🚧 **Underconstruction — SIÒON!** 🚧  
_เสียงไทยที่เท่จนต้องเบิ้ลหูฟัง_  
<br>  
<img src="https://github.com/user-attachments/assets/4f630f41-42ba-42fe-a224-7dd4e9bd1b5a" alt="underconstruction" width="200" />  
<br>  

## 🎤 เกี่ยวกับโปรเจกต์นี้

`ComfyUI-F5-TTS-TH` คือ Node เสริมสำหรับ [ComfyUI](https://github.com/comfyanonymous/ComfyUI)  
ที่เปิดโอกาสให้คุณสามารถแปลงข้อความเป็นเสียง **ภาษาไทย** ได้แบบง่าย ๆ  
รองรับเสียงอ้างอิง + ปรับแต่งข้อความได้เอง — เพื่อการสร้างเสียงที่เป็นเอกลักษณ์ของคุณเอง 🎧

🧠 พลังมาจากโมเดลภาษาไทย:  
- 🧬 [F5-TTS (Original Repo)](https://github.com/SWivid/F5-TTS)  
- 🇹🇭 [F5-TTS-THAI (Thai Model)](https://github.com/VYNCX/F5-TTS-THAI)  
- 🤗 [Model on HuggingFace](https://huggingface.co/VIZINTZOR/F5-TTS-THAI)  

🔁 พัฒนาต่อยอดมาจาก:  
- 💡 [niknah/ComfyUI-F5-TTS](https://github.com/niknah/ComfyUI-F5-TTS) – **ขอบคุณสำหรับโครงสร้างและแรงบันดาลใจ!**

---

## 🚀 วิธีติดตั้ง

### 1. Clone Node
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/gordon123/ComfyUI-F5-TTS-TH.git
```

# 🎤 ComfyUI-F5-TTS-TH 🇹🇭

โมดูล Custom Node สำหรับ ComfyUI ที่ให้คุณสามารถใช้โมเดล F5-TTS-THAI ทำ Text-to-Speech (TTS) ภาษาไทย ได้โดยตรง 🎶  
ใช้โมเดลจาก [F5-TTS-THAI (VYNCX)](https://github.com/VYNCX/F5-TTS-THAI) ซึ่งถูกฝึกมาสำหรับเสียงภาษาไทยโดยเฉพาะ  

---

## 🚀 วิธีติดตั้ง (Installation)

### 🔧 1. Clone repository และติดตั้ง F5-TTS-THAI (แบบ submodule)

```
git clone https://github.com/yourname/ComfyUI-F5-TTS-TH.git
cd ComfyUI-F5-TTS-TH
git submodule update --init --recursive
```

# 1. สร้าง virtual env 
```
python -m venv venv
source venv/bin/activate  # หรือ call venv/Scripts/activate บน Windows
```
# 2. อัปเกรด pip
```
pip install --upgrade pip
```
# 3. ติดตั้งทุก dependencies 
```
pip install -r requirements.txt
```
# 4. ติดตั้ง ffmpeg
```
apt update && apt install -y ffmpeg
```
## 🔧 Custom Node ที่แนะนำให้ติดตั้งเพิ่มเติม

| ชื่อ Node | ใช้ทำอะไร | ลิงก์ |
|-----------|------------|-------|
| **rgthree-comfy** | ระบบ UI Manager + ฟีเจอร์จัด Node เป็นกลุ่ม, ตัวเลือก dropdown, dynamic inputs | [github.com/rgthree/rgthree-comfy](https://github.com/rgthree/rgthree-comfy) |
| **ComfyUI Web Viewer** | ใช้ `VrchAudioSaverNode` สำหรับบันทึกเสียงพร้อม metadata ถูกต้อง ไม่เจอ codec error | [github.com/VrchStudio/comfyui-web-viewer](https://github.com/VrchStudio/comfyui-web-viewer) |
| **ComfyUI Whisper** | แปลงเสียงเป็นข้อความ (speech-to-text) รองรับภาษาไทย | [github.com/ltdrdata/ComfyUI-Whisper](https://github.com/ltdrdata/ComfyUI-Whisper) |

---


