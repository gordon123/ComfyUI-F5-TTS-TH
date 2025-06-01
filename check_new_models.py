import json
import os
from datetime import datetime
from huggingface_hub import HfApi

WATCHED_REPOS = [
    "VIZINTZOR/F5-TTS-THAI",
    "Muscari/F5-TTS-TH_Finetuned",
]
HISTORY_FILE = "model_history.json"

def check_new_models():
    api = HfApi()

    # โหลดรายชื่อเก่า (ถ้ายังไม่มี ให้ตั้งเป็น list ว่าง)
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            old_list = json.load(f)
    else:
        old_list = []

    new_list = []
    new_models = []

    # ดึงชื่อไฟล์ .pt จากทุกรีโปที่เราสนใจ
    for repo in WATCHED_REPOS:
        try:
            files = api.list_repo_files(repo_id=repo)
        except Exception as e:
            print(f"[{datetime.now()}] ❌ เช็กรีโป {repo} ไม่ได้: {e}")
            continue

        for filename in files:
            if filename.endswith(".pt"):
                full_name = f"{repo}/{filename}"
                new_list.append(full_name)
                if full_name not in old_list:
                    new_models.append(full_name)

    # ถ้ามีโมเดลใหม่ ให้พิมพ์ชื่อออกมา
    if new_models:
        print(f"[{datetime.now()}] 🔔 พบโมเดลใหม่:")
        for m in new_models:
            print(f"    • {m}")
    else:
        print(f"[{datetime.now()}] ไม่มีโมเดลใหม่ 💤")

    # บันทึกรายชื่อปัจจุบันลงไฟล์ (สร้างใหม่ถ้ายังไม่มี)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(new_list), f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    check_new_models()
