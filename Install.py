import subprocess
import sys
import os
import urllib.request

class Install:
    # ชี้ไปที่โฟลเดอร์ submodule ใหม่เสมอ
    base_path = os.path.join(os.path.dirname(__file__), "submodules", "F5TTS-on-Pod")
    model_dir = os.path.join(base_path, "ckpts", "thai")
    vocab_dir = os.path.join(base_path, "vocab")
    default_model = "model_1000000.pt"
    model_url_base = "https://huggingface.co/VIZINTZOR/F5-TTS-THAI/resolve/main"

    @staticmethod
    def has_submodule_file():
        # ตรวจสอบว่า submodule ถูก clone มาแล้ว
        return os.path.exists(os.path.join(Install.base_path, "README.md"))

    @staticmethod
    def check_install():
        Install.ensure_model_dir()
        Install.ensure_vocab_dir()
        if not Install.has_submodule_file():
            Install.install_submodule()
        Install.ensure_vocab()
        Install.ensure_default_model()

    @staticmethod
    def install_submodule():
        print("🔧 Initializing F5TTS-on-Pod submodule...")
        try:
            import pygit2  # type: ignore
            repo = pygit2.Repository(os.path.dirname(__file__))
            pygit2.submodules.SubmoduleCollection(repo).update(init=True)
        except Exception as e:
            print("ℹ️ pygit2 not available or failed:", e)
            print("⚠️ Falling back to `git submodule update` via subprocess")

        subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"],
            cwd=os.path.dirname(__file__),
            check=True
        )

        if not Install.has_submodule_file():
            print("❌ Submodule initialization failed. เช็คการตั้งค่า Git อีกทีนะ.")
        else:
            Install.install_requirements()

    @staticmethod
    def install_requirements():
        print("📦 Installing dependencies for F5TTS-on-Pod...")
        # ติดตั้ง repo เป็น package
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "git+https://github.com/VYNCX/F5-TTS-THAI.git"
        ], check=True)

        # ติดตั้ง PyTorch และ Torchaudio
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "torch==2.1.2+cu126", "torchaudio==2.1.2+cu126",
            "--index-url", "https://download.pytorch.org/whl/cu126"
        ], check=True)

    @staticmethod
    def ensure_model_dir():
        os.makedirs(Install.model_dir, exist_ok=True)

    @staticmethod
    def ensure_vocab_dir():
        os.makedirs(Install.vocab_dir, exist_ok=True)

    @staticmethod
    def ensure_vocab():
        vocab_src = os.path.join(Install.vocab_dir, "vocab.txt")
        if not os.path.exists(vocab_src):
            print("⬇️ Downloading vocab.txt...")
            urllib.request.urlretrieve(
                f"{Install.model_url_base}/vocab.txt",
                vocab_src
            )
            print("✅ vocab.txt downloaded.")

    @staticmethod
    def ensure_default_model():
        model_path = os.path.join(Install.model_dir, Install.default_model)
        if not os.path.exists(model_path):
            print(f"⬇️ Downloading default model: {Install.default_model}...")
            urllib.request.urlretrieve(
                f"{Install.model_url_base}/{Install.default_model}",
                model_path
            )
            print(f"✅ Model downloaded: {Install.default_model}")

    @staticmethod
    def has_model_file():
        model_path = os.path.join(Install.model_dir, Install.default_model)
        return os.path.exists(model_path)
