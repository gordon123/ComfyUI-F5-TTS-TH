import subprocess, sys, os
import urllib.request

class Install:
    f5TTSPath = os.path.join(os.path.dirname(__file__), "F5-TTS-THAI")
    model_dir = os.path.join(f5TTSPath, "ckpts", "thai")
    default_model = "model_500000_FP16.pt"
    model_url_base = "https://huggingface.co/VIZINTZOR/F5-TTS-THAI/resolve/main"

    @staticmethod
    def has_submodule_file():
        return os.path.exists(os.path.join(Install.f5TTSPath, "README.md"))

    @staticmethod
    def check_install():
        Install.ensure_model_dir()
        if not Install.has_submodule_file():
            Install.install()
        Install.ensure_vocab()
        Install.ensure_default_model()

    @staticmethod
    def install():
        print("🔧 Checking for F5-TTS-THAI model and dependencies...")
        try:
            import pygit2
            repo = pygit2.Repository(os.path.dirname(__file__))
            pygit2.submodules.SubmoduleCollection(repo).update(init=True)
        except Exception as e:
            print("ℹ️ pygit2 not available, falling back to subprocess git init...")
            print(f"⚠️ pygit2 failed: {e}")

        subprocess.run(
            ['git', 'submodule', 'update', '--init', '--recursive'],
            cwd=os.path.dirname(__file__),
            check=True,
        )

        if not Install.has_submodule_file():
            print("❌ F5TTS: Git submodule failed to initialize. Please check your Git setup.")
        else:
            Install.install_requirements()

    @staticmethod
    def install_requirements():
        print("📦 F5-TTS-THAI: Installing Thai TTS dependencies...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "git+https://github.com/VYNCX/F5-TTS-THAI.git"
        ], check=True)

        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "torch==2.1.2+cu126", "torchaudio==2.1.2+cu126",
            "--index-url", "https://download.pytorch.org/whl/cu126"
        ], check=True)

    @staticmethod
    def ensure_model_dir():
        os.makedirs(Install.model_dir, exist_ok=True)

    @staticmethod
    def ensure_vocab():
        vocab_path = os.path.join(Install.model_dir, "vocab.txt")
        if not os.path.exists(vocab_path):
            print("⬇️ Downloading vocab.txt...")
            urllib.request.urlretrieve(
                f"{Install.model_url_base}/vocab.txt",
                vocab_path
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
