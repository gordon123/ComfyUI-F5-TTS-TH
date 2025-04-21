import subprocess, sys, os


class Install:
    f5TTSPath = os.path.join(os.path.dirname(__file__), "F5-TTS-THAI")

    @staticmethod
    def has_submodule_file():
        return os.path.exists(os.path.join(Install.f5TTSPath, "README.md"))

    @staticmethod
    def check_install():
        if not Install.has_submodule_file():
            Install.install()

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
# F5-TTS-THAI: Installing us cuda 11.8

        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "torch==2.1.2+cu126", "torchaudio==2.1.2+cu126",
            "--index-url", "https://download.pytorch.org/whl/cu126"
        ], check=True)

    @staticmethod
    def has_model_file():
        model_path = os.path.join(Install.f5TTSPath, "ckpts", "thai", "model_475000_FP16.pt")
        return os.path.exists(model_path)
