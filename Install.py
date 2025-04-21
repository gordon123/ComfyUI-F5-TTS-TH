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
        print("F5-TTS-THAI: Initializing Thai TTS submodule")
        try:
            import pygit2
            repo = pygit2.Repository(os.path.dirname(__file__))
            pygit2.submodules.SubmoduleCollection(repo).update(init=True)
        except Exception as e:
            print(f"pygit2 failed: {e}")
        subprocess.run(
            ['git', 'submodule', 'update', '--init', '--recursive'],
            cwd=os.path.dirname(__file__),
            check=True,
        )
        if not Install.has_submodule_file():
            print("F5TTS. Something is wrong …")
        else:
            Install.install_requirements()

    @staticmethod
    def install_requirements():
        print("F5-TTS-THAI: Installing requirements")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r",
             os.path.join(Install.f5TTSPath, 'requirements.txt')],
            cwd=Install.f5TTSPath,
            check=True,
        )

    @staticmethod
    def has_model_file():
        path = os.path.join(Install.f5TTSPath, "ckpts", "thai", "model_475000_FP16.pt")
        return os.path.exists(path)
