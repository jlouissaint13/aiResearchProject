import shutil
import subprocess
import sys
import gpustat
import platform
class Startup:
    def __init__(self):
        #Checking with both just in case
        if sys.platform == "win32" or platform.system() == "Windows":
            self.ollama_setup_windows()
            try:
                for gpu in gpustat.GPUStatCollection.new_query():
                    #Doesn't need to actually be used if there is no nvidia gpu I will just catch the exception and throw the message below
                    gpu_name = gpu.name
            except Exception:
                print( "WARNING:\n"
                "No NVIDIA GPU found. CPU fallback is likely.\n"
                "The LLM will still run, but performance may be significantly slower.\n\n"
                "Alternatives:\n"
                "1) Search without LLM\n"
                "2) Use an API key (Note: this does not guarantee the safety of your data)")



    def ollama_setup_windows(self):
        if shutil.which("wsl") is None:
            print("Setting up linux VM")
            subprocess.run(["powershell", "-Command", "wsl --install -d Ubuntu"], check=True)
            subprocess.run([
                "wsl", "-d", "Ubuntu", "--", "bash", "-c",
                "curl -fsSL https://ollama.com/install.sh | sh"
            ], check=True)
            print("Downloading LLM Model llama3.2...")
            subprocess.run([
                "wsl", "-d", "Ubuntu", "--", "bash", "-c",
                "ollama pull llama:3.2"
            ], check=True)
            subprocess.run([
                "wsl", "-d", "Ubuntu", "--", "bash", "-c",
                "nohup ollama serve > /dev/null 2>&1 &"
            ])