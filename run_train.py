import subprocess
import os
import sys

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    cmd = [
        sys.executable,
        "train.py",
        "--epochs", "100",
        "--batch_size", "16"
    ]
    subprocess.run(cmd)
