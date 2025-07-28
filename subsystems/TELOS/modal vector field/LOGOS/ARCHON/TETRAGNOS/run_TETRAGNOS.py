# File: LOGOS/TETRAGNOS/run_TETRAGNOS.py

import subprocess
import os

def main():
    print("🟠 Initializing TETRAGNOS System...")
    tetragnos_core_path = os.path.join(os.getcwd(), "TETRAGNOS", "LogosFullIntegration.py")
    if os.path.exists(tetragnos_core_path):
        subprocess.run(["python", tetragnos_core_path])
    else:
        print("❌ LogosFullIntegration.py not found in TETRAGNOS/")

if __name__ == "__main__":
    main()
