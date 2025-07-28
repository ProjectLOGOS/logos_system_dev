import os
import subprocess

def launch(script_path):
    subprocess.Popen(["python", script_path], creationflags=subprocess.CREATE_NEW_CONSOLE)

exec(open("GPT-SHARED/primer.py", encoding="utf-8").read())

def menu():
    print("\n=== LOGOS Launcher ===")
    print("1. Launch TELOS")
    print("2. Launch THONOC")
    print("3. Launch TETRAGNOS")
    print("4. Launch ALL Systems")
    print("0. Exit")

def main():
    base_dir = os.getcwd()

    paths = {
        "1": os.path.join(base_dir, "TELOS", "run_TELOS.py"),
        "2": os.path.join(base_dir, "THONOC", "run_THONOC.py"),
        "3": os.path.join(base_dir, "TETRAGNOS", "run_TETRAGNOS.py"),
    }

    while True:
        menu()
        choice = input("Select an option: ").strip()

        if choice == "0":
            print("Exiting launcher.")
            break
        elif choice in ["1", "2", "3"]:
            launch(paths[choice])
        elif choice == "4":
            for script in paths.values():
                launch(script)
            print("✅ All systems launched in parallel.")
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
