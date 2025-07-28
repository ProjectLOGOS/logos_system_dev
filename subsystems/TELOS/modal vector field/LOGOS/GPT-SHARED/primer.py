import sys
import os

# ✅ Step 1: Set the absolute path to your LOGOS directory
LOGOS_ROOT = r"C:\Users\proje\OneDrive\Desktop\LOGOS\PROJECT LOGOS 2025\3 Pillars Complete Packet\LOGOS"

# ✅ Step 2: Define subdirectories to ensure path access
SUBDIRS = ["TELOS", "THONOC", "TETRAGNOS", "GPT-SHARED"]

# ✅ Step 3: Add all subdirectories to system path
for subdir in SUBDIRS:
    path = os.path.join(LOGOS_ROOT, subdir)
    if path not in sys.path:
        sys.path.append(path)

# ✅ Step 4: Confirm visibility
print("✔️ Environment initialized.")
print("Paths available:")
for p in sys.path[-len(SUBDIRS):]:
    print(" -", p)

# ✅ Step 5: Confirm GPT-SHARED contents (e.g. launcher, command sets)
shared_path = os.path.join(LOGOS_ROOT, "GPT-SHARED")
if os.path.exists(shared_path):
    print("\n📁 GPT-SHARED contains:", os.listdir(shared_path))
else:
    print("⚠️ GPT-SHARED directory not found. Check your LOGOS_ROOT path.")
