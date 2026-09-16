
# Restored: these names were used below but never imported, so this
# module raised NameError on import.
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def main():
    ws._persist_confirm_store("9999", "debug-sig-1")
    print("called _persist_confirm_store")

if __name__ == "__main__":
    main()
