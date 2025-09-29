"""Write a small marker file to runtime/logs to confirm write access."""

def main() -> int:
    p = Path("runtime") / "logs" / "test_fs_write.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        f.write("FS WRITE OK\n")
    print(f"wrote: {p}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
