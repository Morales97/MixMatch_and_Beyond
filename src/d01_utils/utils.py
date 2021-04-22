from pathlib import Path

def foo():
    print("bar")

def get_project_root() -> Path:
    print(Path(__file__).parent.parent)
    return Path(__file__).parent.parent

if __name__ == "__main__":
    get_project_root()
