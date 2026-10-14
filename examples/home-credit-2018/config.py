from pathlib import Path

class CONFIG:

    # 项目根目录
    ROOT: Path = Path(__file__).resolve().parent

    # 一级目录
    DATA: Path = ROOT / "data"

    # data 子目录
    RAW_DATA: Path = DATA / "raw_data"
    PROCESSED_DATA: Path = DATA / "processed_data"

if __name__ == "__main__":
    print(CONFIG.ROOT)
    print(CONFIG.DATA)
    print(CONFIG.RAW_DATA)
    print(CONFIG.PROCESSED_DATA)
