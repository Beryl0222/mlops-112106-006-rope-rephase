"""检查交换资料的形状和索引。"""

import argparse
import json
from pathlib import Path

import numpy as np


def inspect(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    validate(data)
    return {
        "资料": path.name,
        "字段": sorted(data),
        "字节": path.stat().st_size,
        "数值环境": np.__version__,
    }


def validate(data: dict) -> None:
    for block in data["blocks"]:
            assert len(block["positions"]) == len(block["q"])
            assert np.shape(block["q"])[1] == data["width"]
            assert all(-128 <= q < 128 for row in block["q"] for q in row)
            assert block["phase"] is None or block["phase"] in data["views"]
        assert np.linalg.norm(data["query"]) <= data["query_norm_ceiling"]


def main() -> None:
    parser = argparse.ArgumentParser(description="检查本地交换资料")
    parser.add_argument("scene", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.scene), ensure_ascii=False))


if __name__ == "__main__":
    main()
