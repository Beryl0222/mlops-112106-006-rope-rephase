"""按会话审查旋转键交付：借用、打分、预算与拒绝原因。"""

import argparse
import json
from pathlib import Path

import numpy as np

from rotary_keys import ExchangeRunner


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return np.round(value, 6).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def run(path: Path) -> dict:
    runner = ExchangeRunner.from_path(path)
    events = runner.run_timeline()
    scores = {}
    for query_id, result in runner.audit_all().items():
        if result["delivered"]:
            scores[query_id] = {
                block_id: runner.score(query_id)[block_id]
                for block_id in result["delivered"]
            }
    return {
        "资料": path.name,
        "流程": events,
        "审计": runner.audit_all(),
        "分数": scores,
        "驻留": runner.residency(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="审查旋转键交接")
    parser.add_argument("scene", type=Path)
    parser.add_argument("--audit", default=None, help="只输出指定查询的审计")
    args = parser.parse_args()

    runner = ExchangeRunner.from_path(args.scene)
    if args.audit is not None:
        runner.borrow(args.audit)
        print(json.dumps(_jsonable(runner.audit(args.audit)),
                         ensure_ascii=False, indent=2))
        return
    print(json.dumps(_jsonable(run(args.scene)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
