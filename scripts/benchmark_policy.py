import dataclasses
import enum
import statistics
import time

import numpy as np
import tyro

from openpi.policies import aloha_policy
from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


class ExampleType(enum.Enum):
    ALOHA = "aloha"
    DROID = "droid"


@dataclasses.dataclass
class Args:
    config: str
    checkpoint_dir: str
    example: ExampleType = ExampleType.ALOHA
    warmup: int = 10
    runs: int = 100


def make_example(example: ExampleType) -> dict:
    match example:
        case ExampleType.ALOHA:
            return aloha_policy.make_aloha_example()
        case ExampleType.DROID:
            return droid_policy.make_droid_example()


def percentile(sorted_values: list[float], p: float) -> float:
    idx = max(0, min(len(sorted_values) - 1, int(len(sorted_values) * p) - 1))
    return sorted_values[idx]


def main(args: Args) -> None:
    config = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(config, args.checkpoint_dir)
    example = make_example(args.example)

    for _ in range(args.warmup):
        outputs = policy.infer(example)
        np.asarray(outputs["actions"])

    infer_times: list[float] = []
    for _ in range(args.runs):
        start = time.perf_counter()
        outputs = policy.infer(example)
        np.asarray(outputs["actions"])
        infer_times.append(time.perf_counter() - start)

    infer_times_sorted = sorted(infer_times)
    avg = statistics.mean(infer_times)
    median = statistics.median(infer_times)
    p90 = percentile(infer_times_sorted, 0.90)
    p95 = percentile(infer_times_sorted, 0.95)

    print(f"config {args.config}")
    print(f"checkpoint_dir {args.checkpoint_dir}")
    print(f"example {args.example.value}")
    print(f"action_horizon {config.model.action_horizon}")
    print(f"warmup {args.warmup}")
    print(f"runs {args.runs}")
    print(f"avg_sec {avg}")
    print(f"median_sec {median}")
    print(f"p90_sec {p90}")
    print(f"p95_sec {p95}")
    print(f"min_sec {infer_times_sorted[0]}")
    print(f"max_sec {infer_times_sorted[-1]}")
    print(f"hz_avg {1.0 / avg}")
    print(f"hz_median {1.0 / median}")
    print(f"hz_p90 {1.0 / p90}")
    print(f"chunk_hz_avg {config.model.action_horizon / avg}")
    print(f"chunk_hz_median {config.model.action_horizon / median}")
    print("first10_sec", [round(x, 6) for x in infer_times[:10]])


if __name__ == "__main__":
    main(tyro.cli(Args))
