# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure concurrent OpenAI chat throughput and latency without dependencies.

Start ``examples/openai_server.py`` first, then run for example::

    python examples/benchmark_openai_concurrency.py --concurrency 1,2,4

The benchmark uses the public streaming API rather than runner internals.  This
makes it representative of agent traffic and keeps it useful across scheduler
implementations.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class RequestMetrics:
    prompt_tokens: int
    completion_tokens: int
    started: float
    first_token: float
    finished: float
    event_times: tuple[float, ...]

    @property
    def ttft(self) -> float:
        return self.first_token - self.started

    @property
    def generation_seconds(self) -> float:
        return max(0.0, self.finished - self.first_token)

    @property
    def output_tokens_per_second(self) -> float:
        return self.completion_tokens / max(self.generation_seconds, 1.0e-9)

    @property
    def inter_token_latencies(self) -> tuple[float, ...]:
        return tuple(b - a for a, b in zip(self.event_times, self.event_times[1:]))


@dataclass(frozen=True)
class LevelSummary:
    concurrency: int
    requests: int
    completion_tokens: int
    wall_seconds: float
    aggregate_tokens_per_second: float
    per_request_tokens_per_second: float
    ttft_p50_ms: float
    ttft_p95_ms: float
    inter_token_p50_ms: float
    inter_token_p95_ms: float
    memory_peak_mib: int | None = None
    memory_growth_mib: int | None = None


def percentile(values: list[float], fraction: float) -> float:
    """Return an interpolated percentile without NumPy."""

    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(
    concurrency: int,
    requests: list[RequestMetrics],
    memory_peak_mib: int | None = None,
    memory_growth_mib: int | None = None,
) -> LevelSummary:
    """Aggregate one concurrency level using exact API usage token counts."""

    if not requests:
        raise ValueError("at least one request is required")
    # Requests are appended one complete concurrency group at a time. Summing
    # each group excludes host-side setup gaps between rounds.
    groups = [
        requests[index : index + concurrency]
        for index in range(0, len(requests), concurrency)
    ]
    wall = sum(
        max(item.finished for item in group) - min(item.started for item in group)
        for group in groups
    )
    wall = max(wall, 1.0e-9)
    tokens = sum(item.completion_tokens for item in requests)
    itl = [latency for item in requests for latency in item.inter_token_latencies]
    return LevelSummary(
        concurrency=concurrency,
        requests=len(requests),
        completion_tokens=tokens,
        wall_seconds=wall,
        aggregate_tokens_per_second=tokens / wall,
        per_request_tokens_per_second=statistics.fmean(
            item.output_tokens_per_second for item in requests
        ),
        ttft_p50_ms=percentile([item.ttft * 1000.0 for item in requests], 0.50),
        ttft_p95_ms=percentile([item.ttft * 1000.0 for item in requests], 0.95),
        inter_token_p50_ms=percentile([value * 1000.0 for value in itl], 0.50),
        inter_token_p95_ms=percentile([value * 1000.0 for value in itl], 0.95),
        memory_peak_mib=memory_peak_mib,
        memory_growth_mib=memory_growth_mib,
    )


def _payload_has_token(chunk: dict[str, object]) -> bool:
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    delta = choices[0].get("delta", {})
    if not isinstance(delta, dict):
        return False
    return bool(
        delta.get("content")
        or delta.get("reasoning_content")
        or delta.get("tool_calls")
    )


def run_request(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    api_key: str | None = None,
) -> RequestMetrics:
    """Run one streamed request and timestamp observable output-token events."""

    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    started = time.perf_counter()
    event_times = []
    usage = None
    with urlopen(Request(url, data=body, headers=headers), timeout=3600) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if _payload_has_token(chunk):
                event_times.append(time.perf_counter())
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
    finished = time.perf_counter()
    if usage is None:
        raise RuntimeError("server did not return usage; include_usage is required")
    first_token = event_times[0] if event_times else finished
    return RequestMetrics(
        prompt_tokens=int(usage["prompt_tokens"]),
        completion_tokens=int(usage["completion_tokens"]),
        started=started,
        first_token=first_token,
        finished=finished,
        event_times=tuple(event_times),
    )


def _gpu_memory_mib(device: int, server_pid: int | None) -> int | None:
    query = (
        "--query-compute-apps=pid,used_gpu_memory"
        if server_pid
        else "--query-gpu=index,memory.used"
    )
    try:
        result = subprocess.run(
            ["nvidia-smi", query, "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    total = 0
    for line in result.stdout.splitlines():
        try:
            index, used = (int(part.strip()) for part in line.split(",", 1))
        except ValueError:
            continue
        if (server_pid is not None and index == server_pid) or (
            server_pid is None and index == device
        ):
            total += used
    return total


class MemorySampler:
    def __init__(self, device: int, server_pid: int | None):
        self.device = device
        self.server_pid = server_pid
        self.baseline = _gpu_memory_mib(device, server_pid)
        self.peak = self.baseline
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self._stop.wait(0.05):
            value = _gpu_memory_mib(self.device, self.server_pid)
            if value is not None:
                self.peak = value if self.peak is None else max(self.peak, value)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()

    @property
    def growth(self) -> int | None:
        if self.peak is None or self.baseline is None:
            return None
        return self.peak - self.baseline


def run_level(
    *,
    concurrency: int,
    rounds: int,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    api_key: str | None,
    device: int,
    server_pid: int | None,
) -> LevelSummary:
    results: list[RequestMetrics] = []
    sampler = MemorySampler(device, server_pid)
    with sampler:
        for round_index in range(rounds):
            barrier = threading.Barrier(concurrency)
            batch: list[RequestMetrics | BaseException | None] = [None] * concurrency

            def worker(index: int):
                try:
                    barrier.wait()
                    unique = (
                        f"{prompt}\nBenchmark request {round_index + 1}-{index + 1}."
                    )
                    batch[index] = run_request(url, model, unique, max_tokens, api_key)
                except BaseException as error:
                    batch[index] = error

            threads = [
                threading.Thread(target=worker, args=(index,))
                for index in range(concurrency)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            errors = [item for item in batch if isinstance(item, BaseException)]
            if errors:
                raise RuntimeError(f"request failed: {errors[0]}") from errors[0]
            results.extend(item for item in batch if isinstance(item, RequestMetrics))
    return summarize(concurrency, results, sampler.peak, sampler.growth)


def _print_table(summaries: list[LevelSummary]):
    print(
        "\n conc   agg tok/s   req tok/s   TTFT p50/p95 ms   ITL p50/p95 ms   GPU MiB (+growth)"
    )
    for item in summaries:
        memory = (
            "n/a"
            if item.memory_peak_mib is None
            else f"{item.memory_peak_mib} (+{item.memory_growth_mib})"
        )
        print(
            f" {item.concurrency:>4}  {item.aggregate_tokens_per_second:>10.2f}"
            f"  {item.per_request_tokens_per_second:>10.2f}"
            f"  {item.ttft_p50_ms:>8.1f}/{item.ttft_p95_ms:<8.1f}"
            f"  {item.inter_token_p50_ms:>7.1f}/{item.inter_token_p95_ms:<7.1f}  {memory}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", default="warp-qwen")
    parser.add_argument("--api-key")
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--prompt",
        default="Explain one practical benefit of continuous batching in two paragraphs.",
    )
    parser.add_argument(
        "--device", type=int, default=0, help="GPU index for memory accounting"
    )
    parser.add_argument(
        "--server-pid", type=int, help="Measure only this server process's GPU memory"
    )
    parser.add_argument("--json", type=Path, help="Also write machine-readable results")
    parser.add_argument(
        "--batch-one-baseline-tps",
        type=float,
        help="Known batch-one tok/s; enforce --batch-one-tolerance around it",
    )
    parser.add_argument(
        "--batch-one-tolerance",
        type=float,
        default=0.01,
        help="Allowed fractional batch-one regression (default: 0.01)",
    )
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=1.5,
        help="Required largest-level speedup over serialized batch one (default: 1.5)",
    )
    args = parser.parse_args()
    levels = sorted({int(value) for value in args.concurrency.split(",")})
    if (
        any(value < 1 for value in levels)
        or args.rounds < 1
        or args.max_tokens < 1
        or not 0.0 <= args.batch_one_tolerance < 1.0
    ):
        parser.error("concurrency, rounds, and max-tokens must be positive")

    print("Warming up the server...")
    run_request(
        args.url,
        args.model,
        args.prompt + "\nWarmup.",
        min(args.max_tokens, 16),
        args.api_key,
    )
    summaries = [
        run_level(
            concurrency=level,
            rounds=args.rounds,
            url=args.url,
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            api_key=args.api_key,
            device=args.device,
            server_pid=args.server_pid,
        )
        for level in levels
    ]
    _print_table(summaries)
    if any(item.concurrency == 1 for item in summaries):
        print("Concurrency 1 is the explicit serialized-serving baseline.")
    if args.json:
        args.json.write_text(
            json.dumps([asdict(item) for item in summaries], indent=2) + "\n"
        )

    batch_one = next((item for item in summaries if item.concurrency == 1), None)
    if args.batch_one_baseline_tps is not None:
        if batch_one is None:
            raise SystemExit("--batch-one-baseline-tps requires concurrency level 1")
        ratio = batch_one.aggregate_tokens_per_second / args.batch_one_baseline_tps
        print(f"Batch-one ratio to recorded baseline: {ratio:.3f}x")
        if ratio < 1.0 - args.batch_one_tolerance:
            raise SystemExit("batch-one throughput regression gate failed")
    if args.min_speedup is not None:
        if batch_one is None:
            raise SystemExit("--min-speedup requires concurrency level 1")
        largest = max(summaries, key=lambda item: item.concurrency)
        speedup = (
            largest.aggregate_tokens_per_second / batch_one.aggregate_tokens_per_second
        )
        print(
            f"Concurrency {largest.concurrency} aggregate speedup over serialized: "
            f"{speedup:.2f}x"
        )
        if speedup < args.min_speedup:
            raise SystemExit("aggregate throughput speedup gate failed")


if __name__ == "__main__":
    main()
