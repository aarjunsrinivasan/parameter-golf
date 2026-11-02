"""Summarize run_core.sh logs as a markdown table (2-seed mean, size check, delta vs baseline).

Usage: python summarize.py [log_dir]   (default: logs)
Reads core_<config>_<gpu>_seq<L>_s<seed>.txt files written by run_core.sh.
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

LIMIT = 16_000_000
NAME = re.compile(r"^core_(?P<cfg>baseline|gdn_h8k42|gdn_h5k64|kda_h8k42|kda_h5k64)_(?P<gpu>[a-z0-9]+)_seq(?P<seq>\d+)_s(?P<seed>\d+)\.txt$")
ORDER = ["baseline", "gdn_h8k42", "gdn_h5k64", "kda_h8k42", "kda_h5k64"]


def last(pattern: str, text: str):
    m = re.findall(pattern, text, flags=re.M)
    return m[-1] if m else None


def parse(path: Path):
    text = path.read_text(errors="replace")
    bpb = last(r"^final_int8_zlib_roundtrip val_loss:\S+ val_bpb:(\S+)", text)
    if bpb is None:
        return None
    stop = last(r"^stopping_early: wallclock_cap train_time:(\d+)ms step:(\d+)/", text)
    size = last(r"^Total submission size int8\+zlib: (\d+) bytes", text)
    params = last(r"^model_params:(\d+)", text)
    train_ms, steps = (int(stop[0]), int(stop[1])) if stop else (None, None)
    return {
        "bpb": float(bpb),
        "steps": steps,
        "ms": train_ms / steps if steps else None,
        "size": int(size) if size else None,
        "params": int(params) if params else None,
    }


def main() -> None:
    log_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "logs")
    groups = defaultdict(dict)
    incomplete = []
    for p in sorted(log_dir.glob("*.txt")):
        m = NAME.match(p.name)
        if not m:
            continue
        r = parse(p)
        if r is None:
            incomplete.append(p.name)
            continue
        groups[(m["gpu"], int(m["seq"]), m["cfg"])][int(m["seed"])] = r

    for gpu, seq in sorted({(g, s) for g, s, _ in groups}):
        rows = [(c, groups[(gpu, seq, c)]) for c in ORDER if (gpu, seq, c) in groups]
        base = groups.get((gpu, seq, "baseline"))
        base_mean = sum(r["bpb"] for r in base.values()) / len(base) if base else None
        print(f"\n### {gpu.upper()}, seq {seq}, FA3, wall-clock-fraction schedule (0.18 / 0.36)\n")
        print("| config | params | ms/step | steps | val_bpb per seed | mean ± half-range | Δ vs baseline | max bytes | ≤16 MB |")
        print("|---|---|---|---|---|---|---|---|---|")
        for cfg, seeds in rows:
            vals = [seeds[s]["bpb"] for s in sorted(seeds)]
            mean = sum(vals) / len(vals)
            half = (max(vals) - min(vals)) / 2
            ms = sum(r["ms"] for r in seeds.values() if r["ms"]) / len(seeds)
            steps = "/".join(str(seeds[s]["steps"]) for s in sorted(seeds))
            per_seed = ", ".join(f"{seeds[s]['bpb']:.4f} (s{s})" for s in sorted(seeds))
            sizes = [r["size"] for r in seeds.values() if r["size"]]
            max_size = max(sizes) if sizes else None
            params = next((r["params"] for r in seeds.values() if r["params"]), None)
            delta = f"{mean - base_mean:+.4f}" if base_mean is not None and cfg != "baseline" else "—"
            ok = "yes" if max_size is not None and max_size <= LIMIT else "**NO**"
            print(
                f"| {cfg} | {params:,} | {ms:.0f} | {steps} | {per_seed} | {mean:.4f} ± {half:.4f} "
                f"| {delta} | {max_size:,} | {ok} |"
            )
    if incomplete:
        print("\nNo final score yet (running or failed):", ", ".join(incomplete))


if __name__ == "__main__":
    main()
