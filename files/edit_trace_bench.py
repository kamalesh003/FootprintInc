"""
edit_trace_bench.py -- replay REAL edit positions through the region-planned incremental transformer.

Needs (same folder): incgraph2.py, slabinc.py, regioninc.py, bench2.py   (unchanged, your files)
Colab:  !pip -q install datasets transformers
        !python edit_trace_bench.py

What it measures
  1. recompute fraction: mean (N-p)/N over real edit positions p  (exact for causal attention: an edit at
     chunk p dirties exactly chunks p..N-1 in every layer; append dirties 1 chunk/layer).
  2. wall time per edit for ordinary vs suffixkv vs incgraph[region+packed], interleaved per step.
  3. correctness: outputs vs `ordinary` every --check steps and at the end (dtype tolerance from bench2.TOL).

Dataset: bigcode/commitpackft (python).  Each commit gives old/new file; the first differing token, divided
by file length, is the RELATIVE edit position r.  r is mapped onto the N-chunk context (p = int(r*N)), and the
replacement chunk is the new file's tokens at that spot.  Weights are random (exactness + speed do not depend
on trained weights).  If the dataset cannot be loaded, a synthetic end-biased distribution is used and printed.
"""
import argparse, json, random, statistics, time
import torch
import bench2 as B2



def load_trace(n_commits, min_tok, vocab_fallback=50257):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    # Bypass commitpackft.py entirely. Load the published JSONL data directly.
    data_url = (
        "https://huggingface.co/datasets/bigcode/commitpackft/"
        "resolve/main/data/python/data.jsonl"
    )

    print(f"[dataset] Loading real Python commit traces from: {data_url}")

    try:
        tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

        ds = load_dataset(
            "json",
            data_files={"train": data_url},
            split="train",
            streaming=True,
        )

        edits, stream = [], []

        for ex in ds:
            old = ex.get("old_contents") or ""
            new = ex.get("new_contents") or ""

            if not old or not new or old == new:
                continue

            # Limit tokenization work for large source files.
            old_ids = tok(
                old[:30000],
                add_special_tokens=False,
            )["input_ids"]

            new_ids = tok(
                new[:30000],
                add_special_tokens=False,
            )["input_ids"]

            if len(old_ids) < min_tok or not new_ids:
                continue

            # First differing token, including pure insertions/appends.
            common = min(len(old_ids), len(new_ids))
            d = next(
                (
                    i for i, (x, y) in enumerate(
                        zip(old_ids[:common], new_ids[:common])
                    )
                    if x != y
                ),
                None,
            )

            if d is None:
                if len(old_ids) == len(new_ids):
                    continue
                d = common

            # Relative position in the old file.
            r = min(d / max(len(old_ids), 1), 1.0)

            # Replacement content begins at the first differing token.
            replacement = new_ids[d:d + 1024]
            if not replacement:
                continue

            edits.append((r, replacement))

            if len(stream) < 8192:
                stream.extend(old_ids[:8192 - len(stream)])

            if len(edits) >= n_commits and len(stream) >= 2048:
                break

        if len(edits) < 20:
            raise RuntimeError(
                f"Only {len(edits)} usable edits found; need at least 20."
            )

        if len(stream) < 2048:
            raise RuntimeError(
                f"Only {len(stream)} context tokens collected; need 2048."
            )

        print(
            f"[dataset] SUCCESS: real commit traces loaded; "
            f"edits={len(edits)}, context_tokens={len(stream)}"
        )

        return "commitpackft-python-jsonl", edits, stream, tok.vocab_size

    except Exception as e:
        raise RuntimeError(
            "Could not load real CommitPackFT Python traces. "
            "No synthetic fallback was used. Check dataset access, "
            "the JSONL URL, and the Hugging Face connection."
        ) from e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--C", type=int, default=128)
    ap.add_argument("--D", type=int, default=768)
    ap.add_argument("--H", type=int, default=12)
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--commits", type=int, default=200)
    ap.add_argument("--check", type=int, default=25)
    ap.add_argument("--systems", default="ordinary,suffixkv,incgraph[region+packed]")
    ap.add_argument("--out", default="edit_trace_results.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, a.dtype)
    N, C, D = a.T // a.C, a.C, a.D

    src, edits, stream, V = load_trace(a.commits, min_tok=256)
    g = torch.Generator().manual_seed(0)
    emb = (torch.randn(V, D, generator=g) * 0.02).to(device=dev, dtype=dtype)

    def E(ids):
        ids = (list(ids) + [0] * C)[:C]
        return emb[torch.tensor(ids, device=dev)].unsqueeze(0).contiguous()

    base = (stream + [0] * a.T)[:a.T]
    chunks = [E(base[i * C:(i + 1) * C]) for i in range(N)]
    trace = [(min(N - 1, int(r * N)), E(ids)) for r, ids in edits]

    ps = [p for p, _ in trace]
    frac = statistics.mean((N - p) / N for p in ps)
    print(f"dataset={src}  edits={len(trace)}  N={N} chunks of {C} tokens")
    print(f"median edited chunk p={statistics.median(ps)}/{N}   mean recompute fraction={frac:.3f}"
          f"   (=> at most {1 / frac:.2f}x fewer chunk-executions than full recompute)")

    params = B2.make_params(a.L, D, dev, dtype)
    facs = B2.factories(params, a.H)
    names = [n for n in a.systems.split(",") if n in facs]
    if "ordinary" not in names:
        names.insert(0, "ordinary")
    systems = {n: facs[n](chunks, N) for n in names}
    times = {n: [] for n in names}
    sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)

    for s, (p, c) in enumerate(trace):
        k = s % len(names)
        for n in names[k:] + names[:k]:                      # rotate order: cancels clock drift
            sync()
            t = time.perf_counter()
            systems[n].step(("edit", p, c))
            sync()
            times[n].append((time.perf_counter() - t) * 1e3)
        if (s + 1) % a.check == 0 or s == len(trace) - 1:
            ref = systems["ordinary"].output().float()
            for n in names:
                o = systems[n].output().float()
                rel = ((o - ref).abs().max() / (ref.abs().max() + 1e-6)).item()
                assert rel < B2.TOL[dtype], f"{n} deviates from ordinary at step {s}: rel={rel:.3e}"
    print("correctness: all systems match ordinary (checked every", a.check, "steps + final)")

    w = 3                                                    # drop warmup steps
    base_ms = statistics.mean(times["ordinary"][w:])
    thirds = [(0, N // 3), (N // 3, 2 * N // 3), (2 * N // 3, N)]
    print(f"\n{'system':<30}{'mean ms':>9}{'p50':>8}{'speedup':>9}   by edit position (first/mid/last third)")
    res = {}
    for n in names:
        t = times[n][w:]
        pp = ps[w:]
        row = []
        for lo, hi in thirds:
            sel = [x for x, q in zip(t, pp) if lo <= q < hi]
            ref = [x for x, q in zip(times["ordinary"][w:], pp) if lo <= q < hi]
            row.append(f"{statistics.mean(ref) / statistics.mean(sel):.2f}x" if sel else "  -  ")
        m = statistics.mean(t)
        print(f"{n:<30}{m:>9.2f}{statistics.median(t):>8.2f}{base_ms / m:>8.2f}x   {' / '.join(row)}")
        res[n] = dict(mean_ms=m, p50_ms=statistics.median(t), speedup=base_ms / m, by_third=row)
    json.dump(dict(dataset=src, N=N, recompute_fraction=frac, results=res), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()