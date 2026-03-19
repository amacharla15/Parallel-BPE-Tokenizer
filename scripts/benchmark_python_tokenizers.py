import csv
import time
from statistics import median

import tiktoken
from transformers import GPT2TokenizerFast

DATA_PATH = "data/wikitext_10k.txt"
OUT_PATH = "results/phase4_python_comparison.csv"
TRIALS = 3

def read_docs(path: str) -> list[str]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.endswith("\r"):
                line = line[:-1]
            if line:
                docs.append(line)
    return docs

def bench_tiktoken(docs: list[str], trials: int):
    enc = tiktoken.get_encoding("r50k_base")

    _ = enc.encode_ordinary_batch(docs, num_threads=8)

    rows = []
    for trial in range(1, trials + 1):
        start = time.perf_counter()
        ids = enc.encode_ordinary_batch(docs, num_threads=8)
        elapsed = time.perf_counter() - start

        total_docs = len(ids)
        total_tokens = sum(len(x) for x in ids)
        docs_per_sec = total_docs / elapsed
        tokens_per_sec = total_tokens / elapsed

        rows.append({
            "tool": "tiktoken_r50k_base",
            "trial": trial,
            "elapsed_sec": elapsed,
            "total_docs": total_docs,
            "total_tokens": total_tokens,
            "docs_per_sec": docs_per_sec,
            "tokens_per_sec": tokens_per_sec,
        })
    return rows

def bench_hf_fast(docs: list[str], trials: int):
    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    _ = tok(docs, add_special_tokens=False)["input_ids"]

    rows = []
    for trial in range(1, trials + 1):
        start = time.perf_counter()
        out = tok(docs, add_special_tokens=False)["input_ids"]
        elapsed = time.perf_counter() - start

        total_docs = len(out)
        total_tokens = sum(len(x) for x in out)
        docs_per_sec = total_docs / elapsed
        tokens_per_sec = total_tokens / elapsed

        rows.append({
            "tool": "hf_gpt2_fast",
            "trial": trial,
            "elapsed_sec": elapsed,
            "total_docs": total_docs,
            "total_tokens": total_tokens,
            "docs_per_sec": docs_per_sec,
            "tokens_per_sec": tokens_per_sec,
        })
    return rows

def print_summary(rows, tool_name):
    times = [r["elapsed_sec"] for r in rows]
    median_elapsed = median(times)

    total_docs = rows[0]["total_docs"]
    total_tokens = rows[0]["total_tokens"]

    print(f"\nTool = {tool_name}")
    for r in rows:
        print(
            f"Trial {r['trial']} | elapsed_sec = {r['elapsed_sec']:.6f}"
            f" | total_docs = {r['total_docs']}"
            f" | total_tokens = {r['total_tokens']}"
            f" | docs_per_sec = {r['docs_per_sec']:.6f}"
            f" | tokens_per_sec = {r['tokens_per_sec']:.6f}"
        )

    print(
        f"Median | elapsed_sec = {median_elapsed:.6f}"
        f" | docs_per_sec = {total_docs / median_elapsed:.6f}"
        f" | tokens_per_sec = {total_tokens / median_elapsed:.6f}"
    )

def main():
    docs = read_docs(DATA_PATH)
    print("Loaded documents:", len(docs))

    tik_rows = bench_tiktoken(docs, TRIALS)
    hf_rows = bench_hf_fast(docs, TRIALS)

    print_summary(tik_rows, "tiktoken_r50k_base")
    print_summary(hf_rows, "hf_gpt2_fast")

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tool",
                "trial",
                "elapsed_sec",
                "total_docs",
                "total_tokens",
                "docs_per_sec",
                "tokens_per_sec",
            ],
        )
        writer.writeheader()
        for row in tik_rows + hf_rows:
            writer.writerow(row)

    print(f"\nSaved CSV to {OUT_PATH}")

if __name__ == "__main__":
    main()