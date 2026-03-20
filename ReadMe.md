# Parallel BPE Tokenizer in C++

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/29b43369-335a-433b-9ecc-bba222056e83" />


A C++ systems project that implements a GPT-2-style byte-level BPE tokenizer from scratch, then scales it with parallel batch processing, a reusable thread pool, and a thread-local cache for repeated chunk reuse.

This project is not just “tokenization.” It is a systems-focused performance project that combines:

- low-level text processing
- parallel programming in C++
- custom thread-pool design
- cache-based optimization
- benchmark-driven engineering
- comparison against production-grade Python baselines

## Why this project matters

Modern LLM pipelines depend heavily on tokenization. Before a model can train or run inference, raw text must be converted into token IDs. That makes tokenization part of the critical path in:

- LLM training data preprocessing
- batch inference pipelines
- dataset preparation
- retrieval pipelines
- offline evaluation systems

This project focuses on the engineering side of that problem:

- implement the tokenizer correctly
- parallelize it for batch workloads
- reduce repeated computation with caching
- benchmark the system honestly against strong baselines

## What I built

### 1. GPT-2-style byte-level BPE tokenizer
I implemented the full tokenization pipeline in C++:

- load `vocab.json` into an `unordered_map<string, int>`
- load `merges.txt` into merge-rank lookup tables
- perform byte-to-unicode mapping
- split text into GPT-2-style chunks
- repeatedly apply BPE merges
- map final token strings to token IDs

I validated correctness against GPT-2-compatible outputs on sample texts before moving to performance work.

### 2. Batch tokenization
I added batch tokenization support so the tokenizer can process many text items at once:

- input: `vector<string>`
- output: `vector<vector<int>>`

This is the foundation for large-scale throughput benchmarking.

### 3. Static-range multithreading
As an intermediate parallelization step, I first implemented fixed-range multithreading:

- split the batch into disjoint index ranges
- launch multiple `std::thread` workers
- each worker tokenizes only its assigned slice
- join all workers before returning results

This helped verify the parallel design before introducing a reusable pool.

### 4. Reusable thread pool
I then replaced fixed chunk assignment with a proper thread pool:

- fixed set of worker threads
- shared queue of tasks
- `std::mutex` for shared state protection
- `std::condition_variable` for sleep/wake behavior
- graceful shutdown and completion waiting

Each input document becomes one task. Workers repeatedly pull tasks from the queue, which improves load balancing compared to static partitioning.

### 5. Thread-local chunk cache
I added a thread-local cache to avoid recomputing BPE for repeated chunks.

The cache stores:

- key: chunk string
- value: token ID vector

This means common chunks like repeated words, punctuation-attached fragments, and frequent subword patterns can skip the expensive BPE merge loop after the first computation on a worker.

I also tracked:

- cache hits
- cache misses
- cache hit rate

## Architecture

### Core tokenizer pipeline

`text -> split into chunks -> chunk to byte-level symbols -> BPE merges -> token strings -> token IDs`

### Parallel batch pipeline

`vector<string> docs -> thread pool -> one task per document -> vector<vector<int>>`

### Cached pipeline

`chunk -> thread-local cache lookup -> if hit return cached IDs -> if miss run full BPE -> store result -> return IDs`

## Benchmark highlights

All benchmarks were run on the same 10K-document WikiText corpus.

### Phase 3: thread-pool parallel batch tokenization

No-cache median throughput:

- 1 thread: **82.8K tokens/sec**
- 2 threads: **164.7K tokens/sec**
- 4 threads: **280.8K tokens/sec**
- 8 threads: **417.3K tokens/sec**

This delivered about **5.0x speedup** from 1 thread to 8 threads.

### Phase 4: thread-local cache optimization

Cached median throughput:

- 1 thread: **757.0K tokens/sec**
- 2 threads: **1.15M tokens/sec**
- 4 threads: **1.72M tokens/sec**
- 8 threads: **2.11M tokens/sec**

Cache hit rates:

- 1 thread: **95.2%**
- 2 threads: **93.0%**
- 4 threads: **90.2%**
- 8 threads: **86.6%**

The cache improved throughput by roughly:

- **9.1x** at 1 thread
- **7.0x** at 2 threads
- **6.1x** at 4 threads
- **5.0x** at 8 threads

### Python baseline comparison

I compared the optimized C++ implementation against strong GPT-2-family baselines on the same 10K-document corpus.

Median throughput:

- `tiktoken`: **2.30M tokens/sec**
- `GPT2TokenizerFast`: **2.40M tokens/sec**
- my C++ tokenizer with cache + 8 threads: **2.11M tokens/sec**

This means the optimized C++ implementation reached:

- about **91.6%** of `tiktoken` throughput
- about **88.1%** of Hugging Face fast throughput

That comparison is workload-specific, but it shows the implementation is competitive rather than just academically correct.

## Why the cache hit rate drops as threads increase

The cache is thread-local, not shared.

That means:

- each worker has its own private cache
- repeated chunks are reused only within that worker
- with more threads, repeated chunks are spread across more separate caches

So 8 threads still improves total runtime through added parallelism, but per-thread cache reuse becomes weaker than with 4 threads. That is why throughput improves while cache hit rate drops.

## Repository structure

```text
include/
  bpe.hpp
  byte_encoder.hpp
  thread_pool.hpp
  tokenizer_assets.hpp

src/
  bpe.cpp
  byte_encoder.cpp
  thread_pool.cpp
  tokenizer_assets.cpp
  main.cpp

data/
  document.txt
  vocab.json
  merges.txt
  wikitext_10k.txt   # generated locally, not required to commit

scripts/
  make_wikitext_10k.py
  benchmark_python_tokenizers.py

results/
  phase3_benchmark.csv
  phase4_cache_benchmark.csv
  phase4_python_comparison.csv
  phase4_python_comparison_summary.txt
Build
g++ -std=c++17 -Iinclude src/main.cpp src/bpe.cpp src/tokenizer_assets.cpp src/byte_encoder.cpp src/thread_pool.cpp -lpcre2-8 -pthread -O2 -o tokenizer_test
Run C++ benchmark
./tokenizer_test

This runs the benchmark over the 10K-document corpus and writes results to the results/ directory.

Generate the 10K benchmark corpus
python scripts/make_wikitext_10k.py
