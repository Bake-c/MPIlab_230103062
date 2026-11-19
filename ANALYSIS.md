# Cache Line and False Sharing Mechanics

Author: Bake-c

## Experimental Setup

I used four threads on a **12th Gen Intel(R) Core(TM) i5-12450H** with
12 logical processors. The threads were pinned to logical
CPUs `[0, 2, 4, 6]`, corresponding to physical cores
`['0', '1', '2', '3']`, using `windows_group_affinity`.

Each thread performed **60,000,000 updates** per run.
I repeated each configuration **3 times** and compared median
wall-clock times. The buffer was aligned to 128 bytes. The cache-line size was
64 bytes, obtained through `64_byte_fallback`.
On Windows, this is the assignment's 64-byte fallback rather than an OS cache-size query.

My instance seed was `10618267950495303938`, derived from my GitHub handle
and CPU signature. Verification status: **Passed**.
The individual runs, addresses, thread seeds and final counters are recorded in
[cache_lab_manifest.json](cache_lab_manifest.json).

## Q1. MESI State Transitions

Assuming the line initially has state `(I, I)` on Core 0 and Core 1, Core 0's
store to `buffer[0]` requires ownership through an RFO/GetM request. The stable
state becomes `(M, I)`. When Core 1 stores to `buffer[1]`, it requests ownership
of the same cache line. Core 0 supplies the latest data and invalidates its copy,
leaving `(I, M)`. A subsequent store by Core 0 transfers ownership back.

If both cores initially hold the line in Shared state, `(S, S)` becomes `(M, I)`
after Core 0 requests an ownership upgrade. If Core 0 already holds the line in
Exclusive state, its E-to-M transition is local. Actual hardware uses additional
transient states and implementation-specific messages. These transitions describe
the MESI model; this benchmark does not capture a hardware protocol trace.

The store buffer can initially accept a store while ownership is pending.
However, the store cannot become globally visible until the core acquires the
line and receives the required invalidation acknowledgements. Repeated ownership
transfers delay draining the store buffer and can eventually stall further stores.
The two threads access different elements, but coherence operates on the entire line.

## Q2. Timings and Coherence Overhead

| Configuration | Median time, ns | ns per update | Ratio to stride=8 | Context switches, voluntary / involuntary |
|---|---:|---:|---:|---|
| stride_1 | 133,836,500 | 0.557652 | 1.1132 | Unavailable (Windows) |
| stride_4 | 122,855,200 | 0.511897 | 1.0218 | Unavailable (Windows) |
| stride_8 | 120,229,800 | 0.500957 | 1.0000 | Unavailable (Windows) |
| stride_16 | 103,418,700 | 0.430911 | 0.8602 | Unavailable (Windows) |
| private | 62,407,800 | 0.260032 | 0.5191 | Unavailable (Windows) |

The slowdown of stride=1 relative to stride=8 was **1.113172x**.
Compared with the private baseline, stride=1 took **2.144548x** as long,
while stride=8 took **1.926519x** as long.

I calculated time per update as:

```text
ns_per_update = elapsed_ns / (4 * iterations_per_thread)
```

This measures the inverse aggregate throughput of all four threads rather than
the latency of one CPU instruction. The difference between stride=1 and stride=8
was **0.056695 ns per aggregate update**.
Scheduling, CPU frequency and synchronization also affect this difference, so it
cannot be interpreted as the isolated cost of one RFO transaction.

Windows does not provide `resource.getrusage`. Context-switch counts are therefore
stored as `null`. On supported systems, the counters cover the process, including
thread setup and synchronization.

## Q3. 64-Byte and 128-Byte Padding

The median time was **120,229,800 ns** for stride=8 and **103,418,700 ns** for stride=16.
The ratio of the 64-byte-spacing time to the 128-byte-spacing time was
**1.162554x**.

Both layouts place the counters in separate 64-byte cache lines. Some processors
have adjacent-line or spatial prefetchers that fetch a neighboring line within an
aligned 128-byte pair. If prefetching creates a cached copy of a line another core
is modifying, additional invalidations can occur. Spacing active counters by
128 bytes places them in different adjacent-line pairs.

This behavior depends on the processor and its prefetch policy. Padding to
128 bytes does not prevent every possible prefetch interaction. My timings show
a performance difference, but they do not prove that the prefetcher caused it.
That would require additional hardware-event measurements or controlled changes
to prefetch settings. The manifest records the actual cache-line indices.

## Q4. Register Promotion and LLVM Verification

An ordinary loop containing `buffer[slot] += value` can be optimized into a
register accumulator followed by one final store. The address remains constant,
and an arithmetic dependency alone does not force intermediate memory accesses.

I used a Numba intrinsic to emit explicit `load volatile i64` and
`store volatile i64` instructions. LLVM must preserve these volatile accesses.
They do not introduce atomic read-modify-write instructions, locks or memory
fences. Atomic operations are unnecessary here because each thread owns a distinct
array element.

The LLVM IR from this run includes:

```llvm
%counter_read.prol = load volatile i64, ptr %sunkaddr2, align 8
store volatile i64 %.159.prol, ptr %sunkaddr2, align 8
```

The audit found volatile loads and stores in cyclic CFG blocks
`['B66.prol', 'B66']`. The compiled signature was
`["(Array(uint64, 1, 'C', False, aligned=True), int64, int64, uint64) -> none"]`. The native function had no calls to the checked
Python API or GIL helpers: **True**.

The full IR is in [worker_strided_accumulate.ll](worker_strided_accumulate.ll).
Its SHA-256 is `69e9f35f74c7cb89611ee93a00eea06e4b16f85e9da7bb18c8e13cabcb5d8e5e`.
The private kernel contains **1**
ordinary i64 store in its native function and no volatile stores.

I checked the final values against an independent reference that exponentiates
an affine transformation modulo 2^64. All configurations and repetitions matched:
**True**. Both kernels use the same LCG arithmetic.
The private baseline removes repeated counter-memory accesses, but its timing is
not a direct measurement of store-forwarding latency alone.

## Q5. CPython GIL and Native Execution

In a conventional CPython build with the GIL enabled, only one thread executes
Python bytecode at a time. Pure Python counter loops therefore do not produce
the same sustained concurrent store contention across four cores. GIL handoffs
and CPU migration can still transfer cache lines, but interpreter overhead,
lock contention and scheduling dominate the experiment. Adding threads usually
fails to accelerate such CPU-bound loops and can make them slower.

With `@njit(nogil=True)`, Numba compiles the loop to native code and releases
the GIL during the call. The four pinned threads can then execute their loops
concurrently. I compiled and checked both kernels before starting the timed runs.
The manifest also records whether all worker execution intervals overlap; this
is a timing check rather than an instruction-level trace.

This explanation applies to CPython with the GIL enabled. Free-threaded builds
need separate consideration. I did not run a separate Python-bytecode benchmark;
all numerical results above come from the Numba kernels.

## Measurement Method

I shuffled the five configurations deterministically between repetitions.
`perf_counter_ns()` started when the barrier released the workers, and the final
timestamp was the latest kernel-completion timestamp. JIT compilation, affinity
setup, thread creation and output-file writing were outside this timed interval.
Barrier release and kernel entry were included.

The measurements remain sensitive to background activity, turbo frequency and
scheduling. Verification checks correctness and LLVM behavior without requiring
a predetermined slowdown. System notes: 64-byte line size assumed because sysfs/sysctl is unavailable.

Run timestamp, UTC: `2026-11-19T07:50:36.730815+00:00`.

## References

- [Numba intrinsic API](https://numba.readthedocs.io/en/stable/extending/high-level.html#implementing-intrinsics)
- [LLVM volatile memory accesses](https://llvm.org/docs/LangRef.html#volatile-memory-accesses)
