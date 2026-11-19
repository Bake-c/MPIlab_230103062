"""Cache-line laboratory. Run: python false_sharing_lab.py --git-user Bake-c."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import random
import re
import statistics
import struct
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import llvmlite
import numpy as np
import numba
from llvmlite import binding, ir
from numba import njit, types, uint64
from numba.extending import intrinsic

try:
    import resource
except ImportError:
    resource = None

GIT_USERNAME = "Bake-c"
NUM_THREADS = 4
ITERATIONS_PER_THREAD = 60_000_000
MASK = (1 << 64) - 1
MULTIPLIER = 6364136223846793005
INCREMENT = 1442695040888963407


class VolatileLoad(ir.instructions.LoadInstr):
    def descr(self, buf):
        parts = []
        super().descr(parts)
        buf.extend(s.replace("load ", "load volatile ", 1) for s in parts)


class VolatileStore(ir.instructions.StoreInstr):
    def descr(self, buf):
        parts = []
        super().descr(parts)
        buf.extend(s.replace("store ", "store volatile ", 1) for s in parts)


@intrinsic
def volatile_accumulate(typingctx, array, index, value):
    """An actual uint64 load/add/store, with no atomic or fence overhead."""
    if not (isinstance(array, types.Array) and array.ndim == 1
            and array.dtype == types.uint64 and isinstance(index, types.Integer)
            and value == types.uint64):
        raise TypeError("Expected a uint64 array, integer index and uint64 value")
    signature = types.uint64(array, index, value)

    def codegen(context, builder, sig, args):
        ary = context.make_array(sig.args[0])(context, builder, args[0])
        slot = context.cast(builder, args[1], sig.args[1], types.intp)
        pointer = builder.gep(ary.data, [slot])
        load = VolatileLoad(builder.block, pointer, name="counter_read")
        load.align = 8
        builder._insert(load)
        updated = builder.add(load, args[2])
        store = VolatileStore(builder.block, updated, pointer)
        store.align = 8
        builder._insert(store)
        return updated

    return signature, codegen


@njit(nogil=True)
def worker_strided_accumulate(buffer, slot, iters, thread_seed):
    state = uint64(thread_seed)
    for _ in range(iters):
        state = state * uint64(MULTIPLIER) + uint64(INCREMENT)
        volatile_accumulate(buffer, slot, state)


@njit(nogil=True)
def worker_private_accumulate(buffer, slot, iters, thread_seed):
    state = uint64(thread_seed)
    accumulator = uint64(0)
    for _ in range(iters):
        state = state * uint64(MULTIPLIER) + uint64(INCREMENT)
        accumulator += state
    buffer[slot] = accumulator


def expected_counter(seed, iters):
    """Independent O(log n) affine reference, all arithmetic modulo 2**64.

    Transform: state'=a*state+b, sum'=sum+c*state+d.
    Binary exponentiation avoids running another 60 million Python iterations.
    """
    a, b, c, d = 1, 0, 0, 0
    pa, pb, pc, pd = MULTIPLIER, INCREMENT, MULTIPLIER, INCREMENT
    while iters:
        if iters & 1:
            a, b, c, d = ((pa * a) & MASK, (pa * b + pb) & MASK,
                          (c + pc * a) & MASK, (d + pc * b + pd) & MASK)
        pa, pb, pc, pd = ((pa * pa) & MASK, (pa * pb + pb) & MASK,
                          (pc + pc * pa) & MASK, (pd + pc * pb + pd) & MASK)
        iters >>= 1
    return (c * seed + d) & MASK


def aligned_buffer(count, alignment=128):
    owner = np.zeros(count * 8 + alignment, dtype=np.uint8)
    offset = (-owner.ctypes.data) % alignment
    buffer = owner[offset:offset + count * 8].view(np.uint64)
    if buffer.ctypes.data % alignment:
        raise RuntimeError("Array alignment failed")
    return buffer  # ndarray.base keeps the owning allocation alive.


def detect_cache_line_size():
    try:
        value = int(Path("/sys/devices/system/cpu/cpu0/cache/index0/"
                         "coherency_line_size").read_text().strip())
        if value > 0:
            return value, "linux_sysfs"
    except (OSError, ValueError):
        pass
    if sys.platform == "darwin":
        try:
            value = int(subprocess.check_output(
                ["sysctl", "-n", "hw.cachelinesize"], text=True).strip())
            if value > 0:
                return value, "sysctl"
        except (OSError, ValueError, subprocess.CalledProcessError):
            pass
    return 64, "64_byte_fallback"


class GroupAffinity(ctypes.Structure):
    _fields_ = [("Mask", ctypes.c_size_t), ("Group", ctypes.c_ushort),
                ("Reserved", ctypes.c_ushort * 3)]


def windows_api():
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.GetCurrentThread.restype = ctypes.c_void_p
    api.GetThreadGroupAffinity.argtypes = [ctypes.c_void_p, ctypes.POINTER(GroupAffinity)]
    api.GetThreadGroupAffinity.restype = ctypes.c_int
    api.SetThreadGroupAffinity.argtypes = [ctypes.c_void_p, ctypes.POINTER(GroupAffinity),
                                          ctypes.POINTER(GroupAffinity)]
    api.SetThreadGroupAffinity.restype = ctypes.c_int
    api.GetLogicalProcessorInformationEx.argtypes = [ctypes.c_int, ctypes.c_void_p,
                                                    ctypes.POINTER(ctypes.c_ulong)]
    api.GetLogicalProcessorInformationEx.restype = ctypes.c_int
    return api


def detect_topology():
    """Prefer one logical CPU per physical core within the allowed affinity."""
    notes = []
    if sys.platform == "win32":
        api = windows_api()
        current = GroupAffinity()
        if not api.GetThreadGroupAffinity(api.GetCurrentThread(), ctypes.byref(current)):
            raise ctypes.WinError(ctypes.get_last_error())
        allowed = [i for i in range(ctypes.sizeof(ctypes.c_size_t) * 8)
                   if current.Mask & (1 << i)]
        size = ctypes.c_ulong(0)
        api.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))
        storage = ctypes.create_string_buffer(size.value)
        if not api.GetLogicalProcessorInformationEx(0, storage, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        blob, offset, core_id, mapping = storage.raw, 0, 0, {}
        while offset < size.value:
            relationship, length = struct.unpack_from("<II", blob, offset)
            if length < 8:
                raise RuntimeError("Invalid Windows topology record")
            if relationship == 0:
                groups = struct.unpack_from("<H", blob, offset + 30)[0]
                for group_index in range(groups):
                    ga = GroupAffinity.from_buffer_copy(
                        blob, offset + 32 + group_index * ctypes.sizeof(GroupAffinity))
                    if ga.Group == current.Group:
                        for cpu in allowed:
                            if ga.Mask & (1 << cpu):
                                mapping[cpu] = str(core_id)
                core_id += 1
            offset += length
        return allowed, mapping, "windows_group_affinity", int(current.Group), notes
    if hasattr(os, "sched_getaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        mapping = {}
        for cpu in allowed:
            root = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
            try:
                mapping[cpu] = (root.joinpath("physical_package_id").read_text().strip()
                                + ":" + root.joinpath("core_id").read_text().strip())
            except OSError:
                pass
        return allowed, mapping, "linux_sched_setaffinity", None, notes
    notes.append("Per-thread affinity and physical-core mapping unavailable on this OS.")
    return list(range(os.cpu_count() or 1)), {}, "unavailable", None, notes


def init_system_context(git_user):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}", git_user):
        raise ValueError("Specify your GitHub handle with --git-user")
    model = platform.processor() or platform.machine()
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            model = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    logical = os.cpu_count() or 1
    cpu_sig = f"{platform.machine()}::{model}::{logical}"
    seed_input = f"{git_user}::{cpu_sig}"
    seed = int(hashlib.sha256(seed_input.encode()).hexdigest()[:16], 16)
    allowed, mapping, affinity, group, notes = detect_topology()
    if len(allowed) < NUM_THREADS:
        raise RuntimeError("At least four allowed logical CPUs are required")
    selected, seen = [], set()
    for cpu in allowed:
        core = mapping.get(cpu)
        if core is not None and core not in seen:
            selected.append(cpu)
            seen.add(core)
    selected = (selected + [cpu for cpu in allowed if cpu not in selected])[:NUM_THREADS]
    distinct = len({mapping[cpu] for cpu in selected}) if all(
        cpu in mapping for cpu in selected) else None
    if distinct is not None and distinct < NUM_THREADS:
        notes.append("Fewer than four distinct physical cores available; SMT affects results.")
    line_size, line_source = detect_cache_line_size()
    if line_size != 64:
        notes.append("Detected cache line differs from the assignment's 64-byte geometry.")
    if line_source == "64_byte_fallback":
        notes.append("64-byte line size assumed because sysfs/sysctl is unavailable.")
    return {
        "git_username": git_user, "cpu_model": model, "cpu_signature": cpu_sig,
        "logical_cores": logical, "allowed_cpus": allowed, "selected_cpus": selected,
        "selected_physical_cores": [mapping.get(cpu) for cpu in selected],
        "distinct_selected_physical_cores": distinct, "affinity_backend": affinity,
        "windows_processor_group": group, "cache_line_size": line_size,
        "cache_line_size_source": line_source, "instance_seed": seed,
        "seed_input": seed_input, "platform": platform.platform(),
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "numba_version": numba.__version__, "llvmlite_version": llvmlite.__version__,
        "notes": notes,
    }


def pin_current_thread(context, cpu):
    if context["affinity_backend"] == "windows_group_affinity":
        api = windows_api()
        desired, previous, observed = GroupAffinity(), GroupAffinity(), GroupAffinity()
        desired.Mask, desired.Group = 1 << cpu, context["windows_processor_group"]
        if not api.SetThreadGroupAffinity(api.GetCurrentThread(), ctypes.byref(desired),
                                          ctypes.byref(previous)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not api.GetThreadGroupAffinity(api.GetCurrentThread(), ctypes.byref(observed)):
            raise ctypes.WinError(ctypes.get_last_error())
        if observed.Mask != desired.Mask or observed.Group != desired.Group:
            raise RuntimeError("Worker affinity verification failed")
        return {"pinned": True, "cpu": cpu, "group": int(observed.Group)}
    if context["affinity_backend"] == "linux_sched_setaffinity":
        os.sched_setaffinity(0, {cpu})  # Linux pid=0 means the calling thread.
        if os.sched_getaffinity(0) != {cpu}:
            raise RuntimeError("Worker affinity verification failed")
        return {"pinned": True, "cpu": cpu}
    return {"pinned": False, "cpu": None}


def context_switches():
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"voluntary": usage.ru_nvcsw, "involuntary": usage.ru_nivcsw}


def warmup_and_self_check():
    for seed in (0, 1, MASK):
        for iters in (0, 1, 2, 17, 1001):
            state, expected = seed, 0
            for _ in range(iters):
                state = (state * MULTIPLIER + INCREMENT) & MASK
                expected = (expected + state) & MASK
            if expected_counter(seed, iters) != expected:
                raise AssertionError("Affine reference disagrees with scalar Python")
            for kernel in (worker_strided_accumulate, worker_private_accumulate):
                buf = aligned_buffer(8)
                kernel(buf, 0, iters, np.uint64(seed))
                if int(buf[0]) != expected or np.any(buf[1:]):
                    raise AssertionError(f"Self-check failed for {kernel.__name__}")


def run_stride_benchmark(context, stride, iters, seeds, private=False):
    buffer = aligned_buffer((NUM_THREADS - 1) * stride + 1)
    kernel = worker_private_accumulate if private else worker_strided_accumulate
    records, errors, clock = [None] * NUM_THREADS, [], {}
    before = context_switches()

    def start_clock():
        clock["start"] = time.perf_counter_ns()

    barrier = threading.Barrier(NUM_THREADS + 1, action=start_clock, timeout=60)

    def worker(tid):
        try:
            affinity = pin_current_thread(context, context["selected_cpus"][tid])
            barrier.wait()
            enter = time.perf_counter_ns()
            kernel(buffer, tid * stride, iters, np.uint64(seeds[tid]))
            end = time.perf_counter_ns()
            records[tid] = {"tid": tid, "slot": tid * stride, "enter_ns": enter,
                            "end_ns": end, "affinity": affinity}
        except BaseException as exc:
            errors.append(f"Thread {tid}: {type(exc).__name__}: {exc}")
            barrier.abort()

    threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(NUM_THREADS)]
    for thread in threads:
        thread.start()
    try:
        barrier.wait()
    except threading.BrokenBarrierError:
        pass
    for thread in threads:
        thread.join()
    after = context_switches()
    if errors or any(r is None for r in records):
        raise RuntimeError("; ".join(errors) or "Worker synchronization failed")
    elapsed = max(r["end_ns"] for r in records) - clock["start"]
    expected = [expected_counter(seed, iters) for seed in seeds]
    actual = [int(buffer[tid * stride]) for tid in range(NUM_THREADS)]
    unused = np.ones(buffer.size, dtype=bool)
    unused[[tid * stride for tid in range(NUM_THREADS)]] = False
    parity = actual == expected and not np.any(buffer[unused])
    if not parity:
        raise AssertionError(f"Counter parity failed: {actual} != {expected}")
    base = int(buffer.ctypes.data)
    line = context["cache_line_size"]
    addresses = [base + tid * stride * 8 for tid in range(NUM_THREADS)]
    return {
        "stride": stride, "private": private, "elapsed_ns": elapsed,
        "ns_per_update": elapsed / (NUM_THREADS * iters),
        "wall_ns_per_thread_iteration": elapsed / iters,
        "context_switches": None if before is None else {
            key: after[key] - before[key] for key in before},
        "context_switch_scope": "unavailable_on_windows" if before is None else "process",
        "buffer_address": base, "alignment_bytes": 128,
        "slot_addresses": addresses, "cache_line_indices": [a // line for a in addresses],
        "expected_counters": expected, "actual_counters": actual, "parity_passed": bool(parity),
        "worker_records": records,
        "all_workers_overlap": max(r["enter_ns"] for r in records)
                               < min(r["end_ns"] for r in records),
    }


def native_function(module, name):
    for function in module.functions:
        if not function.is_declaration and function.name.startswith("_ZN") and name in function.name:
            return function
    raise RuntimeError(f"Cannot find native function {name} in LLVM IR")


def audit_llvm_and_parity(results, output):
    report = {}
    for kernel in (worker_strided_accumulate, worker_private_accumulate):
        llvm = kernel.inspect_llvm(kernel.signatures[0])
        filename = kernel.__name__ + ".ll"
        output.joinpath(filename).write_bytes(llvm.encode("utf-8"))
        module = binding.parse_assembly(llvm)
        module.verify()
        function = native_function(module, kernel.__name__)
        body = str(function)
        blocks = {block.name: block for block in function.blocks}
        edges = {name: re.findall(r'label %([\w.$-]+)', str(list(block.instructions)[-1]))
                 for name, block in blocks.items()}

        def in_cycle(start):
            todo, seen = list(edges[start]), set()
            while todo:
                node = todo.pop()
                if node == start:
                    return True
                if node not in seen:
                    seen.add(node)
                    todo.extend(edges.get(node, []))
            return False

        load_lines = [line.strip() for line in body.splitlines()
                      if re.search(r"\bload volatile i64\b", line)]
        store_lines = [line.strip() for line in body.splitlines()
                       if re.search(r"\bstore volatile i64\b", line)]
        loop_blocks = [name for name, block in blocks.items() if in_cycle(name)
                       and "load volatile i64" in str(block)
                       and "store volatile i64" in str(block)]
        native_only = not re.search(r"(?:call|invoke).*@(?:Py|_Py|numba_gil)", body)
        ordinary_stores = [line.strip() for line in body.splitlines()
                           if re.search(r"\bstore i64\b", line)]
        passed = bool(kernel.nopython_signatures) and native_only
        if kernel is worker_strided_accumulate:
            passed = passed and bool(load_lines and store_lines and loop_blocks)
        else:
            passed = passed and len(ordinary_stores) == 1 and not store_lines
        report[kernel.__name__] = {
            "passed": bool(passed), "nopython_signatures": [str(s) for s in kernel.nopython_signatures],
            "native_function": function.name, "native_body_has_no_python_calls": native_only,
            "volatile_load_examples": load_lines[:2], "volatile_store_examples": store_lines[:2],
            "volatile_loop_blocks": loop_blocks, "ordinary_i64_stores": ordinary_stores,
            "llvm_file": filename, "llvm_sha256": hashlib.sha256(llvm.encode()).hexdigest(),
        }
    report["all_layouts_bit_exact"] = all(
        run["parity_passed"] for result in results.values() for run in result["runs"])
    report["passed"] = (report["all_layouts_bit_exact"]
                        and report["worker_strided_accumulate"]["passed"]
                        and report["worker_private_accumulate"]["passed"])
    return report


def write_analysis(manifest, output):
    context, results, audit = manifest["system_context"], manifest["results"], manifest["audit"]
    s1, s8, s16, private = [results[k]["median_elapsed_ns"]
                           for k in ("stride_1", "stride_8", "stride_16", "private")]
    rows = []
    for key, result in results.items():
        switches = result["representative_run"]["context_switches"]
        cs = "Unavailable (Windows)" if switches is None else (
            f'{switches["voluntary"]} / {switches["involuntary"]}')
        rows.append(f'| {key} | {result["median_elapsed_ns"]:,} | '
                    f'{result["ns_per_update"]:.6f} | '
                    f'{result["median_elapsed_ns"] / s8:.4f} | {cs} |')
    llvm = audit["worker_strided_accumulate"]
    examples = "\n".join(llvm["volatile_load_examples"][:1] + llvm["volatile_store_examples"][:1])
    status = "Passed" if manifest["verification_passed"] else "Incomplete or diagnostic run"
    text = f"""# Cache Line and False Sharing Mechanics

Author: {context['git_username']}

## Experimental Setup

I used four threads on a **{context['cpu_model']}** with
{context['logical_cores']} logical processors. The threads were pinned to logical
CPUs `{context['selected_cpus']}`, corresponding to physical cores
`{context['selected_physical_cores']}`, using `{context['affinity_backend']}`.

Each thread performed **{manifest['iterations_per_thread']:,} updates** per run.
I repeated each configuration **{manifest['repeats']} times** and compared median
wall-clock times. The buffer was aligned to 128 bytes. The cache-line size was
{context['cache_line_size']} bytes, obtained through `{context['cache_line_size_source']}`.
On Windows, this is the assignment's 64-byte fallback rather than an OS cache-size query.

My instance seed was `{context['instance_seed']}`, derived from my GitHub handle
and CPU signature. Verification status: **{status}**.
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
{chr(10).join(rows)}

The slowdown of stride=1 relative to stride=8 was **{s1 / s8:.6f}x**.
Compared with the private baseline, stride=1 took **{s1 / private:.6f}x** as long,
while stride=8 took **{s8 / private:.6f}x** as long.

I calculated time per update as:

```text
ns_per_update = elapsed_ns / (4 * iterations_per_thread)
```

This measures the inverse aggregate throughput of all four threads rather than
the latency of one CPU instruction. The difference between stride=1 and stride=8
was **{(s1 - s8) / (4 * manifest['iterations_per_thread']):.6f} ns per aggregate update**.
Scheduling, CPU frequency and synchronization also affect this difference, so it
cannot be interpreted as the isolated cost of one RFO transaction.

Windows does not provide `resource.getrusage`. Context-switch counts are therefore
stored as `null`. On supported systems, the counters cover the process, including
thread setup and synchronization.

## Q3. 64-Byte and 128-Byte Padding

The median time was **{s8:,} ns** for stride=8 and **{s16:,} ns** for stride=16.
The ratio of the 64-byte-spacing time to the 128-byte-spacing time was
**{s8 / s16:.6f}x**.

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
{examples}
```

The audit found volatile loads and stores in cyclic CFG blocks
`{llvm['volatile_loop_blocks']}`. The compiled signature was
`{llvm['nopython_signatures']}`. The native function had no calls to the checked
Python API or GIL helpers: **{llvm['native_body_has_no_python_calls']}**.

The full IR is in [worker_strided_accumulate.ll](worker_strided_accumulate.ll).
Its SHA-256 is `{llvm['llvm_sha256']}`.
The private kernel contains **{len(audit['worker_private_accumulate']['ordinary_i64_stores'])}**
ordinary i64 store in its native function and no volatile stores.

I checked the final values against an independent reference that exponentiates
an affine transformation modulo 2^64. All configurations and repetitions matched:
**{audit['all_layouts_bit_exact']}**. Both kernels use the same LCG arithmetic.
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
a predetermined slowdown. System notes: {' '.join(context['notes'])}

Run timestamp, UTC: `{manifest['generated_at_utc']}`.

## References

- [Numba intrinsic API](https://numba.readthedocs.io/en/stable/extending/high-level.html#implementing-intrinsics)
- [LLVM volatile memory accesses](https://llvm.org/docs/LangRef.html#volatile-memory-accesses)
"""
    output.joinpath("ANALYSIS.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-user", default=os.getenv("GIT_USERNAME", GIT_USERNAME))
    parser.add_argument("--iterations", type=int, default=ITERATIONS_PER_THREAD)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if not (1 <= args.iterations < 1 << 63) or args.repeats < 1:
        parser.error("iterations and repeats must be positive; iterations < 2**63")
    args.output.mkdir(parents=True, exist_ok=True)
    context = init_system_context(args.git_user)
    print(f"CPU: {context['cpu_model']}", flush=True)
    print(f"Git: {args.git_user}; CPU affinity: {context['selected_cpus']}; "
          f"physical cores: {context['selected_physical_cores']}", flush=True)
    print("Compiling kernels and checking independent uint64 reference...", flush=True)
    warmup_and_self_check()
    seeds = [int(hashlib.sha256(f"{context['instance_seed']}::{tid}".encode())
                 .hexdigest()[:16], 16) for tid in range(NUM_THREADS)]
    variants = {f"stride_{stride}": (stride, False) for stride in (1, 4, 8, 16)}
    variants["private"] = (8, True)
    results = {name: {"runs": []} for name in variants}
    rng = random.Random(context["instance_seed"])
    orders = []
    for repeat in range(args.repeats):
        order = list(variants)
        rng.shuffle(order)
        orders.append(order)
        for name in order:
            stride, private = variants[name]
            run = run_stride_benchmark(context, stride, args.iterations, seeds, private)
            results[name]["runs"].append(run)
            print(f"[{repeat + 1}/{args.repeats}] {name:10s}: "
                  f"{run['elapsed_ns']:>14,} ns; parity=True", flush=True)
    for result in results.values():
        median = statistics.median(run["elapsed_ns"] for run in result["runs"])
        result["median_elapsed_ns"] = median
        result["ns_per_update"] = median / (NUM_THREADS * args.iterations)
        result["representative_run"] = min(
            result["runs"], key=lambda run: abs(run["elapsed_ns"] - median))
    audit = audit_llvm_and_parity(results, args.output)
    full = args.iterations == ITERATIONS_PER_THREAD
    pinned = all(run["worker_records"][tid]["affinity"]["pinned"]
                 for result in results.values() for run in result["runs"]
                 for tid in range(NUM_THREADS))
    topology_ok = context["distinct_selected_physical_cores"] in (None, NUM_THREADS)
    passed = (audit["passed"] and full and topology_ok and context["cache_line_size"] == 64
              and (pinned or context["affinity_backend"] == "unavailable"))
    manifest = {
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification_passed": bool(passed), "full_workload": full,
        "system_context": context, "num_threads": NUM_THREADS,
        "iterations_per_thread": args.iterations, "repeats": args.repeats,
        "thread_seeds": seeds, "execution_orders": orders,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "counter_recurrence": "state=(state*6364136223846793005+1442695040888963407) mod 2^64; counter+=state mod 2^64",
        "all_workers_pinned": pinned, "audit": audit, "results": results,
    }
    args.output.joinpath("cache_lab_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_analysis(manifest, args.output)
    print(f"verification_passed: {bool(passed)}; output: {args.output}", flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
