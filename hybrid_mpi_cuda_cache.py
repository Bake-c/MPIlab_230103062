import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI
from numba import cuda, float64, njit, types
from numba.extending import intrinsic

N, TILE, APRON_TILE = 256, 16, 18
STENCIL_STEPS, SHM_ITERS = 300, 40_000_000
H, LAMBDA = 1.0 / 513.0, 0.15
OPPOSITE = (1, 0, 3, 2)


def init_hybrid_context(github_user):
    world = MPI.COMM_WORLD
    rank = world.Get_rank()
    if world.Get_size() != 4:
        raise RuntimeError("Start this program with four MPI ranks.")
    cuda.select_device(0)
    gpu_uuid = None
    cpu_signature = f"{platform.machine()}::{os.cpu_count()}"
    if rank == 0:
        gpu_uuid = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
        ).strip().splitlines()[0].strip()
    gpu_uuid = world.bcast(gpu_uuid, root=0)
    seed_material = f"{github_user.strip()}::{gpu_uuid}::{cpu_signature}"
    seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
    cart = world.Create_cart([2, 2], periods=[False, False], reorder=False)
    coords = cart.Get_coords(rank)
    peers = cart.Shift(0, 1) + cart.Shift(1, 1)
    node = world.Split_type(MPI.COMM_TYPE_SHARED, key=rank)
    if node.Get_size() != 4:
        raise RuntimeError("The false-sharing benchmark requires all ranks on one node.")

    line_file = Path("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size")
    line_bytes = int(line_file.read_text().strip()) if line_file.exists() else 64
    if line_bytes != 64:
        raise RuntimeError(f"This lab assumes a 64-byte host cache line, found {line_bytes}.")
    win = MPI.Win.Allocate_shared(64 * 8 + 63 if node.Get_rank() == 0 else 0,
                                  disp_unit=8, comm=node)
    mem, _ = win.Shared_query(0)
    raw = np.frombuffer(mem, dtype=np.uint8)
    offset = (-raw.ctypes.data) % 64
    if len(set(node.allgather(offset))) != 1:
        raise RuntimeError("The aligned offset differs across the shared mappings.")
    counters = raw[offset:offset + 64 * 8].view(np.uint64)
    assert counters.size == 64 and counters.ctypes.data % 64 == 0
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    available_cpus = sorted(set().union(*node.allgather(cpus)))
    print(f"Rank {rank} coords={coords} neighbors={peers} allowed_cpus={cpus}", flush=True)
    return cart, coords, peers, node, win, counters, gpu_uuid, cpu_signature, line_bytes, seed, available_cpus


@intrinsic
def shared_increment(typingctx, array, index):
    sig = types.void(array, index)

    def codegen(context, builder, signature, args):
        data = context.make_array(signature.args[0])(context, builder, value=args[0]).data
        ptr = builder.gep(data, [args[1]])
        # Unique writer per address. Relaxed accesses force a load and store on
        # each iteration without a locked fetch-add on the Colab x86 host.
        old = builder.load_atomic(ptr, "monotonic", 8)
        new = builder.add(old, old.type(1))
        builder.store_atomic(new, ptr, "monotonic", 8)
        return context.get_dummy_value()

    return sig, codegen


@njit(nogil=True)
def host_shm_accumulate(shm_view, rank, stride, iterations):
    slot = rank * stride
    for _ in range(iterations):
        shared_increment(shm_view, slot)


def benchmark_shm_strides(node, win, counters):
    rank = node.Get_rank()
    host_shm_accumulate(np.zeros(64, dtype=np.uint64), rank, 16, 1)
    llvm = host_shm_accumulate.inspect_llvm(host_shm_accumulate.signatures[0])
    counts = {
        "atomic_load_i64": len(re.findall(r"\bload atomic i64\b", llvm)),
        "atomic_store_i64": len(re.findall(r"\bstore atomic i64\b", llvm)),
    }
    if not all(counts.values()):
        raise RuntimeError(f"CPU LLVM audit failed: {counts}")
    timings = {}
    win.Lock_all()
    try:
        for stride in (1, 8, 16):
            if rank == 0:
                counters.fill(0)
            win.Sync()
            node.Barrier()
            win.Sync()
            start = MPI.Wtime()
            host_shm_accumulate(counters, rank, stride, SHM_ITERS)
            elapsed = (MPI.Wtime() - start) * 1000
            win.Sync()
            node.Barrier()
            win.Sync()
            slots = [int(counters[i * stride]) for i in range(4)]
            if slots != [SHM_ITERS] * 4:
                raise RuntimeError(f"Shared counter verification failed: {slots}")
            timings[str(stride)] = node.allreduce(elapsed, op=MPI.MAX)
            if rank == 0:
                print(f"shm stride={stride:2d} spacing={stride * 8:3d} B "
                      f"wall_ms={timings[str(stride)]:.6f} counters={slots}", flush=True)
    finally:
        win.Unlock_all()
    return timings, counts


def initialize_fields(coords, peers, seed):
    y = ((coords[0] * N + np.arange(N + 2)) * H)[:, None]
    x = ((coords[1] * N + np.arange(N + 2)) * H)[None, :]
    kappa = 0.16 + 0.08 * np.sin(2 * np.pi * x) * np.cos(4 * np.pi * y)
    rng = np.random.default_rng(seed)
    u = np.zeros((N + 2, N + 2), dtype=np.float64)
    for lo, hi in ((0.23, 0.42), (0.58, 0.77)):
        cx, cy = rng.uniform(lo, hi), rng.uniform(0.25, 0.75)
        amp, sigma = rng.uniform(0.8, 1.3), rng.uniform(0.065, 0.09)
        u += amp * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    source = 0.004 * np.exp(-((x - .50)**2 + (y - .55)**2) / (2 * .045**2))
    for peer, edge in zip(peers, (u[0, :], u[-1, :], u[:, 0], u[:, -1])):
        if peer == MPI.PROC_NULL:
            edge.fill(0)
    return u, kappa, source


@cuda.jit
def pack_halos_kernel(u, send):
    t = cuda.grid(1)
    if t < N:
        i = t + 1
        send[t], send[N + t] = u[1, i], u[N, i]
        send[2 * N + t], send[3 * N + t] = u[i, 1], u[i, N]


@cuda.jit
def unpack_halos_kernel(u, recv):
    t = cuda.grid(1)
    if t < N:
        i = t + 1
        u[0, i], u[N + 1, i] = recv[t], recv[N + t]
        u[i, 0], u[i, N + 1] = recv[2 * N + t], recv[3 * N + t]


@cuda.jit
def tiled_stencil_kernel(u, out, kappa, source, dt, r0, r1, c0, c1):
    tile = cuda.shared.array((APRON_TILE, APRON_TILE), dtype=float64)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    sx, sy = tx + 1, ty + 1
    r = r0 + cuda.blockIdx.y * TILE + ty
    c = c0 + cuda.blockIdx.x * TILE + tx
    
    rr, cc = min(r, r1 + 1), min(c, c1 + 1)
    tile[sy, sx] = u[rr, cc]
    if ty == 0:
        tile[0, sx] = u[r - 1, cc]
    if ty == TILE - 1:
        tile[17, sx] = u[min(r + 1, r1 + 1), cc]
    if tx == 0:
        tile[sy, 0] = u[rr, c - 1]
    if tx == TILE - 1:
        tile[sy, 17] = u[rr, min(c + 1, c1 + 1)]
    cuda.syncthreads()

    if r <= r1 and c <= c1:
        center, k = tile[sy, sx], kappa[r, c]
        north = 0.5 * (k + kappa[r - 1, c]) * (tile[sy - 1, sx] - center)
        south = 0.5 * (k + kappa[r + 1, c]) * (tile[sy + 1, sx] - center)
        west = 0.5 * (k + kappa[r, c - 1]) * (tile[sy, sx - 1] - center)
        east = 0.5 * (k + kappa[r, c + 1]) * (tile[sy, sx + 1] - center)
        diffusion = ((north + south) + (west + east)) / (H * H)
        out[r, c] = center + dt * (diffusion - LAMBDA * center**3 + source[r, c])


def launch_box(u, out, kappa, source, dt, box, stream):
    r0, r1, c0, c1 = box
    blocks = ((c1 - c0 + TILE) // TILE, (r1 - r0 + TILE) // TILE)
    tiled_stencil_kernel[blocks, (TILE, TILE), stream](
        u, out, kappa, source, dt, r0, r1, c0, c1
    )


class HaloBuffers:
    def __init__(self, cart, peers):
        self.cart, self.peers = cart, peers
        self.ds = cuda.device_array(4 * N, dtype=np.float64)
        self.dr = cuda.device_array(4 * N, dtype=np.float64)
        self.hs = cuda.pinned_array(4 * N, dtype=np.float64)
        self.hr = cuda.pinned_array(4 * N, dtype=np.float64)
        self.hr.fill(0)  # MPI.PROC_NULL leaves its receive segment unchanged.
        self.parts = [slice(i * N, (i + 1) * N) for i in range(4)]


def step_overlapped(u, out, k, s, dt, halo, stream_boundary, stream_interior, overlap):
    pack_halos_kernel[4, 64, stream_boundary](u, halo.ds)
    halo.ds.copy_to_host(halo.hs, stream=stream_boundary)
    stream_boundary.synchronize()
    requests = [
        halo.cart.Irecv(halo.hr[part], source=peer, tag=OPPOSITE[i])
        for i, (part, peer) in enumerate(zip(halo.parts, halo.peers))
    ]
    requests += [
        halo.cart.Isend(halo.hs[part], dest=peer, tag=i)
        for i, (part, peer) in enumerate(zip(halo.parts, halo.peers))
    ]
    interior_box = (2, N - 1, 2, N - 1)
    if overlap:
        launch_box(u, out, k, s, dt, interior_box, stream_interior)

    MPI.Request.Waitall(requests)
    halo.dr.copy_to_device(halo.hr, stream=stream_boundary)
    unpack_halos_kernel[4, 64, stream_boundary](u, halo.dr)
    for box in ((1, 1, 1, N), (N, N, 1, N), (2, N - 1, 1, 1), (2, N - 1, N, N)):
        launch_box(u, out, k, s, dt, box, stream_boundary)
    if not overlap:
        launch_box(u, out, k, s, dt, interior_box, stream_boundary)
    stream_boundary.synchronize()
    if overlap:
        stream_interior.synchronize()


def run_mode(cart, initial, k, s, dt, halo, overlap):
    mode = "overlapped" if overlap else "serialized"
    sb, si = cuda.stream(), cuda.stream()
    u, out = cuda.to_device(initial), cuda.to_device(initial)
    host = cuda.pinned_array(initial.shape, dtype=np.float64)
    step_overlapped(u, out, k, s, dt, halo, sb, si, overlap)
    u.copy_to_device(initial)
    out.copy_to_device(initial)
    trace = []
    cart.Barrier()
    start = MPI.Wtime()
    for step in range(STENCIL_STEPS + 1):
        if step:
            step_overlapped(u, out, k, s, dt, halo, sb, si, overlap)
            u, out = out, u
        if step % 100 == 0:
            u.copy_to_host(host, stream=sb)
            sb.synchronize()
            energy = cart.allreduce(float(host[1:-1, 1:-1].sum() * H * H), op=MPI.SUM)
            trace.append({"step": step, "energy": energy})
            if cart.Get_rank() == 0:
                print(f"{mode} step={step:3d} E={energy:.12e}", flush=True)
    wall_ms = cart.allreduce((MPI.Wtime() - start) * 1000, op=MPI.MAX)
    return host[1:-1, 1:-1].copy(), trace, wall_ms


def main(github_user):
    cart, coords, peers, node, win, counters, uuid, cpu_sig, line, seed, cpus = init_hybrid_context(github_user)
    rank = cart.Get_rank()
    if rank == 0:
        print(f"GitHub={github_user} CPU={cpu_sig} GPU UUID={uuid} INSTANCE_SEED={seed} "
              f"mpi_ranks=4 proc_grid=[2, 2] line_bytes={line} allowed_cpus={cpus}", flush=True)
        if cpus and len(cpus) < 4:
            print("Fewer than four logical CPUs are available. Interpret CPU timings as oversubscribed.", flush=True)

    shm_ms, llvm = benchmark_shm_strides(node, win, counters)
    del counters
    win.Free()
    node.Free()
    if rank == 0:
        print(f"CPU LLVM counts={llvm} llvm_audit_passed=True", flush=True)
        print(f"shm_slowdown_stride1_over_stride8={shm_ms['1']/shm_ms['8']:.6f} "
              f"shm_ratio_stride8_over_stride16={shm_ms['8']/shm_ms['16']:.6f}", flush=True)

    initial, k, source = initialize_fields(coords, peers, seed)
    kappa_max = cart.allreduce(float(k.max()), op=MPI.MAX)
    dt_max = H**2 / (4 * kappa_max)
    dt = 0.8 * dt_max
    if rank == 0:
        print(f"kappa_max={kappa_max:.12g} dx={H:.12g} dt={dt:.12e} dt_max={dt_max:.12e}", flush=True)
    dk, ds = cuda.to_device(k), cuda.to_device(source)
    halos = HaloBuffers(cart, peers)
    a, energy_a, serial = run_mode(cart, initial, dk, ds, dt, halos, False)
    b, energy_b, overlap = run_mode(cart, initial, dk, ds, dt, halos, True)
    local_diff = float(np.abs(a - b).max())
    max_diff = cart.allreduce(local_diff if math.isfinite(local_diff) else math.inf, op=MPI.MAX)
    if rank != 0:
        return

    ptx = "\n".join(tiled_stencil_kernel.inspect_asm().values())
    names = ("st.shared.f64", "ld.shared.f64", "st.shared", "ld.shared", "bar.sync")
    counts = {op: len(re.findall(r"\b" + re.escape(op) + r"(?=[.\s])", ptx)) for op in names}
    audit = all(counts[op] for op in ("st.shared", "ld.shared", "bar.sync"))
    energy_match = all(math.isclose(a["energy"], b["energy"], abs_tol=1e-12, rel_tol=0)
                       for a, b in zip(energy_a, energy_b))
    print(f"GPU PTX counts={counts} ptx_audit_passed={audit}", flush=True)
    print(f"gpu_serialized_ms={serial:.6f} gpu_overlapped_ms={overlap:.6f} "
          f"gpu_speedup={serial/overlap:.6f} max_diff={max_diff:.3e} energy_match={energy_match}", flush=True)
    if not audit or not max_diff < 1e-12 or not energy_match:
        raise RuntimeError("PTX audit, final field parity, or global energy check failed.")

    manifest = dict(
        github_username=github_user, gpu_uuid=uuid, cpu_signature=cpu_sig,
        allowed_cpus=cpus, cpu_cache_line_bytes=line, instance_seed=seed,
        mpi_ranks=4, cart_dims=[2, 2], shared_slots=64, iterations_per_rank=SHM_ITERS,
        shm_stride_ms=shm_ms, shm_slowdown_stride1_over_stride8=shm_ms["1"]/shm_ms["8"],
        shm_ratio_stride8_over_stride16=shm_ms["8"]/shm_ms["16"],
        llvm_counts=llvm, llvm_audit_passed=True, steps=STENCIL_STEPS,
        dx=H, dy=H, kappa_max=kappa_max, dt=dt, dt_max=dt_max,
        gpu_serialized_ms=serial, gpu_overlapped_ms=overlap, gpu_speedup=serial/overlap,
        ptx_counts=counts, ptx_audit_passed=audit, max_diff=max_diff,
        energy_match=energy_match, energy_trace={"serialized": energy_a, "overlapped": energy_b}
    )
    Path("hybrid_lab_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print("Wrote hybrid_lab_manifest.json", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or not sys.argv[1].strip():
        raise SystemExit("Pass your exact GitHub username as the sole argument.")
    try:
        main(sys.argv[1])
    except Exception:
        import traceback
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
