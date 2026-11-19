import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI
from numba import cuda, float64


N = 256
TILE = 16
SHARED = 18
STEPS = 400
H = 1.0 / 513.0
LAMBDA = 0.15
OPPOSITE = (1, 0, 3, 2)  # up, down, left, right


def init_distributed_context(github_user):
    world = MPI.COMM_WORLD
    rank = world.Get_rank()
    if world.Get_size() != 4:
        raise RuntimeError("Launch with mpirun -np 4.")

    cuda.select_device(0)
    uuid = None
    if rank == 0:
        uuid = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            text=True,
        ).splitlines()[0].strip()
    uuid = world.bcast(uuid, root=0)

    digest = hashlib.sha256(f"{github_user}::{uuid}".encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16)
    cart = world.Create_cart([2, 2], periods=[False, False], reorder=False)
    coords = cart.Get_coords(rank)
    neighbors = cart.Shift(0, 1) + cart.Shift(1, 1)
    print(f"Rank {rank} coords={coords} neighbors={neighbors}", flush=True)
    return cart, coords, neighbors, uuid, seed


def initialize_subdomains(coords, neighbors, seed):
    y = ((coords[0] * N + np.arange(N + 2)) * H)[:, None]
    x = ((coords[1] * N + np.arange(N + 2)) * H)[None, :]
    kappa = 0.16 + 0.08 * np.sin(2 * np.pi * x) * np.cos(4 * np.pi * y)

    rng = np.random.default_rng(seed)
    u = np.zeros((N + 2, N + 2), dtype=np.float64)
    for lo, hi in ((0.23, 0.42), (0.58, 0.77)):
        cx = rng.uniform(lo, hi)
        cy = rng.uniform(0.25, 0.75)
        amp = rng.uniform(0.8, 1.3)
        sigma = rng.uniform(0.065, 0.09)
        u += amp * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))

    source = 0.004 * np.exp(-((x - 0.50)**2 + (y - 0.55)**2) / (2 * 0.045**2))
    for neighbor, edge in zip(neighbors, (u[0, :], u[-1, :], u[:, 0], u[:, -1])):
        if neighbor == MPI.PROC_NULL:
            edge.fill(0.0)
    return u, kappa, source


@cuda.jit
def pack_halos_kernel(u, send):
    t = cuda.grid(1)
    if t < N:
        i = t + 1
        send[t] = u[1, i]
        send[N + t] = u[N, i]
        send[2 * N + t] = u[i, 1]
        send[3 * N + t] = u[i, N]


@cuda.jit
def unpack_halos_kernel(u, recv):
    t = cuda.grid(1)
    if t < N:
        i = t + 1
        u[0, i] = recv[t]
        u[N + 1, i] = recv[N + t]
        u[i, 0] = recv[2 * N + t]
        u[i, N + 1] = recv[3 * N + t]


@cuda.jit
def tiled_stencil_kernel(u, out, kappa, source, dt, r_start, r_end, c_start, c_end):
    tile = cuda.shared.array((SHARED, SHARED), dtype=float64)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    sx, sy = tx + 1, ty + 1
    r = r_start + cuda.blockIdx.y * TILE + ty
    c = c_start + cuda.blockIdx.x * TILE + tx

    # Clip reads to the update box plus one cell on each side.
    # This keeps the interior stream away from the incoming MPI halos.
    rr = min(r, r_end + 1)
    cc = min(c, c_end + 1)
    tile[sy, sx] = u[rr, cc]
    if ty == 0:
        tile[0, sx] = u[r - 1, cc]
    if ty == TILE - 1:
        tile[SHARED - 1, sx] = u[min(r + 1, r_end + 1), cc]
    if tx == 0:
        tile[sy, 0] = u[rr, c - 1]
    if tx == TILE - 1:
        tile[sy, SHARED - 1] = u[rr, min(c + 1, c_end + 1)]

    # The five-point stencil reads the four axial neighbors.
    cuda.syncthreads()

    if r <= r_end and c <= c_end:
        center = tile[sy, sx]
        k = kappa[r, c]
        north = 0.5 * (k + kappa[r - 1, c]) * (tile[sy - 1, sx] - center)
        south = 0.5 * (k + kappa[r + 1, c]) * (tile[sy + 1, sx] - center)
        west = 0.5 * (k + kappa[r, c - 1]) * (tile[sy, sx - 1] - center)
        east = 0.5 * (k + kappa[r, c + 1]) * (tile[sy, sx + 1] - center)
        diffusion = ((north + south) + (west + east)) / (H * H)
        out[r, c] = center + dt * (diffusion - LAMBDA * center**3 + source[r, c])


class HaloExchange:
    def __init__(self, cart, neighbors):
        self.cart = cart
        self.neighbors = neighbors
        self.d_send = cuda.device_array(4 * N, dtype=np.float64)
        self.d_recv = cuda.device_array(4 * N, dtype=np.float64)
        self.h_send = cuda.pinned_array(4 * N, dtype=np.float64)
        self.h_recv = cuda.pinned_array(4 * N, dtype=np.float64)
        self.h_recv.fill(0.0)
        self.parts = [slice(i * N, (i + 1) * N) for i in range(4)]


def exchange_halos_async(u, halo, stream, start_event, end_event):
    start_event.record(stream)
    pack_halos_kernel[4, 64, stream](u, halo.d_send)
    halo.d_send.copy_to_host(halo.h_send, stream=stream)
    end_event.record(stream)
    stream.synchronize()

    requests = [
        halo.cart.Irecv(halo.h_recv[part], source=peer, tag=OPPOSITE[i])
        for i, (part, peer) in enumerate(zip(halo.parts, halo.neighbors))
    ]
    requests += [
        halo.cart.Isend(halo.h_send[part], dest=peer, tag=i)
        for i, (part, peer) in enumerate(zip(halo.parts, halo.neighbors))
    ]
    return requests


def launch_box(u, out, kappa, source, dt, box, stream):
    r0, r1, c0, c1 = box
    grid = ((c1 - c0 + TILE) // TILE, (r1 - r0 + TILE) // TILE)
    tiled_stencil_kernel[grid, (TILE, TILE), stream](
        u, out, kappa, source, dt, r0, r1, c0, c1
    )


def step_overlapped(u, out, kappa, source, dt, halo, streams, events, overlap):
    boundary, interior = streams
    (pack_start, pack_end), (inner_start, inner_end), (ring_start, ring_end) = events
    requests = exchange_halos_async(u, halo, boundary, pack_start, pack_end)
    inner_box = (2, N - 1, 2, N - 1)

    if overlap:
        inner_start.record(interior)
        launch_box(u, out, kappa, source, dt, inner_box, interior)
        inner_end.record(interior)

    MPI.Request.Waitall(requests)
    ring_start.record(boundary)
    halo.d_recv.copy_to_device(halo.h_recv, stream=boundary)
    unpack_halos_kernel[4, 64, boundary](u, halo.d_recv)
    for box in ((1, 1, 1, N), (N, N, 1, N), (2, N - 1, 1, 1), (2, N - 1, N, N)):
        launch_box(u, out, kappa, source, dt, box, boundary)
    ring_end.record(boundary)

    if not overlap:
        inner_start.record(boundary)
        launch_box(u, out, kappa, source, dt, inner_box, boundary)
        inner_end.record(boundary)

    boundary.synchronize()
    if overlap:
        interior.synchronize()
    return [cuda.event_elapsed_time(start, end) for start, end in events]


def run_mode(cart, initial, kappa, source, dt, halo, overlap):
    name = "overlapped" if overlap else "serialized"
    streams = (cuda.stream(), cuda.stream())
    events = [(cuda.event(), cuda.event()) for _ in range(3)]
    u = cuda.to_device(initial)
    out = cuda.to_device(initial)
    host_u = cuda.pinned_array(initial.shape, dtype=np.float64)
    host_prev = cuda.pinned_array(initial.shape, dtype=np.float64)
    owned = np.s_[1:-1, 1:-1]

    # Compile and warm up before measuring, then restore the initial state.
    step_overlapped(u, out, kappa, source, dt, halo, streams, events, overlap)
    u.copy_to_device(initial)
    out.copy_to_device(initial)

    trace = []
    phases_ms = np.zeros(3, dtype=np.float64)
    cart.Barrier()
    start_time = MPI.Wtime()

    for step in range(STEPS + 1):
        if step:
            phases_ms += step_overlapped(
                u, out, kappa, source, dt, halo, streams, events, overlap
            )
            u, out = out, u

        if step % 100 == 0:
            u.copy_to_host(host_u, stream=streams[0])
            out.copy_to_host(host_prev, stream=streams[0])
            streams[0].synchronize()
            local = np.array([
                host_u[owned].sum() * H * H,
                np.square(host_u[owned] - host_prev[owned]).sum(),
            ], dtype=np.float64)
            reduced = np.empty_like(local)
            cart.Allreduce(local, reduced, op=MPI.SUM)
            l2 = math.sqrt(reduced[1])
            trace.append(dict(step=step, energy=float(reduced[0]), l2_residual=l2))
            if cart.Get_rank() == 0:
                print(f"{name} step={step:3d} E={reduced[0]:.12e} L2={l2:.9e}", flush=True)

    wall_ms = cart.allreduce((MPI.Wtime() - start_time) * 1000, op=MPI.MAX)
    max_phases_ms = np.empty_like(phases_ms)
    cart.Allreduce(phases_ms, max_phases_ms, op=MPI.MAX)
    timing_names = ("pack_d2h_ms", "interior_ms", "h2d_unpack_ring_ms")
    timings = dict(zip(timing_names, map(float, max_phases_ms)))
    return host_u[owned].copy(), trace, wall_ms, timings


def run_solver(github_user):
    cart, coords, neighbors, uuid, seed = init_distributed_context(github_user)
    rank = cart.Get_rank()
    initial, kappa, source = initialize_subdomains(coords, neighbors, seed)
    kappa_max = cart.allreduce(float(kappa.max()), op=MPI.MAX)
    dt_max = H * H / (4 * kappa_max)
    dt = 0.8 * dt_max

    if rank == 0:
        print(f"GitHub={github_user} GPU UUID={uuid} INSTANCE_SEED={seed}", flush=True)
        print(f"mpi_ranks={cart.Get_size()} proc_grid={cart.Get_topo()[0]} "
              f"kappa_max={kappa_max:.12g} dx={H:.12g} "
              f"dt={dt:.12e} dt_max={dt_max:.12e}", flush=True)

    d_kappa = cuda.to_device(kappa)
    d_source = cuda.to_device(source)
    halo = HaloExchange(cart, neighbors)
    a, trace_a, serial_ms, cuda_a = run_mode(
        cart, initial, d_kappa, d_source, dt, halo, overlap=False
    )
    b, trace_b, overlap_ms, cuda_b = run_mode(
        cart, initial, d_kappa, d_source, dt, halo, overlap=True
    )

    local_diff = float(np.max(np.abs(a - b)))
    max_diff = cart.allreduce(
        local_diff if math.isfinite(local_diff) else math.inf, op=MPI.MAX
    )
    if rank != 0:
        return

    ptx = "\n".join(tiled_stencil_kernel.inspect_asm().values())
    opcodes = ("st.shared.f64", "ld.shared.f64", "bar.sync", "st.shared", "ld.shared")
    counts = {
        op: len(re.findall(r"\b" + re.escape(op) + r"(?=[.\s])", ptx))
        for op in opcodes
    }
    audit = all(counts[op] > 0 for op in ("st.shared", "ld.shared", "bar.sync"))
    manifest = dict(
        github_username=github_user,
        gpu_uuid=uuid,
        instance_seed=seed,
        mpi_ranks=cart.Get_size(),
        cart_dims=[2, 2],
        steps=STEPS,
        dx=H,
        dy=H,
        kappa_max=kappa_max,
        dt=dt,
        dt_max=dt_max,
        ptx_counts=counts,
        ptx_audit_passed=audit,
        parity_max_diff=max_diff,
        serialized_ms=serial_ms,
        overlapped_ms=overlap_ms,
        speedup=serial_ms / overlap_ms,
        cuda_event_ms=dict(serialized=cuda_a, overlapped=cuda_b),
        energy_trace=dict(serialized=trace_a, overlapped=trace_b),
    )

    print(f"serialized_ms={serial_ms:.3f} overlapped_ms={overlap_ms:.3f} "
          f"speedup={manifest['speedup']:.4f}x", flush=True)
    print(f"CUDA event phase totals (ms): {manifest['cuda_event_ms']}", flush=True)
    print(f"PTX counts: {counts}", flush=True)
    print(f"ptx_audit_passed={audit} parity_max_diff={max_diff:.3e}", flush=True)

    if not audit or max_diff >= 1e-12:
        raise RuntimeError("PTX audit or final-state parity check failed.")

    Path("submission_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print("Wrote submission_manifest.json", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}", sys.argv[1]):
        raise SystemExit("Pass your GitHub username as the single argument.")

    try:
        run_solver(sys.argv[1])
    except Exception:
        import traceback
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
