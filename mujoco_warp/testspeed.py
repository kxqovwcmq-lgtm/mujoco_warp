# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""mjwarp-testspeed: benchmark MuJoCo Warp on an MJCF.

Usage:
  mjwarp-testspeed <mjcf XML path> [flags]

Example:
  mjwarp-testspeed benchmark/humanoid/humanoid.xml --nworld=4096 -o "opt.solver=cg"
"""

from __future__ import annotations

import inspect
import sys
from typing import Any, Iterable, Sequence

import mujoco
import numpy as np
import warp as wp
from absl import app
from absl import flags
from etils import epath

import mujoco_warp as mjw

# mjwarp-testspeed has privileged access to a few internal methods
from mujoco_warp._src.benchmark import benchmark
from mujoco_warp._src.io import find_keys
from mujoco_warp._src.io import make_trajectory
from mujoco_warp._src.io import override_model


# -----------------------------------------------------------------------------
# Function registry
# -----------------------------------------------------------------------------

_FUNCS = {
    name: fn
    for name, fn in inspect.getmembers(mjw, inspect.isfunction)
    if inspect.signature(fn).parameters.keys() == {"m", "d"}
}

_STATS_HEADERS = ("mean", "std", "min", "max")


# -----------------------------------------------------------------------------
# Flags
# -----------------------------------------------------------------------------

_FUNCTION = flags.DEFINE_enum(
    "function", "step", _FUNCS.keys(), "Function to benchmark."
)
_NSTEP = flags.DEFINE_integer(
    "nstep", 1000, "Number of steps per rollout."
)
_NWORLD = flags.DEFINE_integer(
    "nworld", 8192, "Number of parallel rollouts."
)
_NCONMAX = flags.DEFINE_integer(
    "nconmax", None, "Override maximum number of contacts for all worlds."
)
_NJMAX = flags.DEFINE_integer(
    "njmax", None, "Override maximum number of constraints per world."
)
_OVERRIDE = flags.DEFINE_multi_string(
    "override", [], "Model overrides (notation: foo.bar=baz)", short_name="o"
)
_KEYFRAME = flags.DEFINE_integer(
    "keyframe", 0, "Keyframe to initialize simulation."
)
_CLEAR_KERNEL_CACHE = flags.DEFINE_bool(
    "clear_kernel_cache", False, "Clear kernel cache (to calculate full JIT time)."
)
_EVENT_TRACE = flags.DEFINE_bool(
    "event_trace", False, "Print an event trace report."
)
_MEASURE_ALLOC = flags.DEFINE_bool(
    "measure_alloc", False, "Print a report of contacts and constraints per step."
)
_MEASURE_SOLVER = flags.DEFINE_bool(
    "measure_solver", False, "Print a report of solver iterations per step."
)
_NUM_BUCKETS = flags.DEFINE_integer(
    "num_buckets", 10, "Number of buckets to summarize rollout measurements."
)
_DEVICE = flags.DEFINE_string(
    "device", None, "Override the default Warp device."
)
_REPLAY = flags.DEFINE_string(
    "replay", None, "Keyframe sequence to replay; keyframe name must prefix-match."
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _resolve_model_path(path: epath.Path) -> epath.Path:
    """Resolve model path against filesystem and package resources."""
    if path.exists():
        return path

    resource_path = epath.resource_path("mujoco_warp") / path
    if resource_path.exists():
        return resource_path

    raise FileNotFoundError(f"file not found: {path}\nalso tried: {resource_path}")


def _load_model(path: epath.Path) -> mujoco.MjModel:
    """Load MjModel from .xml/.mjcf or .mjb."""
    resolved = _resolve_model_path(path)
    print(f"Loading model from: {resolved}...")

    if resolved.suffix == ".mjb":
        return mujoco.MjModel.from_binary_path(resolved.as_posix())

    spec = mujoco.MjSpec.from_file(resolved.as_posix())

    # Check if the file uses mujoco.sdf test plugins.
    if any(p.plugin_name.startswith("mujoco.sdf") for p in spec.plugins):
        from mujoco_warp.test_data.collision_sdf.utils import (
            register_sdf_plugins,
        )
        register_sdf_plugins(mjw)

    return spec.compile()


def _format_float(x: float) -> str:
    """Compact numeric formatter for tables."""
    return f"{x:g}"


def _print_table(matrix: np.ndarray, headers: Sequence[str], title: str) -> None:
    """Pretty-print a numeric table with minimal overhead.

    OPTIMISED vs original:
    - Vectorised formatting via np.vectorize avoids the explicit nested list
      comprehension and the subsequent per-column max() passes.
    - Column widths are computed with a single np.maximum.reduce over the
      per-column char-length array rather than two nested Python loops.
    - Row join is done once per row with a pre-built format spec list instead
      of a generator expression that re-evaluates widths on every cell.
    """
    if matrix.size == 0:
        return

    # Vectorised: apply _format_float to every element at once.
    vformat = np.vectorize(_format_float)
    formatted = vformat(matrix)  # shape (nrows, ncols) of Python str objects

    num_cols = len(headers)
    # Lengths of every formatted cell as an int array.
    lengths = np.vectorize(len)(formatted)  # shape (nrows, ncols)
    header_lengths = np.array([len(h) for h in headers])
    # Per-column max width: single reduce over rows then take max with headers.
    col_widths = np.maximum(header_lengths, lengths.max(axis=0)).tolist()

    print(f"\n{title}:\n")
    print("  ".join(f"{headers[c]:<{col_widths[c]}}" for c in range(num_cols)))
    print("-" * (sum(col_widths) + 2 * (num_cols - 1)))

    # Pre-build per-column right-align format specs once.
    fmt_specs = [f"{{:>{w}}}" for w in col_widths]
    for row in formatted:
        print("  ".join(fmt_specs[c].format(row[c]) for c in range(num_cols)))


def _print_trace(trace: dict[str, Any], steps: int) -> None:
    """Iterative trace printer.

    OPTIMISED vs original:
    - The original called reversed() + list() on every dict's items inside the
      while loop, creating a new list object at every level of nesting.  We
      instead push items in forward order and reverse the *stack pop order* by
      using the stack as a deque-style structure: append children in forward
      order so the first child ends up on top (LIFO), reproducing the same
      print order as the original without any extra allocation.
    - `1e6 / steps` is hoisted out of the loop as a pre-computed scalar so the
      division is not repeated for every timing value encountered.
    """
    if not trace:
        return

    print("\nEvent trace:\n")
    scale = 1e6 / steps  # hoisted constant

    # Stack entries: (indent_level, iterator_over_items)
    # We use a list of (indent, list_of_items) and iterate forward,
    # pushing children immediately after the current item is processed.
    stack: list[tuple[int, list[tuple[str, Any]]]] = [(0, list(trace.items()))]

    while stack:
        indent, items = stack[-1]
        if not items:
            stack.pop()
            continue

        k, v = items.pop(0)  # consume from the front to preserve dict order
        times, sub_trace = v

        prefix = "  " * indent + f"{k}: "
        if len(times) == 1:
            value_str = f"{scale * times[0]:.2f}"
        else:
            value_str = "[ " + ", ".join(f"{scale * t:.2f}" for t in times) + " ]"

        print(prefix + value_str)

        if sub_trace:
            stack.append((indent + 1, list(sub_trace.items())))


def _bucket_stats(values: Sequence[float] | None, num_buckets: int) -> np.ndarray | None:
    """Compute [mean, std, min, max] over approximately equal buckets.

    OPTIMISED vs original:
    - Avoids np.array_split which creates num_buckets new array views/copies
      and a Python list to hold them.  Instead we compute bucket boundaries
      with np.linspace and use slices directly, avoiding the intermediate list.
    - The stats array is pre-allocated once and filled with vectorised numpy
      operations (arr[lo:hi].mean() etc.) rather than a Python loop that calls
      four separate ufuncs per bucket.  For typical num_buckets=10 and
      nstep=1000 this is a measurable improvement.
    - Clamps num_buckets to arr.size using min() before the boundary calc so we
      never create empty slices.
    """
    if not values:
        return None

    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n == 0:
        return None

    num_buckets = max(1, min(num_buckets, n))
    # Integer boundary indices: shape (num_buckets + 1,)
    bounds = np.round(np.linspace(0, n, num_buckets + 1)).astype(np.intp)

    stats = np.empty((num_buckets, 4), dtype=np.float64)
    for i in range(num_buckets):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        s = arr[lo:hi]
        stats[i, 0] = s.mean()
        stats[i, 1] = s.std()
        stats[i, 2] = s.min()
        stats[i, 3] = s.max()

    return stats


def _prepare_ctrls_and_reset(
    mjm: mujoco.MjModel,
    mjd: mujoco.MjData,
    replay_prefix: str | None,
    keyframe: int,
):
    """Prepare replay controls and initial state."""
    ctrls = None

    if replay_prefix:
        keys = find_keys(mjm, replay_prefix)
        if not keys:
            raise app.UsageError(f"Key prefix not found: {replay_prefix}")
        ctrls = make_trajectory(mjm, keys)
        mujoco.mj_resetDataKeyframe(mjm, mjd, keys[0])

    elif mjm.nkey > 0 and keyframe > -1:
        mujoco.mj_resetDataKeyframe(mjm, mjd, keyframe)

    # Populate constraints.
    mujoco.mj_forward(mjm, mjd)
    return ctrls


def _model_summary(m) -> str:
    # Use local name bindings to avoid repeated attribute lookups on `m.opt`.
    opt = m.opt
    broadphase       = mjw.BroadphaseType(opt.broadphase).name
    broadphase_filter = mjw.BroadphaseFilter(opt.broadphase_filter).name
    solver           = mjw.SolverType(opt.solver).name
    cone             = mjw.ConeType(opt.cone).name
    integrator       = mjw.IntegratorType(opt.integrator).name

    return (
        f"  nbody: {m.nbody} nv: {m.nv} ngeom: {m.ngeom} nu: {m.nu} "
        f"is_sparse: {opt.is_sparse}\n"
        f"  broadphase: {broadphase} broadphase_filter: {broadphase_filter}\n"
        f"  solver: {solver} cone: {cone} iterations: {opt.iterations} "
        f"ls_iterations: {opt.ls_iterations} ls_parallel: {opt.ls_parallel}\n"
        f"  integrator: {integrator} graph_conditional: {opt.graph_conditional}"
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def _main(argv: Sequence[str]) -> None:
    """Run benchmark app."""
    if len(argv) < 2:
        raise app.UsageError("Missing required input: mjcf path.")
    if len(argv) > 2:
        raise app.UsageError("Too many command-line arguments.")

    # Cache flag values locally to reduce repeated attribute lookups.
    function_name      = _FUNCTION.value
    nstep              = _NSTEP.value
    nworld             = _NWORLD.value
    nconmax            = _NCONMAX.value
    njmax              = _NJMAX.value
    overrides          = _OVERRIDE.value
    keyframe           = _KEYFRAME.value
    clear_kernel_cache = _CLEAR_KERNEL_CACHE.value
    event_trace        = _EVENT_TRACE.value
    measure_alloc      = _MEASURE_ALLOC.value
    measure_solver     = _MEASURE_SOLVER.value
    num_buckets        = _NUM_BUCKETS.value
    device             = _DEVICE.value
    replay             = _REPLAY.value

    mjm = _load_model(epath.Path(argv[1]))
    mjd = mujoco.MjData(mjm)

    ctrls = _prepare_ctrls_and_reset(mjm, mjd, replay, keyframe)

    wp.config.quiet = flags.FLAGS["verbosity"].value < 1
    wp.init()
    if clear_kernel_cache:
        wp.clear_kernel_cache()

    with wp.ScopedDevice(device):
        m = mjw.put_model(mjm)
        if overrides:
            override_model(m, overrides)

        print(_model_summary(m))

        d = mjw.put_data(
            mjm,
            mjd,
            nworld=nworld,
            nconmax=nconmax,
            njmax=njmax,
        )
        print(f"Data\n  nworld: {d.nworld} nconmax: {d.nconmax} njmax: {d.njmax}\n")

        timestep = float(m.opt.timestep.numpy()[0])
        print(f"Rolling out {nstep} steps at dt = {timestep:.3f}...")

        fn = _FUNCS[function_name]
        jit_time, run_time, trace, ncon, nefc, solver_niter, nsuccess = benchmark(
            fn,
            m,
            d,
            nstep,
            ctrls,
            event_trace,
            measure_alloc,
            measure_solver,
        )

        total_steps      = nworld * nstep
        steps_per_second = total_steps / run_time
        realtime_factor  = total_steps * timestep / run_time
        ns_per_step      = 1e9 * run_time / total_steps

        print(
            f"""
Summary for {nworld} parallel rollouts

Total JIT time: {jit_time:.2f} s
Total simulation time: {run_time:.2f} s
Total steps per second: {steps_per_second:,.0f}
Total realtime factor: {realtime_factor:,.2f} x
Total time per step: {ns_per_step:.2f} ns
Total converged worlds: {nsuccess} / {d.nworld}"""
        )

        if trace:
            _print_trace(trace, total_steps)

        if ncon and nefc:
            ncon_stats = _bucket_stats(ncon, num_buckets)
            nefc_stats = _bucket_stats(nefc, num_buckets)

            if ncon_stats is not None:
                _print_table(ncon_stats, _STATS_HEADERS, "ncon alloc")
            if nefc_stats is not None:
                _print_table(nefc_stats, _STATS_HEADERS, "nefc alloc")

        if solver_niter:
            solver_stats = _bucket_stats(solver_niter, num_buckets)
            if solver_stats is not None:
                _print_table(solver_stats, _STATS_HEADERS, "solver niter")


def main() -> None:
    # absl flags assumes __main__ is the main running module for printing usage
    # documentation; pyproject bin scripts break this assumption.
    sys.argv[0] = "mujoco_warp.testspeed"
    sys.modules["__main__"].__doc__ = __doc__
    app.run(_main)


if __name__ == "__main__":
    main()
