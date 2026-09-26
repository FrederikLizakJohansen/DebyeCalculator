"""
Benchmark an earlier DebyeCalculator release against the current source tree and produce the paper figures.

    python paper/benchmark/run_benchmark.py

runs G(r) timings on the CPU and, when available, on the CUDA GPU, for both versions, compares the
calculated patterns, and writes figures, a summary table and the raw results to paper/benchmark/benchmark_output/.
Each measurement runs in its own subprocess, so peak memory is measured per structure and a crash or
out-of-memory error ends only that series. Re-running the command skips finished measurements;
--plot-only redraws the figures from saved results.

The earlier version is checked out with `git worktree` (default: tag v1.0.14), and both versions run in the
current Python environment.
"""

import argparse
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = Path(__file__).resolve().parent / 'benchmark_output'
VERSIONS = ('old', 'new')


# ----------------------------------------------------------------------------------------------------------------------
# Worker: one measurement in a fresh process
# ----------------------------------------------------------------------------------------------------------------------

def _peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e6 if sys.platform == 'darwin' else rss / 1e3


def worker(args: argparse.Namespace) -> None:
    source_root = Path(args.source).resolve()
    sys.path.insert(0, str(source_root))

    import warnings
    warnings.filterwarnings('ignore')
    import torch
    import debyecalculator
    from debyecalculator import DebyeCalculator

    module_path = Path(debyecalculator.__file__).resolve()
    if source_root not in module_path.parents:
        raise RuntimeError(f'Imported debyecalculator from {module_path}, expected a module under {source_root}')

    structure = np.load(args.structure)
    source = (list(structure['elements']), structure['xyz'])
    dtype = getattr(torch, args.dtype)
    on_cuda = args.device == 'cuda'

    kwargs = dict(device=args.device, dtype=dtype)
    if args.batch_size is not None:
        kwargs['batch_size'] = args.batch_size
    calc = DebyeCalculator(**kwargs)

    if args.mode == 'patterns':
        result = calc._get_all(source)
        np.savez(args.output, r=result.r, q=result.q, i=result.i, s=result.s, f=result.f, g=result.g)
        print(json.dumps({'ok': True}))
        return

    def timed_run() -> float:
        if on_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        calc.gr(source)
        if on_cuda:
            torch.cuda.synchronize()
        return time.perf_counter() - start

    rss_before = _peak_rss_mb()
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()

    # The warm-up run includes one-off costs (CUDA context, kernel loading); a slow warm-up is the only measurement
    warmup = timed_run()
    if warmup > args.time_limit:
        times = [warmup]
    else:
        n_reps = max(1, min(args.repetitions, int(args.time_limit / max(warmup, 1e-9))))
        times = [timed_run() for _ in range(n_reps)]

    result = {
        'ok': True,
        'num_atoms': int(len(source[0])),
        'times': times,
        'mean': float(np.mean(times)),
        'std': float(np.std(times)),
        'batch_size': calc.batch_size,
        'module': str(module_path),
        'cpu_peak_rss_mb': _peak_rss_mb(),
        'cpu_rss_increase_mb': _peak_rss_mb() - rss_before,
        'cuda_peak_mb': torch.cuda.max_memory_allocated() / 1e6 if on_cuda else None,
        'num_threads': torch.get_num_threads(),
    }
    print(json.dumps(result))


# ----------------------------------------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------------------------------------

def run_worker(source: Path, extra_args: list, timeout: float) -> dict:
    cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', '--source', str(source)] + [str(a) for a in extra_args]
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {'ok': False, 'error': f'timeout after {timeout:.0f} s'}
    lines = [line for line in proc.stdout.splitlines() if line.startswith('{')]
    if proc.returncode != 0 or not lines:
        error = (proc.stderr.strip().splitlines() or ['unknown error'])[-1]
        if 'out of memory' in proc.stderr.lower():
            error = 'out of memory: ' + error
        return {'ok': False, 'error': error, 'returncode': proc.returncode}
    return json.loads(lines[-1])


def prepare_old_source(out: Path, ref: str) -> Path:
    target = out / f'_source_{ref.replace("/", "_")}'
    if not (target / 'debyecalculator' / '__init__.py').exists():
        subprocess.run(['git', '-C', str(REPO_ROOT), 'worktree', 'add', '--force', '--detach', str(target), ref], check=True)
    return target


def git_describe(path: Path) -> str:
    try:
        return subprocess.run(['git', '-C', str(path), 'describe', '--always', '--dirty', '--tags'],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'


def generate_structures(out: Path, cif: Path, radii: list) -> dict:
    sys.path.insert(0, str(REPO_ROOT))
    import warnings
    warnings.filterwarnings('ignore')
    from debyecalculator.utility.generate import generate_nanoparticles

    structure_dir = out / 'structures'
    structure_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for radius in radii:
        path = structure_dir / f'{cif.stem}_r{radius:g}.npz'
        if not path.exists():
            particle = generate_nanoparticles(str(cif), float(radius), disable_pbar=True, device='cpu')[0]
            np.savez(path, elements=np.array(particle.elements), xyz=particle.xyz.cpu().numpy().astype(np.float64))
        paths[radius] = path
    return paths


def hardware_info(devices: list) -> dict:
    info = {'python': platform.python_version(), 'platform': platform.platform(), 'cpu': platform.processor()}
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    info['cpu'] = line.split(':', 1)[1].strip()
                    break
    except OSError:
        pass
    import torch
    info['torch'] = torch.__version__
    info['cpu_threads'] = torch.get_num_threads()
    if 'cuda' in devices:
        info['gpu'] = torch.cuda.get_device_name(0)
        info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
    return info


def run_benchmarks(args: argparse.Namespace) -> None:
    import torch

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / 'results.json'
    results = json.loads(results_path.read_text()) if results_path.exists() else {'timings': {}}

    devices = args.devices or (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu'])
    sources = {'old': prepare_old_source(out, args.old_ref), 'new': REPO_ROOT}
    radii = sorted(set(args.radii))
    structures = generate_structures(out, Path(args.cif), sorted(set(radii) | {args.accuracy_radius}))

    results['meta'] = {
        'old_ref': args.old_ref,
        'old_version': git_describe(sources['old']),
        'new_version': git_describe(REPO_ROOT),
        'cif': Path(args.cif).name,
        'dtype': args.dtype,
        'time_limit_s': args.time_limit,
        'repetitions': args.repetitions,
        'hardware': hardware_info(devices),
    }

    def save():
        results_path.write_text(json.dumps(results, indent=1))

    for device in devices:
        for version in VERSIONS:
            key = f'{version}-{device}'
            series = results['timings'].setdefault(key, {})
            print(f'\n== {key} ({results["meta"][version + "_version"]})', flush=True)
            for radius in radii:
                entry = series.get(f'{radius:g}')
                if entry is not None:
                    status = f'{entry["mean"]:.4g} s' if entry.get('ok') else entry.get('error')
                    print(f'  r = {radius:5g} Å: saved ({status})', flush=True)
                    if not entry.get('ok') or entry['mean'] > args.time_limit:
                        break
                    continue

                worker_args = ['--structure', structures[radius], '--device', device, '--dtype', args.dtype,
                               '--repetitions', args.repetitions, '--time-limit', args.time_limit]
                if args.batch_size is not None:
                    worker_args += ['--batch-size', args.batch_size]
                timeout = args.time_limit * (args.repetitions + 2) + 300
                entry = run_worker(sources[version], worker_args, timeout)
                entry['radius'] = radius
                series[f'{radius:g}'] = entry
                save()

                if entry.get('ok'):
                    print(f'  r = {radius:5g} Å, N = {entry["num_atoms"]:7d}: {entry["mean"]:.4g} ± {entry["std"]:.2g} s', flush=True)
                    if entry['mean'] > args.time_limit:
                        print(f'  time limit of {args.time_limit} s reached, skipping larger structures', flush=True)
                        break
                else:
                    print(f'  r = {radius:5g} Å: failed ({entry["error"]}), skipping larger structures', flush=True)
                    break

    # Patterns for the accuracy comparison: float64 on the CPU with the old version is the reference
    pattern_dir = out / 'patterns'
    pattern_dir.mkdir(exist_ok=True)
    configurations = [('old', 'cpu', 'float64')] + [(v, d, 'float32') for d in devices for v in VERSIONS] + [('new', 'cpu', 'float64')]
    for version, device, dtype in configurations:
        path = pattern_dir / f'{version}-{device}-{dtype}.npz'
        if path.exists():
            continue
        worker_args = ['--mode', 'patterns', '--structure', structures[args.accuracy_radius], '--device', device,
                       '--dtype', dtype, '--output', path]
        entry = run_worker(sources[version], worker_args, timeout=3600)
        if not entry.get('ok'):
            print(f'Pattern calculation {version}-{device}-{dtype} failed: {entry.get("error")}')
    results['meta']['accuracy_radius'] = args.accuracy_radius
    save()


# ----------------------------------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------------------------------

DEVICE_COLORS = {'cpu': '#1f77b4', 'cuda': '#9467bd'}


def series_arrays(series: dict):
    entries = sorted((e for e in series.values() if e.get('ok')), key=lambda e: e['radius'])
    radius = np.array([e['radius'] for e in entries])
    atoms = np.array([e['num_atoms'] for e in entries])
    mean = np.array([e['mean'] for e in entries])
    std = np.array([e['std'] for e in entries])
    return entries, radius, atoms, mean, std


def device_label(meta: dict, device: str) -> str:
    hardware = meta['hardware']
    if device == 'cuda':
        return hardware.get('gpu', 'GPU')
    return f'CPU, {hardware.get("cpu_threads", "?")} threads'


def add_atom_axis(ax, results: dict):
    # Top axis: number of atoms at the benchmarked diameters
    points = {}
    for series in results['timings'].values():
        for e in series.values():
            if e.get('ok'):
                points[2 * e['radius']] = e['num_atoms']
    if not points:
        return
    diameters = np.array(sorted(points))
    atoms = np.array([points[d] for d in diameters])
    top = ax.twiny()
    top.set_xlim(ax.get_xlim())
    ticks = diameters[np.linspace(0, len(diameters) - 1, min(6, len(diameters))).round().astype(int)]
    top.set_xticks(ticks)
    top.set_xticklabels([f'{points[t]:,}' for t in ticks])
    top.set_xlabel('Number of atoms')


def plot_time(results: dict, out: Path, plt) -> None:
    meta = results['meta']
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, series in results['timings'].items():
        version, device = key.split('-')
        _, radius, atoms, mean, std = series_arrays(series)
        if len(radius) == 0:
            continue
        name = f'{meta[version + "_version"]}' if version == 'old' else 'optimised'
        style = dict(color=DEVICE_COLORS.get(device, 'k'), marker='o' if device == 'cpu' else 'D', markersize=4,
                     linestyle='--' if version == 'old' else '-',
                     markerfacecolor='white' if version == 'old' else DEVICE_COLORS.get(device, 'k'))
        ax.plot(2 * radius, mean, label=f'{name} ({device_label(meta, device)})', **style)
        ax.fill_between(2 * radius, np.maximum(mean - std, 1e-6), mean + std, color=style['color'], alpha=0.15)
        failed = [e for e in series.values() if not e.get('ok')]
        if failed:
            ax.axvline(2 * failed[0]['radius'], color=style['color'], linestyle=':', linewidth=1)
    ax.set_yscale('log')
    ax.set_xlabel('Structure diameter [Å]')
    ax.set_ylabel('G(r) calculation time [s]')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)
    add_atom_axis(ax, results)
    fig.tight_layout()
    save_figure(fig, out / 'figure_benchmark_time')


def plot_speedup(results: dict, out: Path, plt) -> None:
    meta = results['meta']
    fig, ax = plt.subplots(figsize=(8, 4.5))
    devices = sorted({key.split('-')[1] for key in results['timings']})
    for device in devices:
        old = {e['radius']: e for e in results['timings'].get(f'old-{device}', {}).values() if e.get('ok')}
        new = {e['radius']: e for e in results['timings'].get(f'new-{device}', {}).values() if e.get('ok')}
        common = sorted(set(old) & set(new))
        if not common:
            continue
        atoms = np.array([new[r]['num_atoms'] for r in common])
        speedup = np.array([old[r]['mean'] / new[r]['mean'] for r in common])
        ax.plot(atoms, speedup, marker='o', markersize=4, color=DEVICE_COLORS.get(device, 'k'),
                label=device_label(meta, device))
    if 'old-cpu' in results['timings'] and 'cuda' in devices:
        old = {e['radius']: e for e in results['timings']['old-cpu'].values() if e.get('ok')}
        new = {e['radius']: e for e in results['timings'].get('new-cuda', {}).values() if e.get('ok')}
        common = sorted(set(old) & set(new))
        if common:
            ax.plot([new[r]['num_atoms'] for r in common], [old[r]['mean'] / new[r]['mean'] for r in common],
                    marker='s', markersize=4, color='#2ca02c', linestyle='--',
                    label=f'{meta["old_version"]} on CPU vs optimised on GPU')
    ax.axhline(1, color='k', linewidth=0.8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of atoms')
    ax.set_ylabel(f'Speed-up over {meta["old_version"]}')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, out / 'figure_benchmark_speedup')


def plot_memory(results: dict, out: Path, plt) -> None:
    meta = results['meta']
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, series in results['timings'].items():
        version, device = key.split('-')
        entries, radius, atoms, mean, std = series_arrays(series)
        if len(entries) == 0:
            continue
        field = 'cuda_peak_mb' if device == 'cuda' else 'cpu_peak_rss_mb'
        memory = np.array([e[field] for e in entries], dtype=float)
        name = meta['old_version'] if version == 'old' else 'optimised'
        what = 'peak allocated GPU memory' if device == 'cuda' else 'peak process memory (RSS)'
        ax.plot(atoms, memory, color=DEVICE_COLORS.get(device, 'k'), marker='o' if device == 'cpu' else 'D',
                markersize=4, linestyle='--' if version == 'old' else '-',
                markerfacecolor='white' if version == 'old' else DEVICE_COLORS.get(device, 'k'),
                label=f'{name}, {what}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of atoms')
    ax.set_ylabel('Memory [MB]')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, out / 'figure_benchmark_memory')


def plot_accuracy(results: dict, out: Path, plt) -> dict:
    meta = results['meta']
    pattern_dir = out / 'patterns'
    reference_path = pattern_dir / 'old-cpu-float64.npz'
    if not reference_path.exists():
        return {}
    reference = np.load(reference_path)
    comparisons = {}
    for path in sorted(pattern_dir.glob('*.npz')):
        if path == reference_path:
            continue
        data = np.load(path)
        comparisons[path.stem] = {
            name: float(np.max(np.abs(data[name] - reference[name])) / np.max(np.abs(reference[name])))
            for name in 'isfg'
        }

    new_path = pattern_dir / 'new-cpu-float32.npz'
    if not new_path.exists():
        return comparisons
    new = np.load(new_path)
    old32_path = pattern_dir / 'old-cpu-float32.npz'
    old32 = np.load(old32_path) if old32_path.exists() else None

    panels = [('i', 'q', 'I(Q) [counts]', 'Q [Å$^{-1}$]'), ('s', 'q', 'S(Q)', 'Q [Å$^{-1}$]'),
              ('f', 'q', 'F(Q)', 'Q [Å$^{-1}$]'), ('g', 'r', 'G(r)', 'r [Å]')]
    fig, axes = plt.subplots(2, 4, figsize=(15, 5.5), gridspec_kw={'height_ratios': [3, 1.3]}, sharex='col')
    for column, (name, axis, ylabel, xlabel) in enumerate(panels):
        top, bottom = axes[0, column], axes[1, column]
        scale = np.max(np.abs(reference[name]))
        top.plot(reference[axis], reference[name], color='k', linewidth=2.5, alpha=0.35, label=f'{meta["old_version"]}')
        top.plot(new[axis], new[name], color=DEVICE_COLORS['cpu'], linewidth=1, label='optimised')
        top.set_ylabel(ylabel)
        top.grid(alpha=0.3)
        bottom.plot(new[axis], (new[name] - reference[name]) / scale, color=DEVICE_COLORS['cpu'], linewidth=0.8,
                    label='optimised float32')
        if old32 is not None:
            bottom.plot(old32[axis], (old32[name] - reference[name]) / scale, color='#ff7f0e', linewidth=0.8,
                        alpha=0.8, label=f'{meta["old_version"]} float32')
        bottom.set_xlabel(xlabel)
        bottom.set_ylabel('Rel. difference')
        bottom.grid(alpha=0.3)
        bottom.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=7)
    fig.suptitle(f'{meta["cif"]}, radius {meta.get("accuracy_radius", "?")} Å: differences relative to '
                 f'{meta["old_version"]} in float64 (scaled by the maximum of each function)', fontsize=10)
    fig.tight_layout()
    save_figure(fig, out / 'figure_accuracy')
    return comparisons


def save_figure(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix('.png'), dpi=300)
    fig.savefig(stem.with_suffix('.pdf'))
    print(f'Wrote {stem.with_suffix(".png")}')


def write_summary(results: dict, comparisons: dict, out: Path) -> None:
    meta = results['meta']
    hardware = meta['hardware']
    lines = [
        '# Benchmark summary', '',
        f'- Old version: {meta["old_version"]} (ref `{meta["old_ref"]}`)',
        f'- New version: {meta["new_version"]}',
        f'- Structure: {meta["cif"]}, G(r) in {meta["dtype"]}, default calculator parameters',
        f'- CPU: {hardware.get("cpu")} ({hardware.get("cpu_threads")} threads)',
        f'- GPU: {hardware.get("gpu", "none")}',
        f'- PyTorch {hardware.get("torch")}, Python {hardware.get("python")}', '',
    ]
    devices = sorted({key.split('-')[1] for key in results['timings']})
    for device in devices:
        old = {e['radius']: e for e in results['timings'].get(f'old-{device}', {}).values()}
        new = {e['radius']: e for e in results['timings'].get(f'new-{device}', {}).values()}
        lines += [f'## {device_label(meta, device)}', '',
                  '| Radius [Å] | Atoms | Old [s] | New [s] | Speed-up |', '|---|---|---|---|---|']
        for radius in sorted(set(old) | set(new)):
            o, n = old.get(radius, {}), new.get(radius, {})
            atoms = n.get('num_atoms', o.get('num_atoms', ''))
            old_time = f'{o["mean"]:.4g}' if o.get('ok') else (o.get('error', '–') if o else '–')
            new_time = f'{n["mean"]:.4g}' if n.get('ok') else (n.get('error', '–') if n else '–')
            speedup = f'{o["mean"] / n["mean"]:.1f}×' if o.get('ok') and n.get('ok') else ''
            lines.append(f'| {radius:g} | {atoms} | {old_time} | {new_time} | {speedup} |')
        lines.append('')
    if comparisons:
        lines += [f'## Maximum difference relative to {meta["old_version"]} (CPU, float64)', '',
                  '| Configuration | I(Q) | S(Q) | F(Q) | G(r) |', '|---|---|---|---|---|']
        for name, values in comparisons.items():
            lines.append(f'| {name} | ' + ' | '.join(f'{values[k]:.1e}' for k in 'isfg') + ' |')
        lines.append('')
    (out / 'summary.md').write_text('\n'.join(lines))
    print(f'Wrote {out / "summary.md"}')


def make_figures(args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out = Path(args.out)
    results = json.loads((out / 'results.json').read_text())
    plot_time(results, out, plt)
    plot_speedup(results, out, plt)
    plot_memory(results, out, plt)
    comparisons = plot_accuracy(results, out, plt)
    write_summary(results, comparisons, out)


# ----------------------------------------------------------------------------------------------------------------------

def parse_radii(text: str) -> list:
    radii = []
    for part in text.split(','):
        if ':' in part:
            start, stop, step = (float(x) for x in part.split(':'))
            radii += list(np.arange(start, stop + 1e-9, step))
        elif part:
            radii.append(float(part))
    return [round(r, 6) for r in radii]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default=str(DEFAULT_OUT), help='Output directory')
    parser.add_argument('--old-ref', default='v1.0.14', help='Git ref of the version to compare against')
    parser.add_argument('--devices', nargs='+', choices=['cpu', 'cuda'], help='Default: cpu, and cuda if available')
    parser.add_argument('--radii', type=parse_radii, default=parse_radii('2:40:2,45:80:5'),
                        help='Particle radii in Å, as comma-separated values or start:stop:step ranges')
    parser.add_argument('--cif', default=str(REPO_ROOT / 'debyecalculator' / 'data' / 'AntiFluorite_Co2O.cif'))
    parser.add_argument('--dtype', default='float32', choices=['float32', 'float64'])
    parser.add_argument('--batch-size', type=int, default=None, help='Default: each version\'s own default')
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--time-limit', type=float, default=120.0,
                        help='A series stops after a structure whose calculation takes longer than this [s]')
    parser.add_argument('--accuracy-radius', type=float, default=15.0, help='Particle radius for the pattern comparison')
    parser.add_argument('--plot-only', action='store_true', help='Only redraw figures from saved results')

    # Worker-only arguments
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--source', help=argparse.SUPPRESS)
    parser.add_argument('--structure', help=argparse.SUPPRESS)
    parser.add_argument('--device', help=argparse.SUPPRESS)
    parser.add_argument('--mode', default='timing', help=argparse.SUPPRESS)
    parser.add_argument('--output', help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        worker(args)
        return
    if not args.plot_only:
        run_benchmarks(args)
    make_figures(args)


if __name__ == '__main__':
    main()
