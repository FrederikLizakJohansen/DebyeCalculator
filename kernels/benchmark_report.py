"""
Charts and markdown tables from the output of kernels/benchmark.py, for one machine or several.

    python kernels/benchmark_report.py results.jsonl --out-dir kernels/benchmark_results
    python kernels/benchmark_report.py mac.jsonl xeon56el9.jsonl a100el9.jsonl --out-dir combined

Each record is labelled with its machine (the 'label' that benchmark.py stored, else the file name). With several
machines, the report adds tables over all machines and backends, and one chart per machine.
"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# Fixed categorical order (colour follows the backend, never its rank)
BACKENDS = ['torch', 'rust', 'c', 'mps', 'metal', 'cuda', 'triton']
LABELS = {'torch': 'PyTorch CPU (1.1)', 'rust': 'Rust CPU', 'c': 'C CPU', 'mps': 'PyTorch MPS', 'metal': 'Metal MPS',
          'cuda': 'PyTorch CUDA', 'triton': 'Triton CUDA'}
COLORS = {'torch': '#2a78d6', 'rust': '#eb6834', 'c': '#1baf7a', 'mps': '#eda100', 'metal': '#e87ba4', 'cuda': '#008300',
          'triton': '#6250d6'}
TEXT, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
MODES = [('forward', 'I(Q)'), ('forward_backward', 'I(Q) + gradient')]
DTYPES = ['float32', 'float64']


def memory_mb(m):
    """Peak RAM plus peak GPU memory held by tensors (driver memory if tensor memory wasn't recorded)."""
    return m['peak_rss_mb'] + m.get('peak_device_tensors_mb', m['peak_device_mb'])


def label_ends(ax, ends, min_gap=11):
    """Direct labels at the line ends, pushed apart vertically so that none overlap."""
    if not ends:
        return
    points = sorted((ax.transData.transform((x, y))[1], backend, x, y) for backend, x, y in ends)
    placed = []
    for y_px, backend, x, y in points:
        target = y_px if not placed else max(y_px, placed[-1] + min_gap)
        placed.append(target)
        offset = (target - y_px) * 72 / ax.figure.dpi
        ax.annotate(LABELS[backend].replace(' (1.1)', ''), (x, y), xytext=(6, offset), textcoords='offset points',
                    va='center', fontsize=8, color=MUTED)


def load(paths):
    """Records of all files, each with '_label' (its machine); failed runs are returned separately."""
    records, failures = [], []
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                r['_label'] = (r.get('machine') or {}).get('label') or stem
                (failures if 'error' in r else records).append(r)
    return records, failures


def machines(records):
    """Machine labels in order of first appearance, and a one-line description of each."""
    out = {}
    for r in records:
        if r['_label'] not in out:
            m = r.get('machine') or {}
            parts = [m.get('cpu'), f"{m.get('torch_threads')} threads" if m.get('torch_threads') else None, m.get('gpu'),
                     f"PyTorch {m['torch']}" if m.get('torch') else None,
                     f"CUDA {m['cuda']}" if m.get('cuda') else None, f"Triton {m['triton']}" if m.get('triton') else None]
            out[r['_label']] = ', '.join(p for p in parts if p) or r.get('device_name', '')
    return out


def combined_tables(records):
    """Over all machines: one row per machine and backend, one column per particle size."""
    sizes = sorted({r['n_atoms'] for r in records})
    order = list(machines(records))
    lines = []
    for dtype in DTYPES:
        for mode, mode_label in MODES:
            rows = defaultdict(dict)
            for r in records:
                if r['dtype'] == dtype:
                    rows[(r['_label'], r['backend'])][r['n_atoms']] = r[mode]
            if not rows:
                continue
            lines.append(f'\n**{mode_label}, {dtype}**: time / peak memory (RAM + GPU tensors)\n')
            lines.append('| machine | backend | ' + ' | '.join(f'{n:,} atoms' for n in sizes) + ' |')
            lines.append('|---|---|' + '---:|' * len(sizes))
            for key in sorted(rows, key=lambda k: (order.index(k[0]), BACKENDS.index(k[1]))):
                cells = [f"{fmt_time(rows[key][n]['seconds'])} / {memory_mb(rows[key][n]):.0f} MB" if n in rows[key]
                         else '–' for n in sizes]
                lines.append(f'| {key[0]} | {LABELS[key[1]]} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def series(records, mode, dtype, quantity):
    out = defaultdict(list)
    for r in records:
        if r['dtype'] != dtype:
            continue
        value = r[mode]['seconds'] if quantity == 'seconds' else memory_mb(r[mode])
        out[r['backend']].append((r['n_atoms'], value))
    return {b: sorted(v) for b, v in out.items()}


def chart(records, quantity, ylabel, title, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), sharex=True, facecolor=SURFACE)
    for row, dtype in enumerate(DTYPES):
        for col, (mode, mode_label) in enumerate(MODES):
            ax = axes[row][col]
            ax.set_facecolor(SURFACE)
            data = series(records, mode, dtype, quantity)
            ends = []
            for backend in BACKENDS:
                if backend not in data:
                    continue
                n, v = zip(*data[backend])
                ax.plot(n, v, color=COLORS[backend], lw=2, marker='o', ms=6, mec=SURFACE, mew=1.5,
                        label=LABELS[backend], solid_capstyle='round')
                ends.append((backend, n[-1], v[-1]))
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title(f'{mode_label}, {dtype}', fontsize=11, color=TEXT, loc='left')
            ax.grid(True, which='major', color=GRID, lw=0.8)
            ax.tick_params(colors=MUTED, labelsize=9)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            for side in ('left', 'bottom'):
                ax.spines[side].set_color(GRID)
            ax.set_xlim(right=ax.get_xlim()[1] * 2.2)
            label_ends(ax, ends)
            if row == 1:
                ax.set_xlabel('atoms', color=MUTED)
            if col == 0:
                ax.set_ylabel(ylabel, color=MUTED)
            if dtype == 'float64' and {'mps', 'metal'} & {r['backend'] for r in records}:
                ax.text(0.98, 0.03, 'MPS: no float64', transform=ax.transAxes, ha='right', fontsize=8, color=MUTED)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.955), labelcolor=TEXT)
    fig.suptitle(title, x=0.06, ha='left', fontsize=13, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fmt_time(s):
    return f'{s * 1e3:.1f} ms' if s < 1 else f'{s:.2f} s'


def table(records, dtype):
    by = defaultdict(dict)
    for r in records:
        if r['dtype'] == dtype:
            by[r['n_atoms']][r['backend']] = r
    backends = [b for b in BACKENDS if any(b in v for v in by.values())]
    lines = []
    for mode, mode_label in MODES:
        lines.append(f'\n**{mode_label}, {dtype}**: time (speed-up vs PyTorch CPU) / peak memory (RAM + GPU tensors)\n')
        lines.append('| atoms | ' + ' | '.join(LABELS[b] for b in backends) + ' |')
        lines.append('|---:|' + '---:|' * len(backends))
        for n in sorted(by):
            cells = []
            base = by[n].get('torch', {}).get(mode, {}).get('seconds')
            for b in backends:
                r = by[n].get(b)
                if r is None:
                    cells.append('–')
                    continue
                m = r[mode]
                memory = memory_mb(m)
                speed = f' ({base / m["seconds"]:.1f}×)' if base and b != 'torch' else ''
                cells.append(f'{fmt_time(m["seconds"])}{speed} / {memory:.0f} MB')
            lines.append(f'| {n:,} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def gpu_memory(records):
    lines = ['| atoms | backend | I(Q): GPU tensors / GPU driver | I(Q) + gradient: GPU tensors / GPU driver |',
             '|---:|---|---:|---:|']
    for r in sorted(records, key=lambda r: (r['n_atoms'], BACKENDS.index(r['backend']))):
        if r['backend'] in ('mps', 'metal', 'cuda', 'triton') and 'peak_device_tensors_mb' in r['forward']:
            f, fb = r['forward'], r['forward_backward']
            lines.append(f"| {r['n_atoms']:,} | {LABELS[r['backend']]} | {f['peak_device_tensors_mb']:.0f} MB / "
                         f"{f['peak_device_mb']:.0f} MB | {fb['peak_device_tensors_mb']:.0f} MB / {fb['peak_device_mb']:.0f} MB |")
    return '\n'.join(lines)


def accuracy(records):
    lines = ['| atoms | backend | dtype | I(Q) max. rel. error | gradient rel. error |', '|---:|---|---|---:|---:|']
    for r in sorted(records, key=lambda r: (r['n_atoms'], BACKENDS.index(r['backend']), r['dtype'])):
        if 'iq_error' in r:
            lines.append(f"| {r['n_atoms']:,} | {LABELS[r['backend']]} | {r['dtype']} | {r['iq_error']:.1e} | "
                         f"{r['grad_error']:.1e} |")
    return '\n'.join(lines)


def report(records, out_dir, suffix=''):
    """Charts and tables of one machine."""
    chart(records, 'seconds', 'seconds', f'Time of the grid pair sum, by backend{suffix}',
          os.path.join(out_dir, 'time.png'))
    chart(records, 'memory', 'peak memory above baseline (MB)', f'Peak memory, by backend{suffix}',
          os.path.join(out_dir, 'memory.png'))
    with open(os.path.join(out_dir, 'tables.md'), 'w') as f:
        for dtype in DTYPES:
            if any(r['dtype'] == dtype for r in records):
                f.write(table(records, dtype) + '\n')
        if any(r['backend'] in ('mps', 'metal', 'cuda', 'triton') for r in records):
            f.write('\n**GPU memory** (peak above the level before the run; the driver figure includes the allocator\'s '
                    'cached memory)\n\n' + gpu_memory(records) + '\n')
        f.write('\n**Accuracy** (relative to the float64 reference)\n\n' + accuracy(records) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('results', nargs='+', help='JSON-lines files written by benchmark.py')
    parser.add_argument('--out-dir', default='.')
    args = parser.parse_args()
    records, failures = load(args.results)
    os.makedirs(args.out_dir, exist_ok=True)
    labels = machines(records)
    if len(labels) == 1:
        report(records, args.out_dir)
    else:
        for label in labels:
            sub = os.path.join(args.out_dir, label)
            os.makedirs(sub, exist_ok=True)
            report([r for r in records if r['_label'] == label], sub, f' ({label})')
        with open(os.path.join(args.out_dir, 'combined.md'), 'w') as f:
            f.write('**Machines**\n\n' + '\n'.join(f'- **{k}**: {v}' for k, v in labels.items()) + '\n')
            f.write(combined_tables(records) + '\n')
    if failures:
        with open(os.path.join(args.out_dir, 'failures.md'), 'w') as f:
            for r in failures:
                f.write(f"## {r['_label']} {r['backend']} {r['dtype']} r={r['radius']:g}\n\n```\n{r['error']}\n```\n\n")
        print(f'{len(failures)} failed runs, see failures.md')
    print('wrote', args.out_dir, '| machines:', ', '.join(labels))


if __name__ == '__main__':
    main()
