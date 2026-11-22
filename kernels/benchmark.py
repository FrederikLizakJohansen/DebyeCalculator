"""
Benchmark of the grid pair sum backends: time and peak memory of I(Q) (forward) and of I(Q) plus gradients with respect
to the positions (forward + backward), per backend, precision and particle size.

Backends:
    torch   PyTorch on CPU (the code of DebyeCalculator 1.1)
    rust, c native CPU kernels (kernels/, kernels_c/)
    mps     PyTorch on an Apple GPU (float32 only)
    metal   fused Metal kernels on an Apple GPU (float32 only)
    cuda    PyTorch on an NVIDIA GPU
    triton  fused Triton kernels on an NVIDIA GPU
Every configuration runs in a fresh process, so peak memory is not affected by earlier runs. Each result line records
the machine (host, CPU, GPU, Slurm job), so results from several machines can be combined by benchmark_report.py.

    python kernels/benchmark.py --out results.jsonl                 # all available backends, default sizes
    python kernels/benchmark.py --backends torch rust --radii 10 30 --dtypes float32
    python kernels/benchmark.py --backends rust triton cuda --extra-radii 80 100 --extra-backends rust triton

Accuracy is reported relative to the float64 result of the first CPU backend in --reference (default: rust, then c,
then torch), which agree to ~1e-14.
"""
import argparse
import gc
import json
import os
import platform
import socket
import subprocess
import sys
import threading
import time

CIF = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'debyecalculator', 'data', 'AntiFluorite_Co2O.cif')


def cpu_model():
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except OSError:
        pass
    try:
        return subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True).stdout.strip()
    except OSError:
        return platform.processor()


def device_name(torch, device):
    if device == 'cuda':
        return torch.cuda.get_device_name(0)
    if device == 'mps':
        return 'Apple GPU (' + cpu_model() + ')'
    return cpu_model()


def machine_info(torch, label):
    """Where and with what the benchmark ran."""
    info = {
        'label': label,
        'host': socket.gethostname(),
        'cpu': cpu_model(),
        'logical_cpus': os.cpu_count(),
        'torch_threads': torch.get_num_threads(),
        'python': platform.python_version(),
        'torch': torch.__version__,
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'slurm_partition': os.environ.get('SLURM_JOB_PARTITION'),
        'slurm_cpus': os.environ.get('SLURM_CPUS_PER_TASK') or os.environ.get('SLURM_CPUS_ON_NODE'),
    }
    if torch.cuda.is_available():
        info.update(gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda)
        try:
            import triton
            info['triton'] = triton.__version__
        except ImportError:
            pass
    return info


def worker(args):
    import warnings
    warnings.filterwarnings('ignore')
    import numpy as np
    import psutil
    import torch
    from debyecalculator import DebyeCalculator
    from debyecalculator.utility.generate import generate_nanoparticles

    dtype = getattr(torch, args.dtype)
    device = {'mps': 'mps', 'metal': 'mps', 'cuda': 'cuda', 'triton': 'cuda'}.get(args.backend, 'cpu')
    particle = generate_nanoparticles(CIF, args.radius, disable_pbar=True, device='cpu')[0]
    source = (list(particle.elements), particle.xyz.numpy())
    calc = DebyeCalculator(device=device, dtype=dtype, qmin=1.0, qmax=30.0)
    calc._pair_backend = args.backend if args.backend in ('rust', 'c', 'metal', 'triton') else 'torch'
    structure = calc._initialize_structures(source)[0]
    weights = torch.linspace(1.0, 2.0, calc.q.numel(), device=device, dtype=dtype)
    process = psutil.Process()

    def sync():
        if device == 'mps':
            torch.mps.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

    def device_memory():
        if device == 'mps':
            return torch.mps.driver_allocated_memory()
        if device == 'cuda':
            return torch.cuda.memory_reserved()
        return 0

    def tensor_memory():
        """Memory held by live tensors on the GPU (without the allocator's cached heaps)."""
        if device == 'mps':
            return torch.mps.current_allocated_memory()
        if device == 'cuda':
            return torch.cuda.memory_allocated()
        return 0

    def forward():
        return calc.compute_iq(structure), None

    def forward_backward():
        xyz = structure.xyz.detach().clone().requires_grad_(True)
        iq = calc.compute_iq(structure._replace(xyz=xyz))
        (iq * weights).sum().backward()
        return iq, xyz.grad

    def measure(function):
        """
        Minimum time over repeats, and peak memory above the level before the run (sampled every millisecond), after one
        untimed warm-up run (which includes one-time costs: CUDA/MPS start-up, kernel compilation, self-tests).
        """
        del function()[0]
        sync()
        times, peak_rss, peak_device, peak_tensors = [], 0, 0, 0
        while len(times) < 2 or (len(times) < args.repeats and sum(times) < args.budget):
            gc.collect()
            if device == 'mps':
                torch.mps.empty_cache()
            elif device == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            base_rss, base_device, base_tensors = process.memory_info().rss, device_memory(), tensor_memory()
            peak = {'rss': base_rss, 'device': base_device, 'tensors': base_tensors}
            stop = threading.Event()

            def sample():
                while not stop.is_set():
                    peak['rss'] = max(peak['rss'], process.memory_info().rss)
                    peak['device'] = max(peak['device'], device_memory())
                    peak['tensors'] = max(peak['tensors'], tensor_memory())
                    time.sleep(0.001)

            sampler = threading.Thread(target=sample, daemon=True)
            sampler.start()
            start = time.perf_counter()
            result = function()
            sync()
            times.append(time.perf_counter() - start)
            stop.set()
            sampler.join()
            peak['rss'] = max(peak['rss'], process.memory_info().rss)
            if device == 'cuda':
                peak['device'] = max(peak['device'], torch.cuda.max_memory_allocated())
                peak['tensors'] = max(peak['tensors'], torch.cuda.max_memory_allocated())
            peak_rss = max(peak_rss, peak['rss'] - base_rss)
            peak_device = max(peak_device, peak['device'] - base_device)
            peak_tensors = max(peak_tensors, peak['tensors'] - base_tensors)
            del result
        return min(times), peak_rss, peak_device, peak_tensors, len(times)

    record = {'backend': args.backend, 'dtype': args.dtype, 'radius': args.radius, 'n_atoms': len(source[0]),
              'threads': torch.get_num_threads(), 'device_name': device_name(torch, device)}
    for name, function in (('forward', forward), ('forward_backward', forward_backward)):
        seconds, rss, device_bytes, tensor_bytes, runs = measure(function)
        record[name] = {'seconds': seconds, 'peak_rss_mb': rss / 2 ** 20, 'peak_device_mb': device_bytes / 2 ** 20,
                        'peak_device_tensors_mb': tensor_bytes / 2 ** 20, 'runs': runs}
    iq, grad = forward_backward()
    np.savez(args.arrays, iq=iq.detach().cpu().double().numpy(), grad=grad.cpu().double().numpy())
    print(json.dumps(record))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--backends', nargs='+', default=None,
                        help='torch, rust, c, mps, metal, cuda, triton (default: all available)')
    parser.add_argument('--dtypes', nargs='+', default=['float32', 'float64'])
    parser.add_argument('--radii', nargs='+', type=float, default=[10, 20, 30, 45, 63])
    parser.add_argument('--extra-radii', nargs='+', type=float, default=[],
                        help='larger particles, run only for --extra-backends')
    parser.add_argument('--extra-backends', nargs='+', default=['rust', 'triton', 'metal'])
    parser.add_argument('--label', default=None,
                        help='name of this machine in reports (default: Slurm partition, else host name)')
    parser.add_argument('--repeats', type=int, default=5, help='maximum timed runs per configuration (at least 2)')
    parser.add_argument('--budget', type=float, default=10.0, help='seconds of timed runs after which to stop repeating')
    parser.add_argument('--reference', nargs='+', default=['rust', 'c', 'torch'])
    parser.add_argument('--out', default='benchmark_results.jsonl')
    parser.add_argument('--arrays-dir', default=None, help='where to keep I(Q) and gradients (default: next to --out)')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--backend', help=argparse.SUPPRESS)
    parser.add_argument('--dtype', help=argparse.SUPPRESS)
    parser.add_argument('--radius', type=float, help=argparse.SUPPRESS)
    parser.add_argument('--arrays', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args)

    import numpy as np
    import torch
    from debyecalculator.debye_calculator import _NATIVE_BACKENDS

    available = ['torch'] + sorted(_NATIVE_BACKENDS)
    if torch.backends.mps.is_available():
        available.append('mps')
        from debyecalculator import metal_pair_sum
        if metal_pair_sum.library() is not None:
            available.append('metal')
    if torch.cuda.is_available():
        available.append('cuda')
        from debyecalculator import triton_pair_sum
        if triton_pair_sum.library() is not None:
            available.append('triton')
        else:
            print(f'triton: not available ({triton_pair_sum.unavailable_reason()})', file=sys.stderr)
    backends = args.backends or available
    label = args.label or os.environ.get('SLURM_JOB_PARTITION') or socket.gethostname().split('.')[0]
    machine = machine_info(torch, label)
    print('machine:', json.dumps(machine), flush=True)
    arrays_dir = args.arrays_dir or os.path.splitext(os.path.abspath(args.out))[0] + '_arrays'
    os.makedirs(arrays_dir, exist_ok=True)
    reference_backend = next((b for b in args.reference if b in backends and b in available), None)

    with open(args.out, 'a') as out:
        for radius in list(args.radii) + list(args.extra_radii):
            run_backends = backends if radius in args.radii else [b for b in backends if b in args.extra_backends]
            # The float64 reference first, so every other run can be compared with it
            configs = [(b, d) for b in run_backends for d in args.dtypes
                       if not (b in ('mps', 'metal') and d == 'float64')]
            configs.sort(key=lambda c: c != (reference_backend, 'float64'))
            reference = None
            for backend, dtype in configs:
                if backend not in available:
                    print(f'skipping {backend}: not available', file=sys.stderr)
                    continue
                arrays = os.path.join(arrays_dir, f'{backend}_{dtype}_r{radius:g}.npz')
                command = [sys.executable, os.path.abspath(__file__), '--worker', '--backend', backend, '--dtype', dtype,
                           '--radius', str(radius), '--arrays', arrays, '--repeats', str(args.repeats),
                           '--budget', str(args.budget)]
                run = subprocess.run(command, capture_output=True, text=True)
                if run.returncode != 0:
                    print(f'{backend} {dtype} r={radius:g} failed:\n{run.stderr[-2000:]}', file=sys.stderr)
                    failure = {'backend': backend, 'dtype': dtype, 'radius': radius, 'machine': machine,
                               'error': run.stderr[-2000:]}
                    out.write(json.dumps(failure) + '\n')
                    out.flush()
                    continue
                record = json.loads(run.stdout.strip().splitlines()[-1])
                record['machine'] = machine
                result = np.load(arrays)
                if (backend, dtype) == (reference_backend, 'float64'):
                    reference = result
                if reference is not None:
                    record['iq_error'] = float(np.abs(result['iq'] - reference['iq']).max() / np.abs(reference['iq']).max())
                    record['grad_error'] = float(np.linalg.norm(result['grad'] - reference['grad'])
                                                 / np.linalg.norm(reference['grad']))
                    record['reference'] = f'{reference_backend} float64'
                out.write(json.dumps(record) + '\n')
                out.flush()
                f, fb = record['forward'], record['forward_backward']
                print(f"N={record['n_atoms']:7d} {backend:6s} {dtype:7s} "
                      f"I(Q) {f['seconds']:8.3f}s {f['peak_rss_mb']:7.0f}MB RAM {f['peak_device_mb']:5.0f}MB dev "
                      f"({f.get('peak_device_tensors_mb', 0):4.0f}MB tensors) | "
                      f"+grad {fb['seconds']:8.3f}s {fb['peak_rss_mb']:7.0f}MB RAM {fb['peak_device_mb']:5.0f}MB dev "
                      f"({fb.get('peak_device_tensors_mb', 0):4.0f}MB tensors) | "
                      f"err I(Q) {record.get('iq_error', float('nan')):.1e} grad {record.get('grad_error', float('nan')):.1e}",
                      flush=True)


if __name__ == '__main__':
    main()
