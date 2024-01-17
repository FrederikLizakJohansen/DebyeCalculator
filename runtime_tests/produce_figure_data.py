import tempfile
import os
import pkg_resources
import argparse
import torch
from time import time
import numpy as np
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt
from ase.io import write, read
from debyecalculator import DebyeCalculator
from debyecalculator.utility.generate import generate_nanoparticles
from prettytable import PrettyTable, from_csv
from debyecalculator import StructureSourceType

def time_methods(args):

    if args['gen_diffpy']:
        from diffpy.structure import loadStructure
        from diffpy.srreal.pdfcalculator import DebyePDFCalculator
    
    print('Generating data for timing')
    structure_path = args['cif']
    radii = list(np.arange(args['min_radius'], args['max_radius'], args['step_radius']))
    
    def dummy_calculations():
        # Perform some dummy calculations on the GPU
        dummy_data = torch.rand(1000, 1000, device='cuda')
        for _ in range(10000):
            dummy_data = torch.matmul(dummy_data, dummy_data)

    def time_debye_calculator(device, batch_size, output_folder):
        mu, sigma = [], []
        debye_calc = DebyeCalculator(device=device, qmin=1, qmax=25, qstep=0.1, biso=0.3, batch_size=batch_size)
        nps = generate_nanoparticles(structure_path, radii, _reverse_order=False)
        sizes = [r * 2 for r in radii]
        n_atoms = [nano.size for nano in nps]

        pbar = tqdm(desc='Benchmarking...', total=len(radii))
        for i,nano in enumerate(nps):
            timings = []
            for j in range(args['reps']+2):
                t = time()
                debye_calc.gr((nano.elements, nano.xyz))
                t = time() - t
                if j > 1:
                    timings.append(t)
            mu.append(np.mean(timings))
            sigma.append(np.std(timings))
            pbar.update(1)
        pbar.close()

        return sizes, n_atoms, mu, sigma

    def save_csv(sizes, n_atoms, means, stds, batch_size, device):
        if device.lower() == 'cpu':
            name = f'timings_cpu_bs{batch_size}.csv'
        elif device.lower() == 'cuda':
            gpu_id = torch.cuda.get_device_name()
            name = f'timings_cuda_{gpu_id}_bs{batch_size}.csv'
        else:
            raise ValueError("Device not recognised")
        
        pt = create_pt(sizes, n_atoms, means, stds, batch_size, device)
        with open(os.path.join(args['output_folder'], name), 'w', newline='') as f:
            f.write(pt.get_csv_string())

    def create_pt(sizes, n_atoms, means, stds, batch_size, device):

        # Create table
        table_fields = ['Diameter [Å]', 'Num. atoms', 'Mean [s]', 'Std [s]']
        pt = PrettyTable(table_fields)
        pt.align = 'r'
        pt.padding_width = 1
        pt.title = device + ' / Batch Size ' + str(batch_size)
        data = [[str(s), str(n), f'{m:1.5f}', f'{std:1.5f}'] for s,n,m,std in zip(sizes, n_atoms, means, stds)]
        for d in data:
            pt.add_row(d)
        
        return pt

    def print_pt_from_csv(csv_path):
        with open(csv_path, 'r') as f:
            print(from_csv(f))
        
    def time_diffpy(output_folder):
        mu, sigma = [], []
        debye_calc = DebyeCalculator(qmin=1, qmax=25, qstep=0.1, biso=0.3)
        nps, sizes = debye_calc.generate_nanoparticles(structure_path, radii)
        n_atoms = [len(np) for np in nps]
        for i in trange(len(radii), leave=False):
            timings = []
            for _ in range(args['reps']):
                with tempfile.TemporaryDirectory() as tmpdirname:
                    tmp_structure_path = os.path.join(tmpdirname,'tmp_struc.xyz')
                    write(tmp_structure_path, nps[i], 'xyz')
        
                    debye_diffpy = DebyePDFCalculator(
                        rmin=debye_calc.rmin,
                        rmax=debye_calc.rmax,
                        rstep=debye_calc.rstep,
                        qmin=debye_calc.qmin,
                        qmax=debye_calc.qmax,
                        qdamp=debye_calc.qdamp
                    )
                
                    t = time()
                    diffpy_structure = loadStructure(tmp_structure_path)
                    diffpy_structure.B11 = debye_calc.biso
                    diffpy_structure.B22 = debye_calc.biso
                    diffpy_structure.B33 = debye_calc.biso
                    diffpy_structure.B12 = 0
                    diffpy_structure.B13 = 0
                    diffpy_structure.B23 = 0
                    debye_diffpy(diffpy_structure);
                    t = time() - t
                    timings.append(t)

            mu.append(np.mean(timings))
            sigma.append(np.std(timings))
        
            out = np.array([sizes[:(i+1)], n_atoms[:(i+1)], np.array(mu), np.array(sigma)]).T
            np.savetxt(os.path.join(output_folder, 'timings_diffpy.csv'), out, delimiter=',', header='diameter, n_atoms, mu, sigma', fmt='%f')

    # Make output folder
    if not os.path.exists(args['output_folder']):
        os.mkdir(args['output_folder'])

    # Run CPU and save
    if args['gen_cpu']:
        time_debye_calculator(device='cpu', batch_size=args['batch_size_cpu'], output_folder=args['output_folder'])

    # Run CUDA and save
    if args['gen_cuda'] and torch.cuda.is_available():
        sizes, n_atoms, means, stds = time_debye_calculator(device='cuda', batch_size=args['batch_size_cuda'], output_folder=args['output_folder'])
        if args['print_table']:
            print(create_pt(sizes, n_atoms, means, stds, batch_size=args['batch_size_cuda'], device='CUDA'))
        if args['save_csv']:
            save_csv(sizes, n_atoms, means, stds, batch_size=args['batch_size_cuda'], device='CUDA')

    # Run Diffpy and save
    if args['gen_diffpy']:
        time_diffpy(output_folder=args['output_folder'])

    print('Finished generating data')

def produce_figures(args):
    time_methods(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cif', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    parser.add_argument('--min_radius', type=float, default=2)
    parser.add_argument('--max_radius', type=float, default=10)
    parser.add_argument('--step_radius', type=float, default=1)
    parser.add_argument('--batch_size_cpu', type=int, default=1000)
    parser.add_argument('--batch_size_cuda', type=int, default=5000)
    parser.add_argument('--gen_cpu', action='store_true')
    parser.add_argument('--gen_cuda', action='store_true')
    parser.add_argument('--gen_diffpy', action='store_true')
    parser.add_argument('--reps', type=int, default=1)
    parser.add_argument('--print_table', action='store_true')
    parser.add_argument('--save_csv', action='store_true')
    args = parser.parse_args()
    produce_figures(vars(args))
