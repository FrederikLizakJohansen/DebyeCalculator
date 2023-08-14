import tempfile
import os
import argparse
import torch
from time import time
import numpy as np
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt
from ase.io import write
from diffpy.structure import loadStructure
from diffpy.srreal.pdfcalculator import DebyePDFCalculator
from SASCalculator import SASCalculator
from debye_calculator import DebyeCalculator
from generate_nanoparticles import generate_nanoparticles, ase_to_diffpy

def compare_methods(args):

    # Check device
    if torch.cuda.is_available():
        my_device = 'cuda'
        batch_size = args['batch_size_cuda']
    else:
        my_device = 'cpu'
        batch_size = args['batch_size_cpu']
    
    # Make output folder
    if not os.path.exists(args['output_folder']):
        os.mkdir(args['output_folder'])

    # Create instances
    debye_calc = DebyeCalculator(device=my_device, batch_size = batch_size, qmin=1, qmax=40, qstep=0.01, rmax=20, biso=0.1)
    debye_diffpy = DebyePDFCalculator(
        rmin = debye_calc.rmin,
        rmax = debye_calc.rmax,
        rstep = debye_calc.rstep,
        qmin = debye_calc.qmin,
        qmax = debye_calc.qmax,
        qdamp = debye_calc.qdamp,
        qstep = debye_calc.qstep
    )
    # Choose particle
    print('Generating Figure 2')
    nano_particles, nano_sizes = generate_nanoparticles(args['cif'], [10])
    particle = nano_particles[0]
    
    # Calculate I(Q), F(Q) and G(r) for the nanoparticle
    q, iq_dc = debye_calc.iq(particle)
    _, fq_dc = debye_calc.fq(particle)
    r, gr_dc = debye_calc.gr(particle)
    
    # Small Angle Scattering
    debye_calc.update_parameters(qmin=0, qmax=2)
    q_sas, sas_dc = debye_calc.iq(particle)
    debye_calc.update_parameters(qmin=1, qmax=40)
    
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_structure_path = os.path.join(tmpdirname,'tmp_struc.xyz')
        write(tmp_structure_path, particle, 'xyz')
        diffpy_structure = loadStructure(tmp_structure_path)
        diffpy_structure.B11 = debye_calc.biso
        diffpy_structure.B22 = debye_calc.biso
        diffpy_structure.B33 = debye_calc.biso
        diffpy_structure.B12 = 0
        diffpy_structure.B13 = 0
        diffpy_structure.B23 = 0
    """
    diffpy_structure = ase_to_diffpy(particle)

    _, gr_dp = debye_diffpy(diffpy_structure)
    fq_dp = debye_diffpy.fq[int(debye_calc.qmin/debye_calc.qstep):]
    
    # Small angle scattering
    sc = SASCalculator(
        rmin=debye_calc.rmin,
        rmax=debye_calc.rmax,
        rstep=debye_calc.rstep,
        qmin=debye_calc.qmin,
        qmax=debye_calc.qmax,
        qdamp=debye_calc.qdamp
    )
    
    sc.qstep = debye_calc.qstep
    _, iq_dp_full = sc(diffpy_structure)
    iq_dp = iq_dp_full[int(debye_calc.qmin/debye_calc.qstep):]
    
    # Figure
    import matplotlib
    matplotlib.use('Agg')

    fig, ((ax_iq, ax_sas), (ax_fq, ax_gr)) = plt.subplots(2,2, figsize=(13,8))
    
    ax_iq.plot(q,iq_dp, label='DiffPy-CMI')
    ax_iq.plot(q,iq_dc, label='DebyeCalculator (Ours)')
    ax_iq.plot(q,iq_dp-iq_dc, c='#d62728', label='Difference')
    ax_iq.set(xlabel='$Q$ [$\AA^{-1}$]', ylabel='$I(Q)$ [counts]')
    ax_iq.grid(alpha=0.2)
    ax_iq.legend()
    ax_iq.set_title('Scattering Intensity')
    ax_iq.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    ax_sas.plot(q_sas, iq_dp_full[:len(q_sas)], label='DiffPy-CMI')
    ax_sas.plot(q_sas, sas_dc, label='DebyeCalculator (Ours)')
    ax_sas.plot(q_sas, iq_dp_full[:len(q_sas)]-sas_dc, c='#d62728', label='Difference')
    ax_sas.set(xlabel='$Q$ [$\AA^{-1}$]', ylabel='$I(Q)$ [counts]')
    ax_sas.grid(alpha=0.2, which='both')
    ax_sas.set_xscale('log')
    ax_sas.set_title('Small Angle Scattering Intensity')
    ax_sas.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    ax_sas_inset = ax_sas.inset_axes([0.625, 0.625, 0.35, 0.35])
    ax_sas_inset.plot(q_sas, iq_dp_full[:len(q_sas)], label='DiffPy-CMI')
    ax_sas_inset.plot(q_sas, sas_dc, label='DebyeCalculator (Ours)')
    ax_sas_inset.set_yscale('log')
    ax_sas_inset.set_xscale('log')
    ax_sas_inset.set(xlabel='$Q$ [$\AA^{-1}$]', ylabel='$I(Q)$ [counts]')
    ax_sas_inset.grid(alpha=0.2, which='both')
    
    ax_fq.plot(q,fq_dp/max(fq_dp), label='DiffPy-CMI')
    ax_fq.plot(q,fq_dc/max(fq_dc), label='DebyeCalculator (Ours)')
    ax_fq.plot(q, fq_dp/max(fq_dp) - fq_dc/max(fq_dc) - 1, c='#d62728', label='Difference')
    ax_fq.set(xlabel='$Q$ [$\AA^{-1}$]', ylabel='$F(Q)$ [a.u.]', yticks=[])
    ax_fq.grid(alpha=0.2)
    ax_fq.set_title('Reduced Structure Function')
    
    ax_gr.plot(r,gr_dp/max(gr_dp), label='DiffPy-CMI')
    ax_gr.plot(r,gr_dc/max(gr_dc), label='DebyeCalculator (Ours)')
    ax_gr.plot(r, gr_dp/max(gr_dp) - gr_dc/max(gr_dc) - 0.5, c='#d62728', label='Difference')
    ax_gr.set(xlabel='$r$ [$\AA$]', ylabel='$G_r(r)$ [a.u.]', yticks=[])
    ax_gr.grid(alpha=0.2)
    ax_gr.set_title('Reduced Pair Distribution Function')
    
    fig.tight_layout()
    fig.savefig(os.path.join(args['output_folder'], 'figure2.png'), dpi=300)
    plt.close(fig)
    print('Finished Figure 2')

def time_methods(args):
    print('Generating data for Figure 3')
    structure_path = args['cif']
    radii = np.arange(args['min_radius'], args['max_radius'], args['step_radius'])
    particles, sizes = generate_nanoparticles(structure_path, radii)
    n_atoms = [p.get_global_number_of_atoms() for p in particles]
    
    def dummy_calculations():
        # Perform some dummy calculations on the GPU
        dummy_data = torch.rand(1000, 1000, device='cuda')
        for _ in range(10000):
            dummy_data = torch.matmul(dummy_data, dummy_data)

    def time_debye_calculator(device, batch_size):
        mu, sigma = [], []
        debye_calc = DebyeCalculator(device=device, qmin=1, qmax=25, qstep=0.1, biso=0.3, batch_size=batch_size)
        #if device == 'cuda':
            # Move a dummy tensor to the GPU to initialize the CUDA context
            #torch.cuda.FloatTensor(1000).to('cuda')
            #dummy_calculations()
        for i in trange(len(radii), leave=False):
            timings = []
            for j in range(args['reps']+2):
                #if device == 'cuda':
                #    torch.cuda.FloatTensor(1000).to('cuda')
                #    dummy_calculations()
                t = time()
                debye_calc.gr(particles[i]);
                t = time() - t
                if j > 1: 
                    timings.append(t)
            mu.append(np.mean(timings))
            sigma.append(np.std(timings))

        return np.array(mu), np.array(sigma)

    def time_diffpy():
        mu, sigma = [], []
        debye_calc = DebyeCalculator(qmin=1, qmax=25, qstep=0.1, biso=0.3)
        for i in trange(len(radii), leave=False):
            timings = []
            for _ in range(args['reps']):
                with tempfile.TemporaryDirectory() as tmpdirname:
                    tmp_structure_path = os.path.join(tmpdirname,'tmp_struc.xyz')
                    write(tmp_structure_path, particles[i], 'xyz')
        
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

        return np.array(mu), np.array(sigma)

    # Make output folder
    if not os.path.exists(args['output_folder']):
        os.mkdir(args['output_folder'])

    # Run CPU and save
    mu, sigma = time_debye_calculator(device='cpu', batch_size=args['batch_size_cpu'])
    out = np.array([sizes, n_atoms, mu, sigma]).T
    np.savetxt(os.path.join(args['output_folder'],'timings_cpu.csv'), out, delimiter=',', header='diameter, n_atoms, mu, sigma', fmt='%f')

    # Run CUDA and save
    if torch.cuda.is_available():
        mu, sigma = time_debye_calculator(device='cuda', batch_size=args['batch_size_cuda'])
        out = np.array([sizes, n_atoms, mu, sigma]).T
        gpu_id = torch.cuda.get_device_name()
        np.savetxt(os.path.join(args['output_folder'], f'timings_cuda_{gpu_id}.csv'), out, delimiter=',', header='diameter, n_atoms, mu, sigma', fmt='%f')

    # Run Diffpy and save
    mu, sigma = time_diffpy()
    out = np.array([sizes, n_atoms, mu, sigma]).T
    np.savetxt(os.path.join(args['output_folder'], 'timings_diffpy.csv'), out, delimiter=',', header='diameter, n_atoms, mu, sigma', fmt='%f')

    print('Finished generating data for Figure 3')

def produce_figures(args):

    if args['figure2']:
        compare_methods(args)
    if args['figure3']:
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
    parser.add_argument('--figure2', action='store_true')
    parser.add_argument('--figure3', action='store_true')
    parser.add_argument('--reps', type=int, default=1)
    args = parser.parse_args()
    produce_figures(vars(args))
