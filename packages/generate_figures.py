import tempfile
import os
import argparse
import torch
from time import time

import numpy as np
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from ase.io import write

from debye_calculator import DebyeCalculator
from generate_nanoparticles import generate_nanoparticles

from diffpy.structure import loadStructure
from diffpy.srreal.pdfcalculator import DebyePDFCalculator
from SASCalculator import SASCalculator


def main(
    args
):
    # Make figure 2
    if args['figure2']:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        # Create an instance of DebyeCalculator
        debye_calc = DebyeCalculator(device=device, verbose=1, qmin=1, qmax=25, qstep=0.01, biso=0.3, batch_size=4000)
        
        dbc = DebyePDFCalculator(rmin=debye_calc.rmin, rmax=debye_calc.rmax, rstep=debye_calc.rstep,
                                 qmin=debye_calc.qmin, qmax=debye_calc.qmax, qdamp=debye_calc.qdamp,
                                 qstep=debye_calc.qstep)
        # Choose particle
        print('-'*20)
        print('Generating Figure 2')
        print('Generating particle...')
        nano_particles, nano_sizes = generate_nanoparticles(args['cif'], [10])
        particle = nano_particles[0]
        
        # Calculate I(Q), F(Q) and G(r) for the nanoparticle
        q, iq_dc = debye_calc.iq(particle)
        _, fq_dc = debye_calc.fq(particle)
        r, gr_dc = debye_calc.gr(particle)
        
        debye_calc.update_parameters(qmin=0, qmax=2)
        q_sas, sas_dc = debye_calc.iq(particle)
        debye_calc.update_parameters(qmin=1, qmax=25)
        
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
        
        _, gr_dp = dbc(diffpy_structure)
        fq_dp = dbc.fq[int(debye_calc.qmin/debye_calc.qstep):]
        
        sc = SASCalculator(rmin=debye_calc.rmin, rmax=debye_calc.rmax, rstep=debye_calc.rstep,
                           qmin=debye_calc.qmin, qmax=debye_calc.qmax, qdamp=debye_calc.qdamp)
        
        sc.qstep = debye_calc.qstep
        _, iq_dp_full = sc(diffpy_structure)
        iq_dp = iq_dp_full[int(debye_calc.qmin/debye_calc.qstep):]
        
        # Figure
        fig, ((ax_iq, ax_sas), (ax_fq, ax_gr)) = plt.subplots(2,2, figsize=(13,8))
        
        ax_iq.plot(q,iq_dp, label='Diffpy-CMI')
        ax_iq.plot(q,iq_dc, label='Ours')
        ax_iq.plot(q,iq_dp-iq_dc, c='r', label='Difference')
        ax_iq.set(xlabel='$Q \; [\AA^{-1}]$', ylabel='$I(Q) \; [counts]$')
        ax_iq.grid(alpha=0.2)
        ax_iq.legend()
        ax_iq.set_title('Scattering Intensity')
        
        ax_sas.plot(q_sas, iq_dp_full[:len(q_sas)], label='Diffpy-CMI')
        ax_sas.plot(q_sas, sas_dc, label='Ours')
        ax_sas.plot(q_sas, abs(iq_dp_full[:len(q_sas)]-sas_dc), c='r', label='Abs. Difference')
        ax_sas.set(xlabel='$Q \; [\AA^{-1}]$', ylabel='$I(Q) \; [counts]$')
        ax_sas.grid(alpha=0.2)
        ax_sas.legend()
        ax_sas.set_yscale('log')
        ax_sas.set_xscale('log')
        ax_sas.set_title('Small Angle Scattering Intensity')
        
        ax_fq.plot(q,fq_dp/max(fq_dp), label='Diffpy-CMI')
        ax_fq.plot(q,fq_dc/max(fq_dc), label='Ours')
        ax_fq.plot(q, fq_dp/max(fq_dp) - fq_dc/max(fq_dc) - 1, c='r', label='Difference')
        ax_fq.set(xlabel='$Q \; [\AA^{-1}]$', ylabel='$F(Q)\; [a.u.]$', yticks=[])
        ax_fq.grid(alpha=0.2)
        ax_fq.legend()
        ax_fq.set_title('Reduced Structure Function')
        
        ax_gr.plot(r,gr_dp/max(gr_dp), label='Diffpy-CMI')
        ax_gr.plot(r,gr_dc/max(gr_dc), label='Ours')
        ax_gr.plot(r, gr_dp/max(gr_dp) - gr_dc/max(gr_dc) - 0.5, c='r', label='Difference')
        ax_gr.set(xlabel='$r \; [\AA]$', ylabel='$G(r) \; [a.u.]$', yticks=[])
        ax_gr.grid(alpha=0.2)
        ax_gr.legend()
        ax_gr.set_title('Pair Distribution Function')
        
        fig.tight_layout()
        fig.savefig('../figures/figure2.png')
        plt.close(fig)
        print('Finished')
        print('-'*20+'\n')

    # Make figure 3
    if args['figure3']:
        print('-'*20)
        print('Generating Figure 3')
        # Load the structure file
        structure_path = args['cif']
        
        # Generate nanoparticles
        print('Generating particles')
        
        radii_cpu = np.arange(2,args['cpu_radius_max'],1) # Sample radii for cpu generation of nanoparticles (BEWARE OF MEMORY)
        radii_cuda = np.arange(2,args['cuda_radius_max'],1) # Sample radii for gpu/cuda generation of nanoparticles
        radii_diffpy = np.arange(2,args['diffpy_radius_max'],1) # Sample radii for Diffpy-CMI generation of nanoparticles
        
        nano_particles_CPU, nano_sizes_CPU = generate_nanoparticles(structure_path, radii_cpu)
        nano_particles_GPU, nano_sizes_GPU = generate_nanoparticles(structure_path, radii_cuda)
        nano_particles_CMI, nano_sizes_CMI = generate_nanoparticles(structure_path, radii_diffpy)
        
        def dummy_calculations():
            # Perform some dummy calculations on the GPU
            dummy_data = torch.rand(1000, 1000, device='cuda')
            for _ in range(100):
                dummy_data = torch.matmul(dummy_data, dummy_data)
        
        repetitions = args['repetitions']
        
        print('Profiling CPU')
        mu_DC_CPU, sigma_DC_CPU = [], []
        debye_calc_cpu = DebyeCalculator(device='cpu', verbose=1, qmin=1, qmax=25, qstep=0.1, biso=0.3, batch_size=args['batch_size_cpu'])
        for i in trange(len(radii_cpu), leave=False):
            timings_DC_CPU = []
            for _ in range(repetitions):
                debye_calc_cpu.profiler.reset()
                debye_calc_cpu.gr(nano_particles_CPU[i]);
                timings_DC_CPU.append(debye_calc_cpu.profiler.total())
            
            mu_DC_CPU.append(np.mean(timings_DC_CPU))
            sigma_DC_CPU.append(np.std(timings_DC_CPU))
        
        mu_DC_CPU = np.array(mu_DC_CPU)
        sigma_DC_CPU = np.array(sigma_DC_CPU)

        print('Finished\n')
        
        print('Profiling CUDA/GPU')
        # Check if a CUDA-enabled GPU is available
        if torch.cuda.is_available():
            # Move a dummy tensor to the GPU to initialize the CUDA context
            torch.cuda.FloatTensor(1).to('cuda')
        
            # Warm up the GPU
            for _ in range(10):
                dummy_calculations()
        
            # Profiling and Speed Test
            mu_DC_GPU, sigma_DC_GPU = [], []
            debye_calc = DebyeCalculator(device='cuda', verbose=1, qmin=1, qmax=25, qstep=0.1, biso=0.3, batch_size=args['batch_size_cuda'])
            for i in trange(len(radii_cuda), leave=False):
                timings_DC_GPU = []
                for _ in range(repetitions):
                    debye_calc.profiler.reset()
                    debye_calc.gr(nano_particles_GPU[i]);
                    timings_DC_GPU.append(debye_calc.profiler.total())
                mu_DC_GPU.append(np.mean(timings_DC_GPU))
                sigma_DC_GPU.append(np.std(timings_DC_GPU))
                
            mu_DC_GPU = np.array(mu_DC_GPU)
            sigma_DC_GPU = np.array(sigma_DC_GPU)
        else:
            print("No CUDA-enabled GPU found. Make sure you have a compatible GPU and PyTorch installed with CUDA support.")
        
        print('Finished\n')
        
        print('Profiling Diffpy-CMI')
        mu_CMI, sigma_CMI = [], []
        for i in trange(len(radii_diffpy), leave=False):
            timings_CMI = []
            for _ in range(repetitions):
                with tempfile.TemporaryDirectory() as tmpdirname:
                    
                    tmp_structure_path = os.path.join(tmpdirname,'tmp_struc.xyz')
                    write(tmp_structure_path, nano_particles_CMI[i], 'xyz')
                    
                    dbc = DebyePDFCalculator(rmin=debye_calc.rmin, rmax=debye_calc.rmax, rstep=debye_calc.rstep,
                                             qmin=debye_calc.qmin, qmax=debye_calc.qmax, qdamp=debye_calc.qdamp)
                    cmi_time = time()
                    diffpy_structure = loadStructure(tmp_structure_path)
                    diffpy_structure.B11 = debye_calc.biso
                    diffpy_structure.B22 = debye_calc.biso
                    diffpy_structure.B33 = debye_calc.biso
                    diffpy_structure.B12 = 0
                    diffpy_structure.B13 = 0
                    diffpy_structure.B23 = 0
                    dbc(diffpy_structure);
                    cmi_time = (time() - cmi_time)
                    timings_CMI.append(cmi_time)
            
            mu_CMI.append(np.mean(timings_CMI))
            sigma_CMI.append(np.std(timings_CMI))
        
        mu_CMI = np.array(mu_CMI)
        sigma_CMI = np.array(sigma_CMI)
        print('Finished\n')
        
        # Figure
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(11,4))
        
        _markersize = args['markersize']
        
        # Ours CPU
        p = ax1.plot(nano_sizes_CPU, mu_DC_CPU, marker='o', markerfacecolor='w', markersize=_markersize, lw=1, label = 'Ours, CPU')
        ax1.fill_between(nano_sizes_CPU, mu_DC_CPU-sigma_DC_CPU, mu_DC_CPU+sigma_DC_CPU, alpha=0.2, color=p[0].get_color())
        
        # Ours GPU
        p = ax1.plot(nano_sizes_GPU, mu_DC_GPU, marker='o', markerfacecolor='w', markersize=_markersize, lw=1, label = 'Ours, CUDA')
        ax1.fill_between(nano_sizes_GPU, mu_DC_GPU-sigma_DC_GPU, mu_DC_GPU+sigma_DC_GPU, alpha=0.2, color=p[0].get_color())
        
        # Diffpy
        p = ax1.plot(nano_sizes_CMI, mu_CMI, marker='o', markerfacecolor='w', markersize=_markersize, lw=1, label = 'Diffpy-CMI')
        ax1.fill_between(nano_sizes_CMI, mu_CMI-sigma_CMI, mu_CMI+sigma_CMI, alpha=0.2, color=p[0].get_color())
        
        ax1.set_ylabel('Generation Time [s]')
        ax1.set_xlabel('Structure diameter [Å]')  
        ax1.grid(alpha=0.2, which="both")
        ax1.legend()
        ax1.set_yscale('log')
        
        # Ours CPU
        p = ax2.plot([n.get_global_number_of_atoms() for n in nano_particles_CPU], mu_DC_CPU, marker='o', markerfacecolor='w', markersize=_markersize, lw=1, label = 'Ours CPU')
        ax2.fill_between([n.get_global_number_of_atoms() for n in nano_particles_CPU], mu_DC_CPU-sigma_DC_CPU, mu_DC_CPU+sigma_DC_CPU, alpha=0.2, color=p[0].get_color())
        
        # Ours GPU
        p = ax2.plot([n.get_global_number_of_atoms() for n in nano_particles_GPU], mu_DC_GPU, marker='o', markerfacecolor='w', markersize=_markersize, lw=1, label = 'Ours GPU')
        ax2.fill_between([n.get_global_number_of_atoms() for n in nano_particles_GPU], mu_DC_GPU-sigma_DC_GPU, mu_DC_GPU+sigma_DC_GPU, alpha=0.2, color=p[0].get_color())
        
        # Diffpy
        p = ax2.plot([n.get_global_number_of_atoms() for n in nano_particles_CMI], mu_CMI, marker='o', markerfacecolor='w', markersize=_markersize, lw=1, label = 'Diffpy-CMI')
        ax2.fill_between([n.get_global_number_of_atoms() for n in nano_particles_CMI], mu_CMI-sigma_CMI, mu_CMI+sigma_CMI, alpha=0.2, color=p[0].get_color())
        
        ax2.set_xlabel('Number of atoms')  
        ax2.grid(alpha=0.2, which="both")
        # Put a legend below current axis
        ax1.legend(loc='upper center', bbox_to_anchor=(1.1, 1.2),
                  fancybox=True, shadow=False, ncol=3)
        ax2.set_yscale('log')
        
        # fig.tight_layout()
        
        fig.savefig('../figures/figure3.png', bbox_inches='tight', pad_inches=0.5)
        plt.close(fig)
        print('Finished')
        print('-'*20)
    
if __name__ == '__main__':	
    parser = argparse.ArgumentParser()
    parser.add_argument('--cif', type=str, required=True)
    parser.add_argument('--cpu_radius_max', type=float, default=5)
    parser.add_argument('--batch_size_cpu', type=int, default=1000)
    parser.add_argument('--cuda_radius_max', type=float, default=5)
    parser.add_argument('--batch_size_cuda', type=int, default=5000)
    parser.add_argument('--diffpy_radius_max', type=float, default=5)
    parser.add_argument('--figure2', action='store_true')
    parser.add_argument('--figure3', action='store_true')
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--markersize', type=int, default=5)
    
    args = parser.parse_args()
    main(vars(args))
