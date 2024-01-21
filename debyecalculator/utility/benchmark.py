import tempfile
import os
import csv
import pkg_resources
import argparse
import warnings
import torch
from time import time
import numpy as np
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt
from ase.io import write, read
from debyecalculator import DebyeCalculator
from debyecalculator.utility.generate import generate_nanoparticles
from prettytable import PrettyTable, from_csv

from typing import Union, List, Any
from collections import namedtuple

class Statistics:
    """
    A class to store and represent benchmark statistics.

    Attributes:
    - means (List[float]): List of mean values.
    - stds (List[float]): List of standard deviation values.
    - name (str): Name of the statistics.
    - radii (List[float]): List of radii.
    - num_atoms (List[int]): List of number of atoms.
    - cuda_mem (List[float]): List of CUDA memory values.
    - device (str): Device used for benchmarking.
    - batch_size (int): Batch size used for benchmarking.
    """
    def __init__(
        self,
        name: str,
        function_name: str,
        device: str,
        batch_size: int,
        radii: List[float],
        num_atoms: List[int],
        means: List[float],
        stds: List[float],
        cuda_mem_structure: List[float],
        cuda_mem_calculations: List[float],
    ) -> None:
        """
        Initialize Statistics with benchmarking results.

        Parameters:
        - name (str): Name of the statistics.
        - function_name (str) Name of function to be benchmarked
        - device (str): Device used for benchmarking.
        - batch_size (int): Batch size used for benchmarking.
        - means (List[float]): List of mean values.
        - stds (List[float]): List of standard deviation values.
        - radii (List[float]): List of radii.
        - num_atoms (List[int]): List of number of atoms.
        - cuda_mem_structure (List[float]): List of CUDA memory values for generating structures.
        - cuda_mem_calculations (List[float]): List of CUDA memory values for calculating the function.
        """

        self.name = name
        self.function_name = function_name
        self.device = device
        self.batch_size = batch_size
        if self.batch_size == 0:
            self.batch_size_str = 'N/A'
        else:
            self.batch_size_str = str(self.batch_size) 
        self.means = means
        self.stds = stds
        self.radii = radii
        self.num_atoms = num_atoms
        self.cuda_mem_structure = cuda_mem_structure
        self.cuda_mem_calculations = cuda_mem_calculations
        
        # Create table
        self.table_fields = ['Radius [Å]', 'Num. atoms', 'Mean [s]', 'Std [s]', 'MaxAllocCUDAMem (Gen.) [MB]', 'MaxAllocCUDAMem (Calc.) [MB]']
        self.pt = PrettyTable(self.table_fields)
        self.pt.align = 'r'
        self.pt.padding_width = 1
        self.pt.title = self.name + ' / ' + self.function_name + ' / DEVICE:' + self.device.upper() + ' / BATCH SIZE: ' + self.batch_size_str
        self.data = [[str(float(r)), str(int(n)), f'{m:1.5f}', f'{s:1.5f}', f'{cs:1.5f}', f'{cc:1.5f}'] for r,n,m,s,cs,cc in zip(self.radii, list(num_atoms), list(means), list(stds), list(cuda_mem_structure), list(cuda_mem_calculations))]
        for d in self.data:
            self.pt.add_row(d)

    def __str__(self) -> str:
        """
        Return a PrettyTable Statistics table.
        """
        return str(self.pt)

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the Statistics object.
        """
        return f'Statistics (\n\tname = {self.name},\n\tfunction_name = {self.function_name},\n\tdevice = {self.device},\n\tbatch_size = {self.batch_size},\n\tradii = {self.radii},\n\tnum_atoms = {self.num_atoms},\n\tmeans = {self.means},\n\tstds = {self.stds},\n\tcuda_mem_structure = {self.cuda_mem_structure},\n\tcuda_mem_calculations = {self.cuda_mem_calculations},\n)'

class DebyeBenchmarker:
    """
    A class for benchmarking Debye calculations.

    Attributes:
    - radii (List[float]): List of radii for benchmarking.
    - cif (str): Path to reference CIF file used for benchmarking.
    - custom_cif (str): Custom CIF file path (if provided).
    - show_progress_bar (bool): Flag to control progress bar display.
    - debye_calc (DebyeCalculator): Debye calculator instance.
    """
    def __init__(
        self,
        function: str = 'gr',
        radii: Union[List, np.ndarray, torch.Tensor] = [5],
        show_progress_bar: bool = True,
        custom_cif: str = None,
        **kwargs,
    ) -> None:
        """
        Initialize DebyeBenchmarker.

        Parameters:
        - radii (Union[List, np.ndarray, torch.Tensor]): List of radii for benchmarking.
        - show_progress_bar (bool): Flag to control progress bar display.
        - custom_cif (str): Custom CIF file path (if provided).
        - **kwargs: Additional keyword arguments for DebyeCalculator.
        """

        self.set_radii(list(radii))
        self.cif = pkg_resources.resource_filename(__name__, 'benchmark_structure.cif')
        self.custom_cif = custom_cif
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.debye_calc = DebyeCalculator(**kwargs)
            
        self.function_name = function
        if function == 'gr':
            self.func = self.debye_calc.gr
        elif function == 'iq':
            self.func = self.debye_calc.iq
        elif function == 'sq':
            self.func = self.debye_calc.sq
        else:
            raise ValueError("Invalid value for 'function', please provide either 'gr', 'iq' or 'sq'")

        self.show_progress_bar = show_progress_bar

        self.ref_stat_csv_titan_10k = pkg_resources.resource_filename(__name__, 'benchmark_reference_TITANRTX_10k.csv')
        self.reference_stat_titan_10k = from_csv(self.ref_stat_csv_titan_10k)
        self.reference_stat_titan_10k.name = 'TITAN RTX'
        
        self.ref_stat_csv_diffpy = pkg_resources.resource_filename(__name__, 'benchmark_reference_DiffPy.csv')
        self.reference_stat_diffpy = from_csv(self.ref_stat_csv_diffpy)
        self.reference_stat_diffpy.name = 'DiffPy'

    def set_debye_parameters(self, **debye_parameters: Any) -> None:
        """
        Set Debye parameters for the calculator.

        Parameters:
        - **debye_parameters: Keyword arguments for Debye parameters.
        """
        self.debye_calc.update_parameters(debye_parameters)

    def set_device(self, device: str) -> None:
        """
        Set the device for Debye calculations.

        Parameters:
        - device (str): Device to be set for calculations.
        """
        self.debye_calc.update_parameters(device=device)

    def set_batch_size(self, batch_size: int) -> None:
        """
        Set the batch size for Debye calculations.

        Parameters:
        - batch_size (int): Batch size for calculations.
        """
        self.debye_calc.update_parameters(batch_size=batch_size)

    def set_radii(self, radii: Union[List, np.ndarray, torch.Tensor]) -> None:
        """
        Set the radii for benchmarking.

        Parameters:
        - radii (Union[List, np.ndarray, torch.Tensor]): List of radii for benchmarking.
        """
        self.radii = list(radii)

    def get_reference_stat_titan(self):
        return self.reference_stat_titan

    def get_reference_stat_diffpy(self):
        return self.reference_stat_diffpy
    
    def benchmark(
        self,
        generate_individually: bool = True,
        repetitions: int = 1,
        dummy_repititions: int = 2,
    ) -> Statistics:
        """
        Benchmark DebyeCalculator.

        Parameters:
        - generate_individually (bool): Flag to benchmark individually for each radius.
        - repetitions (int): Number of repetitions for benchmarking.

        Returns:
        - Statistics: Benchmark statistics.
        """
        
        # Ignore warnings concerning CUDA availability
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Using CUDA?
            on_cuda = self.debye_calc.device == 'cuda'
             
            # Create metrics arrays
            means = np.zeros(len(self.radii))
            stds = np.zeros(len(self.radii))
            num_atoms = np.zeros(len(self.radii))
            cuda_mem_structure = np.zeros(len(self.radii))
            cuda_mem_calculations = np.zeros(len(self.radii))

            # Create nanoparticles seperate, such that exact metrics can be extracted
            cif_file = self.custom_cif if self.custom_cif is not None else self.cif
            name = cif_file.split('/')[-1]
            
            # Iterator
            if generate_individually:
                nanoparticles = lambda i: generate_nanoparticles(cif_file, self.radii[i], _reverse_order=False, disable_pbar = True, device = self.debye_calc.device, _benchmarking=True)
            else:
                if on_cuda: torch.cuda.reset_max_memory_allocated()
                nanoparticles = generate_nanoparticles(cif_file, self.radii, _reverse_order=False, disable_pbar = True, device=self.debye_calc.device, _benchmarking=True)
                mean_cuda_mem_structure = torch.cuda.max_memory_allocated() / 1_000_000 if on_cuda else 0

            # Benchmark
            pbar = tqdm(desc='Benchmarking Calculator...', total=len(self.radii), disable = not self.show_progress_bar)
            for i in range(len(self.radii)):

                # Reset memory allocation count
                if on_cuda: torch.cuda.reset_max_memory_allocated()
                    
                # Fetch nanoparticle
                nano = nanoparticles(i)[0] if generate_individually else nanoparticles[i]

                # Collect memory allocation
                if generate_individually:
                    cuda_mem_structure[i] = torch.cuda.max_memory_allocated() / 1_000_000 if on_cuda else 0
                else:
                    cuda_mem_structure[i] = mean_cuda_mem_structure

                # Lists
                times = []
                mems_calculations = []
                for j in range(repetitions+dummy_repititions):
                
                    # Reset allocation
                    if on_cuda: torch.cuda.reset_max_memory_allocated()
                    
                    # Time calculation of gr (encompases all possible calculations of DebyeCalculator)
                    t = time()
                    data = self.func((nano.elements, nano.xyz))
                    t = time() - t

                    # Append only after dummy repetitions
                    if j > dummy_repititions-1:
                        times.append(t)

                    # Append memory allocation
                    mems_calculations.append(torch.cuda.max_memory_allocated() / 1_000_000) if on_cuda else mems_calculations.append(0)

                # Collect metrics
                means[i] = np.mean(times)
                stds[i] = np.std(times)
                num_atoms[i] = nano.size
                cuda_mem_calculations[i] = np.mean(mems_calculations)

                pbar.update(1)
            pbar.close()

        return Statistics(
            name = name,
            function_name = self.function_name,
            device = self.debye_calc.device,
            batch_size = self.debye_calc.batch_size,
            radii = self.radii, 
            num_atoms = list(num_atoms),
            means = list(means),
            stds = list(stds),
            cuda_mem_structure = list(cuda_mem_structure),
            cuda_mem_calculations = list(cuda_mem_calculations)
        )

def to_csv(stat: Statistics, path: str) -> None:
    """
    Save Statistics instance to a CSV file.

    Parameters:
    - stat (Statistics): Statistics instance to be saved.
    - path (str): Path to save the CSV file.
    """
    metadata = []
    metadata.insert(0, f'# NAME {stat.name}')
    metadata.insert(1, f'# FUNCTION NAME {stat.function_name}')
    metadata.insert(2, f'# DEVICE {stat.device}')
    metadata.insert(3, f'# BATCH SIZE {stat.batch_size}')

    with open(path, 'w', newline='') as f:
        for md in metadata:
            f.writelines(md + '\n')
        f.write(stat.pt.get_csv_string())

def from_csv(path: str) -> Statistics:
    """
    Load Statistics instance from a CSV file.

    Parameters:
    - path (str): Path to the CSV file.

    Returns:
    - Statistics: Loaded Statistics instance.
    """
    name = 'N/A'
    function_name = 'N/A'
    device = 'N/A'
    batch_size = 0

    with open(path, 'r') as f:
        while True:
            line = f.readline().strip()
            if line.startswith('# NAME'):
                name = line.split('# NAME')[-1]
            elif line.startswith('# FUNCTION NAME'):
                function_name = line.split('# FUNCTION NAME')[-1]
            elif line.startswith('# DEVICE'):
                device = line.split('# DEVICE')[-1]
            elif line.startswith('# BATCH SIZE'):
                batch_size = int(line.split('# BATCH SIZE')[-1])
            else:
                break
        data_lines = f.readlines()

    try:
        csv_reader = csv.reader(data_lines, delimiter=',')
    except:
        raise IOError('Error in reading CSV file')

    radii = []
    num_atoms = []
    means = []
    stds = []
    cuda_mem_structure = []
    cuda_mem_calculations = []

    for row in csv_reader:
        radii.append(float(row[0]))
        num_atoms.append(int(row[1]))
        means.append(float(row[2]))
        stds.append(float(row[3]))
        cuda_mem_structure.append(float(row[4]))
        cuda_mem_calculations.append(float(row[5]))

    return Statistics(
        name = name,
        function_name = function_name,
        device = device,
        batch_size = batch_size,
        radii = radii,
        num_atoms = num_atoms,
        means = means,
        stds = stds,
        cuda_mem_structure = cuda_mem_structure,
        cuda_mem_calculations = cuda_mem_calculations,
    )

def plot_time_statistics(
    statistics: List[Statistics] = [],
    labels: Union[List[str], None] = None,
    figsize: tuple = (8,6),
    include_references: bool = False,
    log_scale: bool = True,
    return_fig: bool = False,
) -> Union[None, plt.figure]:
    """
    Plots time statistics

    Parameters:
    - statistics (List[Statistics]): List of Statistics objects
    - labels (Union[List[str], None]): List of labels for the plot. If None, labels are generated from statistics.
    - figsize (tuple): Figure size, default is (8, 6).
    - include_references (bool): Whether to include reference data, default is False.
    - log_scale (bool): Whether to use a logarithmic scale for the y-axis, default is True.
    - return_fig (bool): If True, returns the matplotlib figure instead of displaying it, default is False.

    Returns:
    - Union[None, plt.figure]: If return_fig is True, returns the matplotlib figure. Otherwise, displays the plot.
    """
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=figsize)
    
    if include_references:
        benchmarker = DebyeBenchmarker()
        loaded_stats_diffpy = benchmarker.get_reference_stat_diffpy()
        loaded_stats_titan = benchmarker.get_reference_stat_titan()
        
        means_diffpy = np.array(loaded_stats_diffpy.means)
        stds_diffpy = np.array(loaded_stats_diffpy.stds)
        means_titan = np.array(loaded_stats_titan.means)
        stds_titan = np.array(loaded_stats_titan.stds)
        
        p = ax1.plot(loaded_stats_diffpy.radii, means_diffpy, label='DiffPy', marker='o')
        ax1.fill_between(loaded_stats_diffpy.radii, means_diffpy - stds_diffpy, means_diffpy + stds_diffpy, 
                         color=p[0].get_color(), alpha=0.2)

        p = ax1.plot(loaded_stats_titan.radii, means_titan, label='TITAN RTX, Batch size: 10k', marker='o')
        ax1.fill_between(loaded_stats_titan.radii, means_titan - stds_titan, means_titan + stds_titan, 
                         color=p[0].get_color(), alpha=0.2)
        
        p = ax2.plot(loaded_stats_diffpy.num_atoms, means_diffpy, label='DiffPy', marker='o')
        ax2.fill_between(loaded_stats_diffpy.num_atoms, means_diffpy - stds_diffpy, means_diffpy + stds_diffpy, 
                         color=p[0].get_color(), alpha=0.2)

        p = ax2.plot(loaded_stats_titan.num_atoms, means_titan, label='TITAN RTX, Batch size: 10k', marker='o')
        ax2.fill_between(loaded_stats_titan.num_atoms, means_titan - stds_titan, means_titan + stds_titan, 
                         color=p[0].get_color(), alpha=0.2)
        
    if labels == None:
        labels = [' '.join([stat.function_name, stat.device.upper(), stat.batch_size_str]) for stat in statistics]

    for stat,label in zip(statistics, labels):
        means = np.array(stat.means)
        stds = np.array(stat.stds)
        
        p = ax1.plot(stat.radii, means, label=label, marker='o')
        ax1.fill_between(stat.radii, means - stds, means + stds, color=p[0].get_color(), alpha=0.2)
        
        p = ax2.plot(stat.num_atoms, means, label=label, marker='o')
        ax2.fill_between(stat.num_atoms, means - stds, means + stds, color=p[0].get_color(), alpha=0.2)

    ax1.set_xlabel('Nanoparticle Radius (Å)')
    ax2.set_xlabel('Number of atoms')
    
    for ax in [ax1, ax2]:
        ax.set_ylabel('Calculation Time (s)')
        ax.legend()
        ax.grid(alpha=0.2)
        if log_scale:
            ax.set_yscale('log')
            
    fig.tight_layout()
    
    if return_fig:
        plt.close(fig)
        return fig
    else:
        plt.show()
        return

def plot_memory_statistics(
    statistics: List[Statistics] = [],
    labels: Union[List[str], None] = None,
    figsize: tuple = (8,8),
    include_references: bool = False,
    log_scale: bool = True,
    return_fig: bool = False,
) -> Union[None, plt.figure]:
    """
    Plots memory statistics
    
    Parameters:
    - statistics (List[Statistics]): List of Statistics objects
    - labels (Union[List[str], None]): List of labels for the plot. If None, labels are generated from statistics.
    - figsize (tuple): Figure size, default is (8, 8).
    - include_references (bool): Whether to include reference data, default is False.
    - log_scale (bool): Whether to use a logarithmic scale for the y-axis, default is True.
    - return_fig (bool): If True, returns the matplotlib figure instead of displaying it, default is False.

    Returns:
    - Union[None, plt.figure]: If return_fig is True, returns the matplotlib figure. Otherwise, displays the plot.
    """
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=figsize)
    
    if include_references:
        benchmarker = DebyeBenchmarker()
        loaded_stats_titan = benchmarker.get_reference_stat_titan()
        
        cuda_mem_structure_titan = loaded_stats_titan.cuda_mem_structure
        cuda_mem_calculations_titan = loaded_stats_titan.cuda_mem_calculations
        
        radii_titan = np.array(loaded_stats_titan.radii)
        
        ax1.plot(radii_titan, cuda_mem_structure_titan, label='TITAN RTX, Batch size: 10k', marker='o')
        ax1.legend()
        ax1.grid(alpha=0.2)
        
        ax2.plot(radii_titan, cuda_mem_calculations_titan, label='TITAN RTX, Batch size: 10k', marker='o')
        ax2.legend()
        ax2.grid(alpha=0.2)
        
    if labels == None:
        labels = [' '.join([stat.function_name, stat.device.upper(), stat.batch_size_str]) for stat in statistics]
        
    for stat, label in zip(statistics, labels):
        cuda_mem_structure = stat.cuda_mem_structure
        cuda_mem_calculations = stat.cuda_mem_calculations
        
        radii = np.array(stat.radii)
        
        ax1.plot(radii, cuda_mem_structure, label=label, marker='o')
        ax1.legend()
        ax1.grid(alpha=0.2)
        
        ax2.plot(radii, cuda_mem_calculations, label=label, marker='o')
        ax2.legend()
        ax2.grid(alpha=0.2)
        
    ax1.set_title('Comparison of Maximum CUDA Memory Usage (Generation)')
    ax2.set_title('Comparison of Maximum CUDA Memory Usage (Calculation)')
    
    for ax in [ax1, ax2]:
        ax.set_xlabel('Nanoparticle Radius (Å)')
        ax.set_ylabel('Max CUDA Memory Usage (MB)')
        if log_scale:
            ax.set_yscale('log')
    
    fig.tight_layout()
    
    if return_fig:
        plt.close(fig)
        return fig
    else:
        plt.show()
        return
