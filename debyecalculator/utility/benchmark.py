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
        means: List[float],
        stds: List[float],
        name: str,
        radii: List[float],
        num_atoms: List[int],
        cuda_mem: List[float],
        device: str,
        batch_size: int
    ) -> None:
        """
        Initialize Statistics with benchmarking results.

        Parameters:
        - means (List[float]): List of mean values.
        - stds (List[float]): List of standard deviation values.
        - name (str): Name of the statistics.
        - radii (List[float]): List of radii.
        - num_atoms (List[int]): List of number of atoms.
        - cuda_mem (List[float]): List of CUDA memory values.
        - device (str): Device used for benchmarking.
        - batch_size (int): Batch size used for benchmarking.
        """

        self.means = means
        self.stds = stds
        self.name = name
        self.radii = radii
        self.num_atoms = num_atoms
        self.cuda_mem = cuda_mem
        self.device = device
        self.batch_size = batch_size
        
        # Create table
        self.table_fields = ['Radius [Å]', 'Num. atoms', 'Mean [s]', 'Std [s]', 'MaxAlloc. CUDA mem [MB]']
        self.pt = PrettyTable(self.table_fields)
        self.pt.align = 'r'
        self.pt.padding_width = 1
        self.pt.title = 'Benchmark Generator / ' + self.device + ' / Batch Size: ' + str(self.batch_size)
        self.data = [[str(float(r)), str(int(n)), f'{m:1.5f}', f'{s:1.5f}', f'{c:1.5f}'] for r,n,m,s,c in zip(self.radii, list(num_atoms), list(means), list(stds), list(cuda_mem))]
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
        return f'Statistics (\n\tname = {self.name},\n\tradii = {self.radii},\n\tnum_atoms = {self.num_atoms},\n\tmeans = {self.means},\n\tstds = {self.stds},\n\tcuda_mem = {self.cuda_mem},\n\tbatch_size = {self.batch_size},\n\tdevice = {self.device}\n)'

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

        self.show_progress_bar = show_progress_bar
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.debye_calc = DebyeCalculator(**kwargs)

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
    
    def benchmark_generation(
        self,
        individually: bool = False,
        repetitions: int = 1,
    ) -> Statistics:
        """
        Benchmark nanoparticle generation.

        Parameters:
        - individually (bool): Flag to benchmark individually for each radius.
        - repetitions (int): Number of repetitions for benchmarking.

        Returns:
        - Statistics: Benchmark statistics.
        """
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Create metrics arrays
            means = np.zeros(len(self.radii))
            stds = np.zeros(len(self.radii))
            num_atoms = np.zeros(len(self.radii))
            cuda_mem = np.zeros(len(self.radii))
            
            cif_file = self.custom_cif if self.custom_cif is not None else self.cif
            name = cif_file.split('/')[-1]
            
            if not individually:
                times = []
                mems = []
                if self.debye_calc.device == 'cuda':
                    torch.cuda.reset_max_memory_allocated()
                for j in range(repetitions + 2):
                    t = time()
                    nanoparticles = generate_nanoparticles(cif_file, self.radii, _reverse_order=False, disable_pbar = True, device=self.debye_calc.device)
                    t = time() - t
                    if j > 1:
                        times.append(t)

                    if self.debye_calc.device == 'cuda':
                        mems.append(torch.cuda.max_memory_allocated() / 1_000_000)
                    else:
                        mems.append(0)

                # Collect metrics
                means[:] = np.mean(times) / len(self.radii)
                stds[:] = np.std(times) / len(self.radii)
                num_atoms = [n.size for n in nanoparticles]
                cuda_mem[:] = np.mean(mems)
            else:
                pbar = tqdm(desc='Benchmarking Nanoparticle Generation...', total=len(self.radii), disable = not self.show_progress_bar)
                for i in range(len(self.radii)):
                    times = []
                    mems = []
                    if self.debye_calc.device == 'cuda':
                        torch.cuda.reset_max_memory_allocated()
                    for j in range(repetitions + 2):
                        t = time()
                        nanoparticles = generate_nanoparticles(cif_file, [self.radii[i]], _reverse_order=False, disable_pbar = True, device=self.debye_calc.device)
                        t = time() - t
                        if j > 1:
                            times.append(t)

                        if self.debye_calc.device == 'cuda':
                            mems.append(torch.cuda.max_memory_allocated() / 1_000_000)
                        else:
                            mems.append(0)

                    # Collect metrics
                    means[i] = np.mean(times)
                    stds[i] = np.std(times)
                    num_atoms[i] = nanoparticles[0].size
                    cuda_mem[i] = np.mean(mems)
                    
                    pbar.update(1)
                pbar.close()
        
        return Statistics(list(means), list(stds), name, self.radii, list(num_atoms), list(cuda_mem), self.debye_calc.device, self.debye_calc.batch_size)

    def benchmark_calculator(
        self,
        repetitions: int = 1,
    ) -> Statistics:
        """
        Benchmark Debye calculator.

        Parameters:
        - repetitions (int): Number of repetitions for benchmarking.
        - csv_output (Union[None, str]): Path to save CSV output (if provided).

        Returns:
        - Statistics: Benchmark statistics.
        """
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
             
            # Create metrics arrays
            means = np.zeros(len(self.radii))
            stds = np.zeros(len(self.radii))
            num_atoms = np.zeros(len(self.radii))
            cuda_mem = np.zeros(len(self.radii))

            # Create nanoparticles seperate, such that exact metrics can be extracted
            cif_file = self.custom_cif if self.custom_cif is not None else self.cif
            name = cif_file.split('/')[-1]
            nanoparticles = generate_nanoparticles(cif_file, self.radii, _reverse_order=False, disable_pbar = True, device=self.debye_calc.device)

            # Benchmark
            pbar = tqdm(desc='Benchmarking Calculator...', total=len(self.radii), disable = not self.show_progress_bar)
            for i,nano in enumerate(nanoparticles):
                times = []
                mems = []
                if self.debye_calc.device == 'cuda':
                    torch.cuda.reset_max_memory_allocated()
                for j in range(repetitions+2):

                    t = time()
                    data = self.debye_calc.gr((nano.elements, nano.xyz))
                    t = time() - t
                    if j > 1:
                        times.append(t)

                    if self.debye_calc.device == 'cuda':
                        mems.append(torch.cuda.max_memory_allocated() / 1_000_000)
                    else:
                        mems.append(0)

                # Collect metrics
                means[i] = np.mean(times)
                stds[i] = np.std(times)
                num_atoms[i] = nano.size
                cuda_mem[i] = np.mean(mems)

                pbar.update(1)
            pbar.close()

        return Statistics(list(means), list(stds), name, self.radii, list(num_atoms), list(cuda_mem), self.debye_calc.device, self.debye_calc.batch_size)

def to_csv(stat: Statistics, path: str) -> None:
    """
    Save Statistics instance to a CSV file.

    Parameters:
    - stat (Statistics): Statistics instance to be saved.
    - path (str): Path to save the CSV file.
    """
    metadata = []
    metadata.insert(0, f'# NAME {stat.name}')
    metadata.insert(1, f'# DEVICE {stat.device}')
    metadata.insert(2, f'# BATCH SIZE {stat.batch_size}')

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
    device = 'N/A'
    batch_size = 0

    with open(path, 'r') as f:
        while True:
            line = f.readline().strip()
            if line.startswith('# NAME'):
                name = line.split('# NAME')[-1]
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
    cuda_mem = []
    for row in csv_reader:
        radii.append(float(row[0]))
        num_atoms.append(int(row[1]))
        means.append(float(row[2]))
        stds.append(float(row[3]))
        cuda_mem.append(float(row[4]))
    return Statistics(means, stds, name, radii, num_atoms, cuda_mem, device, batch_size)
