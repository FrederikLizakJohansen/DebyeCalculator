import os
import sys
import base64
import yaml
import pkg_resources
import warnings
from glob import glob
from datetime import datetime
from typing import Union, Tuple, Any, List

import torch
from torch import cdist
from torch.nn.functional import pdist

import numpy as np
import matplotlib.pyplot as plt

from ase import Atoms
from ase.io import read, write
from ase.build import make_supercell
from ase.build.tools import sort as ase_sort

from debyecalculator.utility.profiling import Profiler

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from ipywidgets import HBox, VBox, Layout
from tqdm.auto import tqdm

import collections
import timeit

class DebyeCalculator:
    """
    Calculate the scattering intensity I(q) through the Debye scattering equation, the Total Scattering Structure Function S(q), 
    the Reduced Total Scattering Function F(q), and the Reduced Atomic Pair Distribution Function G(r) for a given atomic structure.


    Parameters:
        qmin (float): Minimum q-value for the scattering calculation. Default is 1.0.
        qmax (float): Maximum q-value for the scattering calculation. Default is 30.0.
        qstep (float): Step size for the q-values in the scattering calculation. Default is 0.1.
        qdamp (float): Damping parameter caused by the truncated Q-range of the Fourier transformation. Default is 0.04.
        rmin (float): Minimum r-value for the pair distribution function (PDF) calculation. Default is 0.0.
        rmax (float): Maximum r-value for the PDF calculation. Default is 20.0.
        rstep (float): Step size for the r-values in the PDF calculation. Default is 0.01.
        rthres (float): Threshold value for exclusion of distances below this value in the scattering calculation. Default is 0.0.
        biso (float): Debye-Waller isotropic atomic displacement parameter. Default is 0.3.
        device (str): Device to use for computation (e.g., 'cuda' for GPU or 'cpu' for CPU). Default is 'cuda' if the computer has a GPU.
        batch_size (int or None): Batch size for computation. If None, the batch size will be automatically set. Default is None.
        lorch_mod (bool): Flag to enable Lorch modification. Default is False.
        radiation_type (str): Type of radiation for form factor calculations ('xray' or 'neutron'). Default is 'xray'.
        profile (bool): Activate profiler. Default is False.
    """

    def __init__(
        self,
        qmin: float = 1.0,
        qmax: float = 30.0,
        qstep: float = 0.1,
        qdamp: float = 0.04,
        rmin: float = 0.0,
        rmax: float = 20.0,
        rstep: float = 0.01,
        rthres: float = 0.0,
        biso: float = 0.3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        batch_size: Union[int, None] = 10000,
        lorch_mod: bool = False,
        radiation_type: str = 'xray',
        profile: bool = False,
        _max_batch_size: int = 4000,
    ) -> None:

        self.profile = profile
        if self.profile:
            self.profiler = Profiler()
        
        # Initial parameters
        self.device = device
        self.batch_size = batch_size
        self.lorch_mod = lorch_mod
        self.radiation_type = radiation_type
        
        # Standard Debye parameters
        self.qmin = qmin
        self.qmax = qmax
        self.qstep = qstep
        self.qdamp = qdamp
        self.rmin = rmin
        self.rmax = rmax
        self.rstep = rstep
        self.rthres = rthres
        self.biso = biso

        # Initialise ranges
        self.q = torch.arange(self.qmin, self.qmax, self.qstep).unsqueeze(-1).to(device=self.device)
        self.r = torch.arange(self.rmin, self.rmax, self.rstep).unsqueeze(-1).to(device=self.device)

        # Form factor coefficients
        with open(pkg_resources.resource_filename(__name__, 'form_factor_coef.yaml'), 'r') as yaml_file:
            self.FORM_FACTOR_COEF = yaml.safe_load(yaml_file)

        # Formfactor retrieval lambda
        for k,v in self.FORM_FACTOR_COEF.items():
            if None in v:
                v = [value if value is not None else np.nan for value in v]
            self.FORM_FACTOR_COEF[k] = torch.tensor(v).to(device=self.device, dtype=torch.float32)
        if radiation_type.lower() in ['xray', 'x']:
            self.form_factor_func = lambda p: torch.sum(p[:5] * torch.exp(-1*p[6:11] * (self.q / (4*torch.pi)).pow(2)), dim=1) + p[5]
        elif radiation_type.lower() in ['neutron', 'n']:
            self.form_factor_func = lambda p: p[11].unsqueeze(-1)

        # Batch size
        self._max_batch_size = _max_batch_size

    def __repr__(
        self,
    ):
        parameters = {'qmin': self.qmin, 'qmax': self.qmax, 'qdamp': self.qdamp, 'qstep': self.qstep,
                      'rmin': self.rmin, 'rmax': self.rmax, 'rstep': self.rstep, 'rthres': self.rthres,
                      'biso': self.biso}

        return f"DebyeCalculator{parameters}"
    
    def update_parameters(
        self,
        **kwargs: Any,
    ) -> None:
        """
        Set or update the parameters of the DebyeCalculator.

        Parameters:
            **kwargs: Arbitrary keyword arguments to update the parameters.
        """
        for k,v in kwargs.items():
            try:
                setattr(self, k, v)
            except:
                print("Failed to update parameters because of unexpected parameter names")
                return
            
        # Re-initialise ranges
        if np.any([k in ['qmin','qmax','qstep','rmin', 'rmax', 'rstep'] for k in kwargs.keys()]):
            self.q = torch.arange(self.qmin, self.qmax, self.qstep).unsqueeze(-1).to(device=self.device)
            self.r = torch.arange(self.rmin, self.rmax, self.rstep).unsqueeze(-1).to(device=self.device)

    def _initialise_structures(
        self,
        structure_path: Union[str, Atoms, List[Atoms]],
        radii: Union[List[float], float, None] = None,
        disable_pbar: bool = False,
    ) -> None:

        """
        Initialise atomic structures and unique elements form factors from an input file.

        Parameters:
            structure_path (Union[str, Atoms, List[Atoms]]): Path to the atomic structure file in XYZ/CIF format or stored ASE Atoms objects.
            radii (Union[List[float], float, None]): List/float of radii/radius of particle(s) to generate with parsed CIF
        """
        # Check if input is a file or ASE Atoms object
        if isinstance(structure_path, str):
            # Check file and extention
            structure_ext = structure_path.split('.')[-1]
            if structure_ext not in ['xyz', 'cif']:
                raise TypeError('FAILED: Invalid file/file-extention, accepts only .xyz or .cif data files')
        elif isinstance(structure_path, Atoms) or all(isinstance(lst_elm, Atoms) for lst_elm in structure_path):
            structure_ext = 'ase'
        else:
            raise TypeError('FAILED: Invalid structure format, accepts only .xyz, .cif data files or ASE Atoms objects')

        # If cif, check for radii and generate particles
        if structure_ext == 'cif':
            if radii is not None:
                ase_structures, _ = self.generate_nanoparticles(structure_path, radii, disable_pbar=disable_pbar)
                self.num_structures = len(ase_structures)
            else:
                raise ValueError('FAILED: When providing .cif data file, please provide radii (Union[List[float], float]) to generate from.')

            self.struc_elements = []
            self.struc_size = []
            self.struc_occupancy = []
            self.struc_xyz = []
            
            for structure in ase_structures:
                elements = structure.get_chemical_symbols()
                size = len(elements)
                occupancy = torch.ones((size), dtype=torch.float32).to(device=self.device)
                xyz = torch.tensor(np.array(structure.get_positions())).to(device=self.device, dtype=torch.float32)

                self.struc_elements.append(elements)
                self.struc_size.append(size)
                self.struc_occupancy.append(occupancy)
                self.struc_xyz.append(xyz)

        elif structure_ext == 'xyz':
                
            self.num_structures = 1
            struc = np.genfromtxt(structure_path, dtype='str', skip_header=2) # Gen
            self.struc_elements = [struc[:,0]] # Identities
            self.struc_size = [len(self.struc_elements[0])] # Size

            # Append occupancy if nothing is provided
            if struc.shape[1] == 5:
                self.struc_occupancy = [torch.from_numpy(struc[:,-1]).to(device=self.device, dtype=torch.float32)]
                self.struc_xyz = [torch.tensor(struc[:,1:-1].astype('float')).to(device=self.device, dtype=torch.float32)]
            else:
                self.struc_occupancy = [torch.ones((self.struc_size[0]), dtype=torch.float32).to(device=self.device)]
                self.struc_xyz = [torch.tensor(struc[:,1:].astype('float')).to(device=self.device, dtype=torch.float32)]
        elif structure_ext == 'ase':
            if isinstance(structure_path, Atoms):
                ase_structures = [structure_path]
            else:
                ase_structures = structure_path
            
            self.num_structures = len(ase_structures)
            
            self.struc_elements = []
            self.struc_size = []
            self.struc_occupancy = []
            self.struc_xyz = []
            
            for structure in ase_structures:
                elements = structure.get_chemical_symbols()
                size = len(elements)
                occupancy = torch.ones((size), dtype=torch.float32).to(device=self.device)
                xyz = torch.tensor(np.array(structure.get_positions())).to(device=self.device, dtype=torch.float32)

                self.struc_elements.append(elements)
                self.struc_size.append(size)
                self.struc_occupancy.append(occupancy)
                self.struc_xyz.append(xyz)
        else:
            raise TypeError('FAILED: Invalid structure format, accepts only .xyz, .cif data files or ASE Atoms objects')

        # Unique elements and their counts
        self.triu_indices = []
        self.unique_inverse = []
        self.struc_unique_form_factors = []
        self.struc_form_avg_sq = []
        self.struc_inverse = []

        for i in range(self.num_structures):

            # Get unique elements and construc form factor stacks
            unique_elements, inverse, counts = np.unique(self.struc_elements[i], return_counts=True, return_inverse=True)

            triu_indices = torch.triu_indices(self.struc_size[i], self.struc_size[i], 1)
            unique_inverse = torch.from_numpy(inverse[triu_indices]).to(device=self.device)
            struc_unique_form_factors = torch.stack([self.form_factor_func(self.FORM_FACTOR_COEF[el]) for el in unique_elements])

            self.triu_indices.append(triu_indices)
            self.unique_inverse.append(unique_inverse)
            self.struc_unique_form_factors.append(struc_unique_form_factors)

            # Calculate average squared form factor and self scattering inverse indices
            counts = torch.from_numpy(counts).to(device=self.device)
            compositional_fractions = counts / torch.sum(counts)
            struc_form_avg_sq = torch.sum(compositional_fractions.reshape(-1,1) * struc_unique_form_factors, dim=0)**2
            struc_inverse = torch.from_numpy(np.array([inverse[i] for i in range(self.struc_size[i])])).to(device=self.device)

            self.struc_form_avg_sq.append(struc_form_avg_sq)
            self.struc_inverse.append(struc_inverse)

    def iq(
        self,
        structure_path: Union[str, Atoms, List[Atoms]],
        radii: Union[List[float], float, None] = None,
        keep_on_device: bool = False,
        _total_scattering: bool = False,
    ) -> Union[Tuple[np.float32, Union[List[np.float32], np.float32]], Tuple[torch.FloatTensor, Union[List[torch.FloatTensor], torch.FloatTensor]]]:

        """
        Calculate the scattering intensity I(Q) for the given atomic structure.

        Parameters:
            structure_path (Union[str, Atoms, List[Atoms]]): Path to the atomic structure file in XYZ/CIF format or stored ASE Atoms objects.
            radii (Union[List[float], float, None]): List/float of radii/radius of particle(s) to generate with parsed CIF
            keep_on_device (bool): Flag to keep the results on the class device. Default is False.
            _total_scattering (bool): Flag to return the scattering intensity I(Q) without the self-scattering contribution. Default is False.

        Returns:
            Tuple of torch tensors containing Q-values and scattering intensity I(Q) if keep_on_device is True, otherwise, numpy arrays on CPU.
        """
        
        # Raises errors if wrong path or parameters
        if not os.path.exists(structure_path):
            raise FileNotFoundError(f"{structure_path} not found.")
        if self.qmin < 0:
            raise ValueError("qmin must be non-negative.")
        if self.qmax < 0:
            raise ValueError("qmax must be non-negative.")
        if self.qstep < 0:
            raise ValueError("qstep must be non-negative.")
        if self.qdamp < 0:
            raise ValueError("qdamp must be non-negative.")
        if self.rmin < 0:
            raise ValueError("rmin must be non-negative.")
        if self.rmax < 0:
            raise ValueError("rmax must be non-negative.")
        if self.rstep < 0:
            raise ValueError("rstep must be non-negative.")
        if self.rthres < 0:
            raise ValueError("rthres must be non-negative.")
        if self.biso < 0:
            raise ValueError("biso must be non-negative.")
        if self.batch_size is not None and self.batch_size < 0:
            raise ValueError("batch_size must be non-negative.")
        
        # Initialise structure
        self._initialise_structures(structure_path, radii, disable_pbar = True)

        if self.profile:
            self.profiler.time('Setup structures and form factors')

        # Calculate I(Q) for all initialised structures
        iq_output = []
        for i in range(self.num_structures):

            # Calculate distances and batch
            if self.batch_size is None:
                self.batch_size = self._max_batch_size
            dists = pdist(self.struc_xyz[i]).split(self.batch_size)
            indices = self.triu_indices[i].split(self.batch_size, dim=1)
            inverse_indices = self.unique_inverse[i].split(self.batch_size, dim=1)

            if self.profile:
                self.profiler.time('Batching and Distances')

            # Calculate scattering using Debye Equation
            iq = torch.zeros((len(self.q))).to(device=self.device, dtype=torch.float32)
            for d, inv_idx, idx in zip(dists, inverse_indices, indices):
                mask = d >= self.rthres
                occ_product = self.struc_occupancy[i][idx[0]] * self.struc_occupancy[i][idx[1]]
                sinc = torch.sinc(d[mask] * self.q / torch.pi)
                ffp = self.struc_unique_form_factors[i][inv_idx[0]] * self.struc_unique_form_factors[i][inv_idx[1]]
                iq += torch.sum(occ_product.unsqueeze(-1)[mask] * ffp[mask] * sinc.permute(1,0), dim=0)

            # Apply Debye-Weller Isotropic Atomic Displacement
            if self.biso != 0.0:
                iq *= torch.exp(-self.q.squeeze(-1).pow(2) * self.biso/(8*torch.pi**2))
            
            # For total scattering
            if _total_scattering:
                if self.profile:
                    self.profiler.time('I(Q)')
                iq_output.append(iq) # TODO Times 2
                continue

            # Self-scattering contribution
            sinc = torch.ones((self.struc_size[i], len(self.q))).to(device=self.device)
            iq += torch.sum((self.struc_occupancy[i].unsqueeze(-1) * self.struc_unique_form_factors[i][self.struc_inverse[i]])**2 * sinc, dim=0) / 2
            iq *= 2

            if self.profile:
                self.profiler.time('I(Q)')

            iq_output.append(iq)
            
        if _total_scattering:
            return self.q.squeeze(-1), iq_output

        if keep_on_device:
            if self.num_structures == 1:
                return self.q.squeeze(-1), iq_output[0]
            else:
                return self.q.squeeze(-1), iq_output
        else:
            if self.num_structures == 1:
                return self.q.squeeze(-1).cpu().numpy(), iq_output[0].cpu().numpy()
            else:
                return self.q.squeeze(-1).cpu().numpy(), [iq.cpu().numpy() for iq in iq_output]

    def sq(
        self,
        structure_path: Union[str, Atoms, List[Atoms]],
        radii: Union[List[float], float, None] = None,
        keep_on_device: bool = False,
    ) -> Union[Tuple[np.float32, Union[List[np.float32], np.float32]], Tuple[torch.FloatTensor, Union[List[torch.FloatTensor], torch.FloatTensor]]]:

        """
        Calculate the structure function S(Q) for the given atomic structure.

        Parameters:
            structure_path (Union[str, Atoms, List[Atoms]]): Path to the atomic structure file in XYZ/CIF format or stored ASE Atoms objects.
            keep_on_device (bool): Flag to keep the results on the class device. Default is False.

        Returns:
            Tuple of torch tensors containing Q-values and structure function S(Q) if keep_on_device is True, otherwise, numpy arrays on CPU.
        """

        # Calculate Scattering S(Q)
        _, iq = self.iq(structure_path, radii, keep_on_device=True, _total_scattering=True)
        
        sq_output = []
        for i in range(self.num_structures):
            sq = iq[i]/self.struc_form_avg_sq[i]/self.struc_size[i]
            sq_output.append(sq)

        if keep_on_device:
            if self.num_structures == 1:
                return self.q.squeeze(-1), sq_output[0]
            else:
                return self.q.squeeze(-1), sq_output
        else:
            if self.num_structures == 1:
                return self.q.squeeze(-1).cpu().numpy(), sq_output[0].cpu().numpy()
            else:
                return self.q.squeeze(-1).cpu().numpy(), [sq.cpu().numpy() for sq in sq_output]
    
    def fq(
        self,
        structure_path: Union[str, Atoms, List[Atoms]],
        radii: Union[List[float], float, None] = None,
        keep_on_device: bool = False,
    ) -> Union[Tuple[np.float32, Union[List[np.float32], np.float32]], Tuple[torch.FloatTensor, Union[List[torch.FloatTensor], torch.FloatTensor]]]:
        """
        Calculate the reduced structure function F(Q) for the given atomic structure.

        Parameters:
            structure_path (Union[str, Atoms, List[Atoms]]): Path to the atomic structure file in XYZ/CIF format or stored ASE Atoms objects.
            keep_on_device (bool): Flag to keep the results on the class device. Default is False.

        Returns:
            Tuple of torch tensors containing Q-values and reduced structure function F(Q) if keep_on_device is True, otherwise, numpy arrays on CPU.
        """
        # Calculate Scattering S(Q)
        _, iq = self.iq(structure_path, radii, keep_on_device=True, _total_scattering=True)

        fq_output = []
        for i in range(self.num_structures):
            sq = iq[i]/self.struc_form_avg_sq[i]/self.struc_size[i]
            fq = self.q.squeeze(-1) * sq
            fq_output.append(fq)

        if keep_on_device:
            if self.num_structures == 1:
                return self.q.squeeze(-1), fq_output[0]
            else:
                return self.q.squeeze(-1), fq_output
        else:
            if self.num_structures == 1:
                return self.q.squeeze(-1).cpu().numpy(), fq_output[0].cpu().numpy()
            else:
                return self.q.squeeze(-1).cpu().numpy(), [fq.cpu().numpy() for fq in fq_output]

    def gr(
        self,
        structure_path: Union[str, Atoms, List[Atoms]],
        radii: Union[List[float], float, None] = None,
        keep_on_device: bool = False,
    ) -> Union[Tuple[np.float32, Union[List[np.float32], np.float32]], Tuple[torch.FloatTensor, Union[List[torch.FloatTensor], torch.FloatTensor]]]:

        """
        Calculate the reduced pair distribution function G(r) for the given atomic structure.

        Parameters:
            structure_path (Union[str, Atoms, List[Atoms]]): Path to the atomic structure file in XYZ/CIF format or stored ASE Atoms objects.
            keep_on_device (bool): Flag to keep the results on the class device. Default is False.

        Returns:
            Tuple of torch tensors containing Q-values and PDF G(r) if keep_on_device is True, otherwise, numpy arrays on CPU.
        """
        if self.profile:
            self.profiler.reset()

        # Calculate Scattering I(Q), S(Q), F(Q)
        _, iq = self.iq(structure_path, radii, keep_on_device=True, _total_scattering=True)

        gr_output = []
        for i in range(self.num_structures):
            sq = iq[i]/self.struc_form_avg_sq[i]/self.struc_size[i]
            if self.profile:
                self.profiler.time('S(Q)')
            fq = self.q.squeeze(-1) * sq
            if self.profile:
                self.profiler.time('F(Q)')
        
            # Calculate total scattering, G(r)
            damp = 1 if self.qdamp == 0.0 else torch.exp(-(self.r.squeeze(-1) * self.qdamp).pow(2) / 2)
            lorch_mod = 1 if self.lorch_mod == None else torch.sinc(self.q * self.lorch_mod*(torch.pi / self.qmax))
            if self.profile:
                self.profiler.time('Modifications, Qdamp/Lorch')
            gr = (2 / torch.pi) * torch.sum(fq.unsqueeze(-1) * torch.sin(self.q * self.r.permute(1,0))*self.qstep * lorch_mod, dim=0) * damp
            if self.profile:
                self.profiler.time('G(r)')

            gr_output.append(gr)
        
        if keep_on_device:
            if self.num_structures == 1:
                return self.r.squeeze(-1), gr_output[0]
            else:
                return self.r.squeeze(-1), gr_output
        else:
            if self.num_structures == 1:
                return self.r.squeeze(-1).cpu().numpy(), gr_output[0].cpu().numpy()
            else:
                return self.r.squeeze(-1).cpu().numpy(), [gr.cpu().numpy() for gr in gr_output]

    def _get_all(
        self,
        structure_path: Union[str, Atoms, List[Atoms]],
        radii: Union[List[float], float, None] = None,
        keep_on_device: bool = False,
    ) -> Union[Tuple[np.float32,np.float32,Union[List[np.float32], np.float32],Union[List[np.float32],np.float32], Union[List[np.float32],np.float32], Union[List[np.float32], np.float32]],
               Tuple[torch.FloatTensor,torch.FloatTensor,Union[List[torch.FloatTensor], torch.FloatTensor],Union[List[torch.FloatTensor],torch.FloatTensor], Union[List[torch.FloatTensor],torch.FloatTensor], Union[List[torch.FloatTensor], torch.FloatTensor]]]:

        """
        Calculate I(Q), S(Q), F(Q) and G(r) for the given atomic structure and return all.

        Parameters:
            structure_path (Union[str, Atoms, List[Atoms]]): Path to the atomic structure file in XYZ/CIF format or stored ASE Atoms objects.
            keep_on_device (bool): Flag to keep the results on the class device. Default is False.

        Returns:
            Tuple of torch tensors containing of r-values, Q-values and Union[List[float_vec], float_vec] of I(Q), S(Q), F(Q) and G(r) if keep_on_device is True, otherwise, numpy arrays on CPU.
        """

        # Initialise structure
        self._initialise_structures(structure_path, radii, disable_pbar = True)

        # Calculate I(Q) for all initialised structures
        iq_output = []
        sq_output = []
        fq_output = []
        gr_output = []
        for i in range(self.num_structures):

            # Calculate distances and batch
            if self.batch_size is None:
                self.batch_size = self._max_batch_size
            dists = pdist(self.struc_xyz[i]).split(self.batch_size)
            indices = self.triu_indices[i].split(self.batch_size, dim=1)
            inverse_indices = self.unique_inverse[i].split(self.batch_size, dim=1)

            # Calculate scattering using Debye Equation
            iq = torch.zeros((len(self.q))).to(device=self.device, dtype=torch.float32)
            for d, inv_idx, idx in zip(dists, inverse_indices, indices):
                mask = d >= self.rthres
                occ_product = self.struc_occupancy[i][idx[0]] * self.struc_occupancy[i][idx[1]]
                sinc = torch.sinc(d[mask] * self.q / torch.pi)
                ffp = self.struc_unique_form_factors[i][inv_idx[0]] * self.struc_unique_form_factors[i][inv_idx[1]]
                iq += torch.sum(occ_product.unsqueeze(-1)[mask] * ffp[mask] * sinc.permute(1,0), dim=0)

            # Apply Debye-Weller Isotropic Atomic Displacement
            if self.biso != 0.0:
                iq *= torch.exp(-self.q.squeeze(-1).pow(2) * self.biso/(8*torch.pi**2))
        
            # Calculate S(Q), F(Q) and G(r)
            sq = iq/self.struc_form_avg_sq[i]/self.struc_size[i]
            sq_output.append(sq)
            
            fq = self.q.squeeze(-1) * sq
            fq_output.append(fq)

            damp = 1 if self.qdamp == 0.0 else torch.exp(-(self.r.squeeze(-1) * self.qdamp).pow(2) / 2)
            lorch_mod = 1 if self.lorch_mod == None else torch.sinc(self.q * self.lorch_mod*(torch.pi / self.qmax))
            gr = (2 / torch.pi) * torch.sum(fq.unsqueeze(-1) * torch.sin(self.q * self.r.permute(1,0))*self.qstep * lorch_mod, dim=0) * damp
            gr_output.append(gr)
            
            # Self-scattering contribution
            sinc = torch.ones((self.struc_size[i], len(self.q))).to(device=self.device)
            iq += torch.sum((self.struc_occupancy[i].unsqueeze(-1) * self.struc_unique_form_factors[i][self.struc_inverse[i]])**2 * sinc, dim=0) / 2
            iq *= 2

            iq_output.append(iq)
            
        if keep_on_device:
            if self.num_structures == 1:
                return self.r.squeeze(-1), self.q.squeeze(-1), iq_output[0], sq_output[0], fq_output[0], gr_output[0]
            else:
                return self.r.squeeze(-1), self.q.squeeze(-1), iq_output, sq_output, fq_output, gr_output
        else:
            if self.num_structures == 1:
                return self.r.squeeze(-1).cpu().numpy(), self.q.squeeze(-1).cpu().numpy(), iq_output[0].cpu().numpy(), sq_output[0].cpu().numpy(), fq_output[0].cpu().numpy(), gr_output[0].cpu().numpy()
            else:
                return self.r.squeeze(-1).cpu().numpy(), self.q.squeeze(-1).cpu().numpy(), [iq.cpu().numpy() for iq in iq_output], [sq.cpu().numpy() for sq in sq_output], [fq.cpu().numpy() for fq in fq_output], [gr.cpu().numpy() for gr in gr_output]

    def generate_nanoparticles(
        self,
        structure_path: str,
        radii: Union[List[float], float],
        sort_atoms: bool = True,
        disable_pbar: bool = False,
        _override_device: bool = True,
    ) -> Tuple[Union[List[Atoms], Atoms], Union[List[float], float]]:

        """
        Generate nanoparticles from a given structure and list of radii.
    
        Args:
            structure_path (str): Path to the input structure file.
            radii (Union[List[float], float]): List of floats or float of radii for nanoparticles to be generated.
            sort_atoms (bool, optional): Whether to sort atoms in the nanoparticle. Defaults to True.
            _override_device (bool): Ignore object device and run in CPU
    
        Returns:
            list: List of ASE Atoms objects representing the generated nanoparticles.
            list: List of nanoparticle sizes (diameter) corresponding to each radius.
        """

        # Fix radii type
        if isinstance(radii, list):
            radii = [float(r) for r in radii]
        elif isinstance(radii, float):
            radii = [radii]
        elif isinstance(radii, int):
            radii = [float(radii)]
        else:
            raise ValueError('FAILED: Please provide valid radii for generation of nanoparticles')

        # DEV: Override device
        device = 'cpu' if _override_device else self.device

        # Read the input unit cell structure
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            unit_cell = read(structure_path)
        cell_dims = np.array(unit_cell.cell.cellpar()[:3])
        r_max = np.amax(radii)
    
        # Create a supercell to encompass the entire range of nanoparticles and center it
        supercell_matrix = np.diag((np.ceil(r_max / cell_dims)) * 2 + 2)
        cell = make_supercell(prim=unit_cell, P=supercell_matrix)
        cell.center(about=0.)
    
        # Convert positions to torch and send to device
        positions = torch.from_numpy(cell.get_positions()).to(dtype = torch.float32, device = device)

        # Find all metals and center around the nearest metal
        ligands = ['O', 'Cl', 'H'] # Placeholder
        metal_filter = torch.BoolTensor([a not in ligands for a in cell.get_chemical_symbols()]).to(device = device)
        center_dists = torch.norm(positions, dim=1)
        positions -= positions[metal_filter][torch.argmin(center_dists[metal_filter])]
        center_dists = torch.norm(positions, dim=1)
        min_metal_dist = torch.min(pdist(positions[metal_filter]))
        min_bond_dist = torch.amin(cdist(positions[metal_filter], positions[~metal_filter]))
        # Update the cell positions
        cell.positions = positions.cpu()
    
        # Initialize nanoparticle lists and progress bar
        nanoparticle_list = []
        nanoparticle_sizes = []
        pbar = tqdm(desc=f'Generating nanoparticles in range: [{np.amin(radii)},{np.amax(radii)}]', leave=False, total=len(radii), disable=disable_pbar)
    
        # Generate nanoparticles for each radius
        for r in sorted(radii, reverse=True):

            # Mask all atoms within radius
            incl_mask = (center_dists <= r) | ((center_dists <= r + min_metal_dist) & ~metal_filter)

            # Modify objects based on mask
            cell = cell[incl_mask.cpu()]
            center_dists = center_dists[incl_mask]
            metal_filter = metal_filter[incl_mask]
            positions = positions[incl_mask]
            
            # Find interdistances from all included atoms and remove 0's from diagonal
            interface_dists = cdist(positions, positions).fill_diagonal_(min_metal_dist*2)
    
            # Remove floating atoms
            interaction_mask = torch.amin(interface_dists, dim=0) < min_bond_dist*1.2
            
            # Modify objects based on mask
            cell = cell[interaction_mask.cpu()]
            center_dists = center_dists[interaction_mask]
            metal_filter = metal_filter[interaction_mask]
            positions = positions[interaction_mask]
            
            # Determine NP size
            nanoparticle_size = torch.amax(center_dists) * 2

            # Sort the atoms
            if sort_atoms:
                sorted_cell = ase_sort(cell)
                if sorted_cell.get_chemical_symbols()[0] in ligands:
                    sorted_cell = sorted_cell[::-1]
    
                # Append nanoparticle
                nanoparticle_list.append(sorted_cell)
            else:
                # Append nanoparticle
                nanoparticle_list.append(cell)

            # Append size
            nanoparticle_sizes.append(nanoparticle_size.item())

            pbar.update(1)
        pbar.close()
    
        return nanoparticle_list, nanoparticle_sizes

    def _is_notebook(
        self,
    ) -> bool:
        
        """
        Checks if the code is running within a Jupyter Notebook, Google Colab, or other interactive environment.
        
        Returns:
            bool: True if running in a Jupyter Notebook or Google Colab, False otherwise.
        """
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True # Jupyter notebook or qtconsole
            elif shell == 'google.colab._shell':
                return True # Google Colab
            elif shell == 'Shell':
                return True # Apparently also Colab?
            else:
                return False # Other cases
        except NameError:
            return False # Standard Python Interpreter

    def interact(
        self,
        _cont_updates: bool = False
    ) -> None:

        """
        Initiates an interactive visualization and data analysis tool within a Jupyter Notebook or Google Colab environment.

        Args:
            _cont_updates (bool, optional): If True, enables continuous updates for interactive widgets. Defaults to False.
        """

        # Check if interaction is valid
        if not self._is_notebook():
            print("FAILED: Interactive mode is exlusive to Jupyter Notebook or Google Colab")
            return

        # Scattering parameters
        qmin = self.qmin
        qmax = self.qmax
        qstep = self.qstep
        qdamp = self.qdamp
        rmin = self.rmin
        rmax = self.rmax
        rstep = self.rstep
        rthres = self.rthres
        biso = self.biso
        device = 'cuda' if torch.cuda.is_available() else self.device
        batch_size = self.batch_size
        lorch_mod = self.lorch_mod
        radiation_type = self.radiation_type
        profile = False

        with open(pkg_resources.resource_filename(__name__, 'display_assets/choose_hardware.png'), 'rb') as f:
            choose_hardware_img = f.read()
        with open(pkg_resources.resource_filename(__name__, 'display_assets/batch_size.png'), 'rb') as f:
            batch_size_img = f.read()
        
        """ Utility widgets """

        # Spacing widget
        spacing_10px = widgets.Text(description='', layout=Layout(visibility='hidden', height='10px'), disabled=True)
        spacing_5px = widgets.Text(description='', layout=Layout(visibility='hidden', height='5px'), disabled=True)

        """ File Selection Tab """
        
        # Load diplay display_assets
        with open(pkg_resources.resource_filename(__name__, 'display_assets/enter_path.png'), 'rb') as f:
            enter_path_img = f.read()
        with open(pkg_resources.resource_filename(__name__, 'display_assets/select_files.png'), 'rb') as f:
            select_files_img = f.read()
        with open(pkg_resources.resource_filename(__name__, 'display_assets/radius_a.png'), 'rb') as f:
            radius_a_img = f.read()
        with open(pkg_resources.resource_filename(__name__, 'display_assets/file_1.png'), 'rb') as f:
            file_1_img = f.read()
        with open(pkg_resources.resource_filename(__name__, 'display_assets/file_2.png'), 'rb') as f:
            file_2_img = f.read()

        # Layout
        file_tab_layout = Layout(
            display='flex',
            flex_flow='column',
            align_items='stretch',
            order='solid',
            width='90%',
        )

        # File selection sizes
        header_widths = [105*1.8, 130*1.8]
        header_widths = [str(i)+'px' for i in header_widths]

        # Folder selection
        folder = widgets.Text(description='', placeholder='Enter data directory', disabled=False, layout=Layout(width='650px'))
        
        # Dropdown file sections
        DEFAULT_MSGS = ['No valid files in entered directory', 'Select data file']
        select_file_1 = widgets.Dropdown(options=DEFAULT_MSGS, value=DEFAULT_MSGS[0], disabled=True, layout=Layout(width='650px'))
        select_file_2 = widgets.Dropdown(options=DEFAULT_MSGS, value=DEFAULT_MSGS[0], disabled=True, layout=Layout(width='650px'))
        
        # File 1
        select_file_desc_1 = HBox([widgets.Image(value=file_1_img, format='png', layout=Layout(object_fit='contain', object_position='20px', width='32px'))], layout=Layout(width='88px'))
        select_radius_desc_1 = HBox([widgets.Image(value=radius_a_img, format='png', layout=Layout(object_fit='contain', object_position='20px', width='60px', visibility='hidden'))], layout=Layout(width='88px'))
        select_radius_1 = widgets.FloatText(min = 0, max = 50, step=0.01, value=5, disabled = False, layout = Layout(width='50px', visibility='hidden'))
        cif_text_1 = widgets.Text(
            value='Given radius, generate spherical nanoparticles (NP) from crystallographic information files (CIFs)', 
            disabled=True,
            layout=Layout(width='595px', visibility='hidden')
        )

        # File 2
        select_file_desc_2 = HBox([widgets.Image(value=file_2_img, format='png', layout=Layout(object_fit='contain', object_position='20px', width='32px'))], layout=Layout(width='88px'))
        select_radius_desc_2 = HBox([widgets.Image(value=radius_a_img, format='png', layout=Layout(object_fit='contain', object_position='20px', width='60px', visibility='hidden'))], layout=Layout(width='88px'))
        select_radius_2 = widgets.FloatText(min = 0, max = 50, step=0.01, value=5, disabled = False, layout = Layout(width='50px', visibility='hidden'))
        cif_text_2 = widgets.Text(
            value='Given radius, generate spherical nanoparticles (NP) from crystallographic information files (CIFs)', 
            disabled=True,
            layout=Layout(width='595px', visibility='hidden')
        )

        # File selection Tab
        file_tab = VBox([
            # Enter path
            widgets.Image(value=enter_path_img, format='png', layout=Layout(object_fit='contain', width=header_widths[0])),
            folder,

            spacing_10px,
            
            # Select file(s)
            widgets.Image(value=select_files_img, format='png', layout=Layout(object_fit='contain', width=header_widths[1])),

            # Select file 1
            HBox([select_file_desc_1, select_file_1]),

            # if CIF, radius options
            HBox([select_radius_desc_1, select_radius_1, cif_text_1]),

            spacing_10px,

            # Select file 2
            HBox([select_file_desc_2, select_file_2]),

            # If CIF, radius options
            HBox([select_radius_desc_2, select_radius_2, cif_text_2]),
        ], layout = file_tab_layout)
        
        """ Scattering Options Tab """

        # Load display_assets
        with open(pkg_resources.resource_filename(__name__, 'display_assets/qslider.png'), 'rb') as f:
            qslider_img = f.read()
        with open(pkg_resources.resource_filename(__name__, 'display_assets/rslider.png'), 'rb') as f:
            rslider_img = f.read()
        with open(pkg_resources.resource_filename(__name__, 'display_assets/qdamp.png'), 'rb') as f:
            qdamp_img = f.read()
        with open(pkg_resources.resource_filename(__name__, 'display_assets/global_biso.png'), 'rb') as f:
            global_biso_img = f.read()
        with open(pkg_resources.resource_filename(__name__, 'display_assets/a.png'), 'rb') as f:
            a_img = f.read()
        with open(pkg_resources.resource_filename(__name__, 'display_assets/a_inv.png'), 'rb') as f:
            a_inv_img = f.read()
        with open(pkg_resources.resource_filename(__name__, 'display_assets/a_sq.png'), 'rb') as f:
            a_sq_img = f.read()
        with open(pkg_resources.resource_filename(__name__, 'display_assets/qstep.png'), 'rb') as f:
            qstep_img = f.read()
        with open(pkg_resources.resource_filename(__name__, 'display_assets/rstep.png'), 'rb') as f:
            rstep_img = f.read()
        with open(pkg_resources.resource_filename(__name__, 'display_assets/rthres.png'), 'rb') as f:
            rthres_img = f.read()
        with open(pkg_resources.resource_filename(__name__, 'display_assets/radiation_type.png'), 'rb') as f:
            radiation_type_img = f.read()
        with open(pkg_resources.resource_filename(__name__, 'display_assets/scattering_parameters.png'), 'rb') as f:
            scattering_parameters_img = f.read()
        with open(pkg_resources.resource_filename(__name__, 'display_assets/presets.png'), 'rb') as f:
            presets_img = f.read()

        # Radiation 
        radtype_button = widgets.ToggleButtons(
            options=['xray', 'neutron'],
            value=radiation_type,
            layout = Layout(width='800px'),
            button_style='primary'
        )

        # Q value slider
        qslider = widgets.FloatRangeSlider(
            value=[qmin, qmax],
            min=0.0, max=50.0, step=0.01,
            orientation='horizontal',
            readout=True,
            style={'font_weight':'bold', 'slider_color': 'white', 'description_width': '100px'},
            layout = widgets.Layout(width='80%'),
        )

        # r value slider
        rslider = widgets.FloatRangeSlider(
            value=[rmin, rmax],
            min=0, max=100.0, step=rstep,
            orientation='horizontal',
            readout=True,
            style={'font_weight':'bold', 'slider_color': 'white', 'description_width': '100px'},
            layout = widgets.Layout(width='80%'),
        )

        # Qdamp box
        qdamp_box = widgets.FloatText(
            min=0.00,max=0.10, step=0.01,
            value=qdamp, 
            layout = widgets.Layout(width='50px'),
        )

        # B iso box
        biso_box = widgets.FloatText(
            min=0.00, max=1.00, step=0.01,
            value=biso,
            layout = widgets.Layout(width='50px'),
        )

        # Qstep box
        qstep_box = widgets.FloatText(
            min = 0.001, max = 1, step=0.001,
            value=qstep,
            layout=Layout(width='50px'),
        )

        # rstep box
        rstep_box = widgets.FloatText(
            min = 0.001, max = 1, step=0.001,
            value=rstep,
            layout=Layout(width='50px'),
        )

        # rthreshold box
        rthres_box = widgets.FloatText(
            min = 0.001, max = 1, step=0.001,
            value=rthres,
            layout=Layout(width='50px'),
        )

        # Lorch modification button
        lorch_mod_button = widgets.ToggleButton(
            value=lorch_mod,
            description='Lorch modification (OFF)',
            layout=Layout(width='250px'),
            button_style='primary',
        )
        
        # SAS preset button
        sas_preset_button = widgets.Button(
            description = 'Small Angle Scattering preset',
            layout=Layout(width='300px'),
            button_style='primary',
        )

        # Powder diffraction preset
        pd_preset_button = widgets.Button(
            description = 'Powder Diffraction preset',
            layout=Layout(width='300px'),
            button_style='primary',
        )
        
        # Total scattering preset
        ts_preset_button = widgets.Button(
            description = 'Total Scattering preset',
            layout=Layout(width='300px'),
            button_style='primary',
        )
        
        # Total scattering preset
        reset_button = widgets.Button(
            description = 'Reset scattering options',
            layout=Layout(width='300px'),
            button_style='danger',
        )

        # Scattering Tab sizes
        header_widths = [90*1.15, 135*1.15, 110*1.15]
        header_widths = [str(i)+'px' for i in header_widths]
        a_inv_width = '27px'
        a_width = '12px'
        a_sq_width = '19px'
        
        # Scattering tab
        scattering_tab = VBox([
            # Radiation button
            widgets.Image(value=radiation_type_img, format='png', layout=Layout(object_fit='contain', width=header_widths[0])),
            radtype_button,

            spacing_10px,

            # Scattering parameters
            widgets.Image(value=scattering_parameters_img, format='png', layout=Layout(object_fit='contain', width=header_widths[1])),

            # Q slider
            HBox([
                # Q slider img
                HBox([widgets.Image(value=qslider_img, format='png', layout=Layout(object_fit='contain', width='120px'))], layout=Layout(width='150px')),
                # Q slider
                qslider,
            ]),

            spacing_5px,

            # r slider
            HBox([
                # r slider img
                HBox([widgets.Image(value=rslider_img, format='png', layout=Layout(object_fit='contain', width='110px'))], layout=Layout(width='150px')),
                # r slider
                rslider, 
            ]),

            spacing_5px,

            # Other
            HBox([
                # Qstep img
                HBox([widgets.Image(value=qstep_img, format='png', layout=Layout(object_fit='contain', object_position='', width='65px'))], layout=Layout(width='75px')),
                # Qstep box
                qstep_box, 
                
                # r step img
                widgets.Text(description='', layout=Layout(visibility='hidden', width='60px'), disabled=True),
                HBox([widgets.Image(value=rstep_img, format='png', layout=Layout(object_fit='contain', object_position='', width='55px'))], layout=Layout(width='65px')),
                # r step box
                rstep_box,

                # Q damp img
                widgets.Text(description='', layout=Layout(visibility='hidden', width='60px'), disabled=True),
                HBox([widgets.Image(value=qdamp_img, format='png', layout=Layout(object_fit='contain', object_position='', width='75px'))], layout=Layout(width='85px')),
                # Q damp box
                qdamp_box,

                # r thres img
                widgets.Text(description='', layout=Layout(visibility='hidden', width='60px'), disabled=True),
                HBox([widgets.Image(value=rthres_img, format='png', layout=Layout(object_fit='contain', object_position='', width='55px'))], layout=Layout(width='65px')),
                # r thres
                rthres_box,
                
                # Global B iso img
                widgets.Text(description='', layout=Layout(visibility='hidden', width='60px'), disabled=True),
                HBox([widgets.Image(value=global_biso_img, format='png', layout=Layout(object_fit='contain', object_position='', width='95px'))], layout=Layout(width='105px')),
                # Global B iso box
                biso_box,
            ]),
            
            spacing_5px,

            # Global B iso
            HBox([
                # Lorch mod button
                lorch_mod_button, 
                # Unit
                HBox([widgets.Image(value=a_img, format='png', layout=Layout(object_fit='contain', width=a_width, visibility='hidden'))], layout=Layout(width='50px')),
            ]),

            spacing_10px,

            # Presets
            widgets.Image(value=presets_img, format='png', layout=Layout(object_fit='contain', width=header_widths[2])),
            HBox([sas_preset_button, pd_preset_button, ts_preset_button, reset_button]),
        ])

        """ Plotting Options """

        # Load display display_assets
        with open(pkg_resources.resource_filename(__name__, 'display_assets/iq_scaling.png'), 'rb') as f:
            iq_scaling_img = f.read()
        with open(pkg_resources.resource_filename(__name__, 'display_assets/show_hide.png'), 'rb') as f:
            show_hide_img = f.read()
        with open(pkg_resources.resource_filename(__name__, 'display_assets/max_norm.png'), 'rb') as f:
            max_norm_img = f.read()
        with open(pkg_resources.resource_filename(__name__, 'display_assets/iq.png'), 'rb') as f:
            iq_img = f.read()
        with open(pkg_resources.resource_filename(__name__, 'display_assets/sq.png'), 'rb') as f:
            sq_img = f.read()
        with open(pkg_resources.resource_filename(__name__, 'display_assets/fq.png'), 'rb') as f:
            fq_img = f.read()
        with open(pkg_resources.resource_filename(__name__, 'display_assets/gr.png'), 'rb') as f:
            gr_img = f.read()
        
        # Y-axis I(Q) scale button
        scale_type_button = widgets.ToggleButtons( options=['linear', 'logarithmic'], value='linear', button_style='primary', layout=Layout(width='600'))

        # Show/Hide buttons
        show_iq_button = widgets.Checkbox(value = True)
        show_sq_button = widgets.Checkbox(value = True)
        show_fq_button = widgets.Checkbox(value = True)
        show_gr_button = widgets.Checkbox(value = True)

        # Max normalize buttons
        normalize_iq = widgets.Checkbox(value = False)
        normalize_sq = widgets.Checkbox(value = False)
        normalize_fq = widgets.Checkbox(value = False)
        normalize_gr = widgets.Checkbox(value = False)
        
        # Plotting tab sizes
        function_offset = '-90px 3px'
        function_size = 35
        header_scale = 0.95
        header_widths = [130, 120, 147]
        header_widths = [str(i * header_scale)+'px' for i in header_widths]

        # Plotting tab 
        plotting_tab = VBox([
            # I(Q) scaling img
            widgets.Image(value=iq_scaling_img, format='png', layout=Layout(object_fit='contain', width=header_widths[0])),
            scale_type_button,
            
            spacing_10px,
            
            # Show / Hide img
            widgets.Image(value=show_hide_img, format='png', layout=Layout(object_fit='contain', width=header_widths[1])),
            
            # Options
            HBox([
                HBox([show_iq_button, widgets.Image(value=iq_img, format='png', width=function_size, layout=Layout(object_fit='contain', object_position=function_offset))]),
                HBox([show_sq_button, widgets.Image(value=sq_img, format='png', width=function_size, layout=Layout(object_fit='contain', object_position=function_offset))]),
                HBox([show_fq_button, widgets.Image(value=fq_img, format='png', width=function_size, layout=Layout(object_fit='contain', object_position=function_offset))]),
                HBox([show_gr_button, widgets.Image(value=gr_img, format='png', width=function_size, layout=Layout(object_fit='contain', object_position=function_offset))]),
            ]),

            spacing_10px,

            # Max normalization img
            widgets.Image(value=max_norm_img, format='png', layout=Layout(object_fit='contain', width=header_widths[2])),

            # Options
            HBox([
                HBox([normalize_iq, widgets.Image(value=iq_img, format='png', width=function_size, layout=Layout(object_fit='contain', object_position=function_offset))]),
                HBox([normalize_sq, widgets.Image(value=sq_img, format='png', width=function_size, layout=Layout(object_fit='contain', object_position=function_offset))]),
                HBox([normalize_fq, widgets.Image(value=fq_img, format='png', width=function_size, layout=Layout(object_fit='contain', object_position=function_offset))]),
                HBox([normalize_gr, widgets.Image(value=gr_img, format='png', width=function_size, layout=Layout(object_fit='contain', object_position=function_offset))]),
            ]),
        ])


        """ Hardware Options Tab """

        # Hardware button
        hardware_button = widgets.ToggleButtons(options=['cpu', 'cuda'], value=device, button_style='primary')

        # Distance batch-size box
        batch_size_box = widgets.IntText(min = 100, max = 10000, value=batch_size)
        
        # Hardware tab sizes
        header_scale = 1
        header_widths = [120, 175]
        header_widths = [str(i * header_scale)+'px' for i in header_widths]

        # Hardware tab
        hardware_tab = VBox([
            # Choose hardware img
            widgets.Image(value=choose_hardware_img, format='png', layout=Layout(object_fit='contain', width=header_widths[0])),

            # Hardware box
            hardware_button,

            spacing_10px,

            # Distance batch_size img
            widgets.Image(value=batch_size_img, format='png', layout=Layout(object_fit='contain', width=header_widths[1])),

            # Distance batch size box
            batch_size_box,
        ])


        """ Display tabs """
    
        # Display Tabs
        tabs = widgets.Tab([
            file_tab,
            scattering_tab,
            plotting_tab,
            hardware_tab,
        ])
    
        # Set tab titles
        tabs.set_title(0, 'File Selection')
        tabs.set_title(1, 'Scattering Options')
        tabs.set_title(2, 'Plotting Options')
        tabs.set_title(3, 'Hardware Options')
        
        # Plot button and Download buttons
        plot_button = widgets.Button(description='Plot data', layout=Layout(width='50%', height='90%'), button_style='primary', icon='fa-pencil-square-o')
        download_button = widgets.Button(description="Download- and plot data", layout=Layout(width='50%', height='90%'), button_style='success', icon='fa-download')
        
        def display_tabs():
            display(VBox([tabs, HBox([plot_button, download_button], layout=Layout(width='100%', height='50px'))]))


        """ Download utility """
        
        # Download options
        def create_download_link(select_file, select_radius, filename_prefix, data, header=None):
        
            # Collect Metadata
            metadata ={
                'qmin': qslider.value[0],
                'qmax': qslider.value[1],
                'qdamp': qdamp_box.value,
                'qstep': qstep_box.value,
                'rmin': rslider.value[0], 
                'rmax': rslider.value[1],
                'rstep': rstep_box.value, 
                'rthres': rthres_box.value,
                'biso': biso_box.value,
                'device': hardware_button.value,
                'batch_size': batch_size_box.value, 
                'lorch_mod': lorch_mod_button.value,
                'radiation_type': radtype_button.value
            }
    
            # Join content
            output = ''
            content = "\n".join([",".join(map(str, np.around(row,len(str(qstep_box.value))))) for row in data])
            for k,v in metadata.items():
                output += f'{k}:{v}\n'
            output += '\n'
            if header:
                output += header + '\n'
            output += content
        
            # Encode as base64
            b64 = base64.b64encode(output.encode()).decode()
        
            # Add Time
            t = datetime.now()
            year = f'{t.year}'[-2:]
            month = f'{t.month}'.zfill(2)
            day = f'{t.day}'.zfill(2)
            hours = f'{t.hour}'.zfill(2)
            minutes = f'{t.minute}'.zfill(2)
            seconds = f'{t.second}'.zfill(2)
            
            # Make filename
            if select_radius is not None:
                filename = filename_prefix + '_' + select_file.value.split('/')[-1].split('.')[0] + '_radius' + str(select_radius.value) + '_' + month + day + year + '_' + hours + minutes + seconds + '.csv'
            else:
                filename = filename_prefix + '_' + select_file.value.split('/')[-1].split('.')[0] + '_' + month + day + year + '_' + hours + minutes + seconds + '.csv'
        
            # Make href and return
            href = filename_prefix + ':\t' + f'<a href="data:text/csv;base64,{b64}" download="{filename}">{filename}</a>'
            return href
        
        def create_structure_download_link(select_file, select_radius, filename_prefix, ase_atoms):
            
            # Get atomic properties
            positions = ase_atoms.get_positions()
            elements = ase_atoms.get_chemical_symbols()
            num_atoms = len(ase_atoms)
        
            # Make header
            header = str(num_atoms) + "\n\n"
        
            # Join content 
            content = header + "\n".join([el + '\t' + "\t".join(map(str,np.around(row, 3))) for row, el in zip(positions, elements)])
            
            # Encode as base64
            b64 = base64.b64encode(content.encode()).decode()
            
            # Add Time
            t = datetime.now()
            year = f'{t.year}'[-2:]
            month = f'{t.month}'.zfill(2)
            day = f'{t.day}'.zfill(2)
            hours = f'{t.hour}'.zfill(2)
            minutes = f'{t.minute}'.zfill(2)
            seconds = f'{t.second}'.zfill(2)
        
            # Make ilename
            filename = filename_prefix + '_' + select_file.value.split('/')[-1].split('.')[0] + '_radius' + str(select_radius.value) + '_' + month + day + year + '_' + hours + minutes + seconds + '.xyz'
        
            # Make href and return
            href = filename_prefix + ':\t' + f'<a href="data:text/xyz;base64,{b64}" download="{filename}">{filename}</a>'
            return href
        

        @download_button.on_click
        def on_download_button_click(button):
            global debye_outputs
            # Try to compile all the data and create html link to download files
            try:
                # clear and display
                clear_output(wait=True)
                display_tabs()

                debye_outputs = []
                for select_file, select_radius in zip([select_file_1, select_file_2], [select_radius_1, select_radius_2]):
                    try:
                        path_ext = select_file.value.split('.')[-1]
                    except Exception as e:
                        return
                    if (select_file.value is not None) and (select_file.value not in DEFAULT_MSGS) and (path_ext in ['xyz', 'cif']):
                        try:
                            debye_calc = DebyeCalculator(
                                device=hardware_button.value, 
                                batch_size=batch_size_box.value,
                                radiation_type=radtype_button.value,
                                qmin=qslider.value[0], 
                                qmax=qslider.value[1], 
                                qstep=qstep_box.value, 
                                qdamp=qdamp_box.value,
                                rmin=rslider.value[0],
                                rmax=rslider.value[1], 
                                rstep=rstep_box.value, 
                                rthres=rthres_box.value, 
                                biso=biso_box.value,
                                lorch_mod=lorch_mod_button.value
                            )
                            if (select_radius.layout.visibility != 'hidden') and (select_radius.value > 8):
                                print(f'Generating nanoparticle of radius {select_radius.value} using {select_file.value.split("/")[-1]} ...')
                            debye_outputs.append(debye_calc._get_all(select_file.value, select_radius.value))
                        except Exception as e:
                            print(f'FAILED: Could not load data file: {path}', end='\r')
                
                if len(debye_outputs) < 1:
                    print('FAILED: Please select data file(s)', end="\r")
                    return

                i = 0
                for select_file, select_radius in zip([select_file_1, select_file_2], [select_radius_1, select_radius_2]):
                    
                    # Display download links
                    if select_file.value not in DEFAULT_MSGS:

                        # Print
                        print('Download links for '  + select_file.value.split('/')[-1] + ':')
                            
                        r, q, iq, sq, fq, gr = debye_outputs[i]

                        iq_data = np.column_stack([q, iq])
                        sq_data = np.column_stack([q, sq])
                        fq_data = np.column_stack([q, fq])
                        gr_data = np.column_stack([r, gr])

                        if select_radius.layout.visibility == 'visible':
                            ase_atoms, _ = DebyeCalculator().generate_nanoparticles(select_file.value, select_radius.value)
                            display(HTML(create_structure_download_link(select_file, select_radius, f'structure', ase_atoms[0])))
                            display(HTML(create_download_link(select_file, select_radius, 'iq', iq_data, "q,I(Q)")))
                            display(HTML(create_download_link(select_file, select_radius, 'sq', sq_data, "q,S(Q)")))
                            display(HTML(create_download_link(select_file, select_radius, 'fq', fq_data, "q,F(Q)")))
                            display(HTML(create_download_link(select_file, select_radius, 'gr', gr_data, "r,G(r)")))
                        else:
                            display(HTML(create_download_link(select_file, None, 'iq', iq_data, "q,I(Q)")))
                            display(HTML(create_download_link(select_file, None, 'sq', sq_data, "q,S(Q)")))
                            display(HTML(create_download_link(select_file, None, 'fq', fq_data, "q,F(Q)")))
                            display(HTML(create_download_link(select_file, None, 'gr', gr_data, "r,G(r)")))
                        print('\n')
                        i += 1
                    
                update_figure(debye_outputs)
        
            except Exception as e:
                raise(e)
                print('FAILED: Please select data file(s)', end="\r")

        """ Observer utility """
                      
        # Define a function to update the scattering patterns based on the selected parameters
        def update_options(change):
            folder = change.new
            paths = sorted(glob(os.path.join(folder, '*.xyz')) + glob(os.path.join(folder, '*.cif')))
            if len(paths):
                for select_file in [select_file_1, select_file_2]:
                    select_file.options = ['Select data file'] + paths
                    select_file.value = 'Select data file'
                    select_file.disabled = False
            else:
                for select_file in [select_file_1, select_file_2]:
                    select_file.options = [DEFAULT_MSGS[0]]
                    select_file.value = DEFAULT_MSGS[0]
                    select_file.disabled = True
        
        
        def update_options_radius_1(change):
            #select_radius = change.new
            selected_ext = select_file_1.value.split('.')[-1]
            if selected_ext == 'xyz':
                select_radius_desc_1.children[0].layout.visibility = 'hidden'
                select_radius_1.layout.visibility = 'hidden'
                cif_text_1.layout.visibility = 'hidden'
            elif selected_ext == 'cif':
                select_radius_desc_1.children[0].layout.visibility = 'visible'
                select_radius_1.layout.visibility = 'visible'
                cif_text_1.layout.visibility = 'visible'
            else:
                select_radius_desc_1.children[0].layout.visibility = 'hidden'
                select_radius_1.layout.visibility = 'hidden'
                cif_text_1.layout.visibility = 'hidden'
        
        def update_options_radius_2(change):
            #select_radius = change.new
            selected_ext = select_file_2.value.split('.')[-1]
            if selected_ext == 'xyz':
                select_radius_desc_2.children[0].layout.visibility = 'hidden'
                select_radius_2.layout.visibility = 'hidden'
                cif_text_2.layout.visibility = 'hidden'
            elif selected_ext == 'cif':
                select_radius_desc_2.children[0].layout.visibility = 'visible'
                select_radius_2.layout.visibility = 'visible'
                cif_text_2.layout.visibility = 'visible'
            else:
                select_radius_desc_2.children[0].layout.visibility = 'hidden'
                select_radius_2.layout.visibility = 'hidden'
                cif_text_2.layout.visibility = 'hidden'
        
        # Link the update functions to the dropdown widget's value change event
        folder.observe(update_options, names='value')
        select_file_1.observe(update_options_radius_1, names='value')
        select_file_2.observe(update_options_radius_2, names='value')
        

        """ Plotting utility """

        def togglelorch(change):
            if change['new']:
                lorch_mod_button.description = 'Lorch modification (ON)'
            else:
                lorch_mod_button.description = 'Lorch modification (OFF)'

        lorch_mod_button.observe(togglelorch, 'value')

        @sas_preset_button.on_click
        def sas_preset(b=None):
            # Change scale type
            scale_type_button.value = 'logarithmic'

            # Hide all but IQ
            show_iq_button.value = True
            show_fq_button.value = False
            show_sq_button.value = False
            show_gr_button.value = False

            # Set qmin and qmax
            qslider.value = [0.0, 3.0]
            qstep_box.value = 0.01
        
        @pd_preset_button.on_click
        def pd_preset(b=None):
            # Change scale type
            scale_type_button.value = 'linear'

            # Hide all but IQ
            show_iq_button.value = True
            show_fq_button.value = False
            show_sq_button.value = False
            show_gr_button.value = False

            # Set qmin and qmax
            qslider.value = [1.0, 8.0]
            qstep_box.value = 0.1
        
        @ts_preset_button.on_click
        def ts_preset(b=None):
            # Change scale type
            scale_type_button.value = 'linear'

            # Hide all but IQ
            show_iq_button.value = True
            show_fq_button.value = True
            show_sq_button.value = False
            show_gr_button.value = True

            # Set qmin and qmax
            qslider.value = [1.0, 30.0]
            qstep_box.value = 0.1
        
        @reset_button.on_click
        def reset(b=None):
            # Change scale type
            scale_type_button.value = 'linear'

            # Hide all but IQ
            show_iq_button.value = True
            show_fq_button.value = True
            show_sq_button.value = True
            show_gr_button.value = True

            # Set qmin and qmax
            qslider.value = [1.0, 30.0]
            rslider.value = [0.0, 20.0]
            qstep_box.value = 0.1
            rstep_box.value = 0.01
            biso_box.value = 0.3
            qdamp_box.value = 0.04
            rthres_box.value = 0.0

        def update_figure(debye_outputs, _unity_sq=True):

            xseries, yseries = [], []
            xlabels, ylabels = [], []
            scales, titles = [], []
            axis_ids = []

            normalize_iq_text = ' [counts]' if not normalize_iq.value else ' [normalized]'
            normalize_sq_text = '' if not normalize_iq.value else ' [normalized]'
            normalize_fq_text = '' if not normalize_iq.value else ' [normalized]'
            normalize_gr_text = '' if not normalize_iq.value else ' [normalized]'

            for do in debye_outputs:
                if show_iq_button.value:
                    axis_ids.append(0)
                    xseries.append(do[1]) # q
                    iq_ = do[2] if not normalize_iq.value else do[2]/max(do[2]) 
                    yseries.append(iq_) # iq
                    xlabels.append('$Q$ [$\AA^{-1}$]')
                    ylabels.append('$I(Q)$' + normalize_iq_text)
                    if scale_type_button.value == 'logarithmic':
                        scales.append('log')
                    else:
                        scales.append('linear')
                    scale = scale_type_button.value
                    titles.append('Scattering Intensity, I(Q)')
                if show_sq_button.value:
                    axis_ids.append(1)
                    xseries.append(do[1]) # q
                    sq_ = do[3] if not normalize_sq.value else do[3]/max(do[3]) 
                    yseries.append(sq_) # sq
                    xlabels.append('$Q$ [$\AA^{-1}$]')
                    ylabels.append('$S(Q)$' + normalize_sq_text)
                    scales.append('linear')
                    titles.append('Structure Function, S(Q)')
                if show_fq_button.value:
                    axis_ids.append(2)
                    xseries.append(do[1]) # q
                    fq_ = do[4] if not normalize_fq.value else do[4]/max(do[4]) 
                    yseries.append(fq_) # fq
                    xlabels.append('$Q$ [$\AA^{-1}$]')
                    ylabels.append('$F(Q)$'+ normalize_fq_text)
                    scales.append('linear')
                    titles.append('Reduced Structure Function, F(Q)')
                if show_gr_button.value:
                    axis_ids.append(3)
                    xseries.append(do[0]) # r
                    gr_ = do[5] if not normalize_gr.value else do[5]/max(do[5]) 
                    yseries.append(gr_) # gr
                    xlabels.append('$r$ [$\AA$]')
                    ylabels.append('$G(r)$' + normalize_gr_text)
                    scales.append('linear')
                    titles.append('Reduced Pair Distribution Function, G(r)')

            sup_title = []
            labels = []
            if select_file_1.value not in ['Select data file', 'No valid files in entered directory']:

                sup_title.append(select_file_1.value.split('/')[-1])

                if select_radius_1.layout.visibility == 'hidden':
                    labels.append(sup_title[-1])
                else:
                    labels.append(sup_title[-1] + ', rad.: ' + str(select_radius_1.value) + ' Å')

            if select_file_2.value not in ['Select data file', 'No valid files in entered directory']:

                sup_title.append(select_file_2.value.split('/')[-1])

                if select_radius_2.layout.visibility == 'hidden':
                    labels.append(sup_title[-1])
                else:
                    labels.append(sup_title[-1] + ', rad.: ' + str(select_radius_2.value) + ' Å')

            if len(labels) == 0:
                return

            num_plots = int(show_iq_button.value) + int(show_sq_button.value) + int(show_fq_button.value) + int(show_gr_button.value) 
            if num_plots == 4:
                fig, axs = plt.subplots(2,2,figsize=(12, 8), dpi=75)
                axs = axs.ravel()
            elif num_plots == 3:
                fig, axs = plt.subplots(3,1,figsize=(12,8), dpi=75)
            elif num_plots == 2:
                fig, axs = plt.subplots(2,1,figsize=(12,8), dpi=75)
            elif num_plots == 1:
                fig, axs = plt.subplots(figsize=(12,6), dpi=75)
                axs = [axs]
            else:
                return

            for i,(x,y,xl,yl,s,t,l) in enumerate(zip(xseries, yseries, xlabels, ylabels, scales, titles, np.repeat(labels, num_plots))):
                
                ii = i % num_plots
                axs[ii].set_xscale(s)
                axs[ii].set_yscale(s)
                axs[ii].plot(x,y, label=l)
                axs[ii].set(xlabel=xl, ylabel=yl, title=t)
                axs[ii].relim()
                axs[ii].autoscale_view()
                axs[ii].grid(alpha=0.2, which='both')
                axs[ii].legend()

            if len(sup_title) == 1:
                title = f"Showing files: {sup_title[0]}"
            else:
                title = f"Showing files: {sup_title[0]} and {sup_title[1]}"
            fig.suptitle(title)
            fig.tight_layout()

        @plot_button.on_click
        def update_parameters(b=None):
            global debye_outputs

            debye_outputs = []
            for select_file, select_radius in zip([select_file_1, select_file_2], [select_radius_1, select_radius_2]):
                try:
                    path_ext = select_file.value.split('.')[-1]
                except Exception as e:
                    return
                if (select_file.value is not None) and (select_file.value not in [DEFAULT_MSGS]) and (path_ext in ['xyz', 'cif']):
                    try:
                        # TODO if not changed, dont make new object
                        debye_calc = DebyeCalculator(
                            device=hardware_button.value, 
                            batch_size=batch_size_box.value,
                            radiation_type=radtype_button.value,
                            qmin=qslider.value[0], 
                            qmax=qslider.value[1], 
                            qstep=qstep_box.value, 
                            qdamp=qdamp_box.value,
                            rmin=rslider.value[0],
                            rmax=rslider.value[1], 
                            rstep=rstep_box.value, 
                            rthres=rthres_box.value, 
                            biso=biso_box.value,
                            lorch_mod=lorch_mod_button.value
                        )
                        if not select_radius.disabled and select_radius.value > 8:
                            print(f'Generating nanoparticle of radius {select_radius.value} using {select_file.value.split("/")[-1]} ...')
                        debye_outputs.append(debye_calc._get_all(select_file.value, select_radius.value))
                    except Exception as e:
                        print(f'FAILED: Could not load data file: {path}', end='\r')

            # Clear and display
            clear_output(wait=True)
            display_tabs()

            if len(debye_outputs) < 1:
                print('FAILED: Please select data file(s)', end="\r")
                return

            update_figure(debye_outputs)

        # Display tabs when function is called
        display_tabs()
