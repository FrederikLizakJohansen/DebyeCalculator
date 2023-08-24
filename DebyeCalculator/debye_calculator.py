import os
import sys
import base64
import yaml
import pkg_resources
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

from profiling import Profiler

import ipywidgets as widgets
from IPython.display import display, HTML
from ipywidgets import interact, interact_manual
from tqdm.auto import tqdm

class DebyeCalculator:
    """
    Calculate the scattering intensity I(q) through the Debye scattering equation, the Total Scattering Structure Function S(q), 
    the Reduced Total Scattering Function F(q), and the Reduced Atomic Pair Distribution Function G(r) for a given atomic structure.


    Parameters:
        qmin (float): Minimum q-value for the scattering calculation. Default is 0.0.
        qmax (float): Maximum q-value for the scattering calculation. Default is 30.0.
        qstep (float): Step size for the q-values in the scattering calculation. Default is 0.1.
        qdamp (float): Damping parameter for Debye-Waller isotropic atomic displacement. Default is 0.0.
        rmin (float): Minimum r-value for the pair distribution function (PDF) calculation. Default is 0.0.
        rmax (float): Maximum r-value for the PDF calculation. Default is 20.0.
        rstep (float): Step size for the r-values in the PDF calculation. Default is 0.01.
        rthres (float): Threshold value for exclusion of distances below this value in the scattering calculation. Default is 0.0.
        biso (float): Debye-Waller isotropic atomic displacement parameter. Default is 0.0.
        device (str): Device to use for computation (e.g., 'cuda' for GPU or 'cpu' for CPU). Default is 'cuda'.
        batch_size (int or None): Batch size for computation. If None, the batch size will be automatically set. Default is None.
        lorch_mod (bool): Flag to enable Lorch modification. Default is False.
        radiation_type (str): Type of radiation for form factor calculations ('xray' or 'neutron'). Default is 'xray'.
        profile (bool): Activate profiler. Default is False.
    """
    def __init__(
        self,
        qmin: float = 1.0,
        qmax: float = 30.0,
        qstep: float = 0.01,
        qdamp: float = 0.04,
        rmin: float = 0.0,
        rmax: float = 20.0,
        rstep: float = 0.01,
        rthres: float = 0.0,
        biso: float = 0.3,
        device: str = 'cpu',
        batch_size: Union[int, None] = 1000,
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

    def _initialise_structure(
        self,
        structure_path: str,
    ) -> None:
        """
        Initialize atomic structure and unique element form factors from an input file.

        Parameters:
            structure_path (str): Path to the atomic structure file in XYZ format.
        """
        # XYZ file
        if isinstance(structure_path, str):
            path_ext = os.path.splitext(structure_path)[1]
            if path_ext == '.xyz':
                struc = np.genfromtxt(structure_path, dtype='str', skip_header=2)
                self.struc_elements = struc[:,0]
                self.struc_size = len(self.struc_elements)
                self.num_pairs = self.struc_size * (self.struc_size - 1) // 2
                if struc.shape[1] == 5:
                    self.struc_occupancy = torch.from_numpy(struc[:,-1]).to(device=self.device, dtype=torch.float32)
                    self.struc_xyz = torch.tensor(struc[:,1:-1].astype('float')).to(device=self.device, dtype=torch.float32)
                else:
                    self.struc_occupancy = torch.ones((self.struc_size), dtype=torch.float32).to(device=self.device)
                    self.struc_xyz = torch.tensor(struc[:,1:].astype('float')).to(device=self.device, dtype=torch.float32)
            else:
                raise NotImplementedError('Structure File Extention Not Supported')
        # ASE Atoms object
        elif isinstance(structure_path, Atoms):
            self.struc_elements = structure_path.get_chemical_symbols()
            self.struc_size = len(self.struc_elements)
            self.num_pairs = self.struc_size * (self.struc_size - 1) // 2
            self.struc_occupancy = torch.ones((self.struc_size), dtype=torch.float32).to(device=self.device)
            self.struc_xyz = torch.tensor(np.array(structure_path.get_positions())).to(device=self.device, dtype=torch.float32)
        else:
            raise FileNotFoundError(structure_path)

        # Unique elements and their counts
        unique_elements, inverse, counts = np.unique(self.struc_elements, return_counts=True, return_inverse=True)
        self.triu_indices = torch.triu_indices(self.struc_size, self.struc_size, 1)
        self.unique_inverse = torch.from_numpy(inverse[self.triu_indices]).to(device=self.device)
        self.struc_unique_form_factors = torch.stack([self.form_factor_func(self.FORM_FACTOR_COEF[el]) for el in unique_elements])
        
        # Get f_avg_sqrd and f_sqrd_avg
        counts = torch.from_numpy(counts).to(device=self.device)
        compositional_fractions = counts / torch.sum(counts)
        self.struc_form_avg_sq = torch.sum(compositional_fractions.reshape(-1,1) * self.struc_unique_form_factors, dim=0)**2

        # self scattering
        self.struc_inverse = torch.from_numpy(np.array([inverse[i] for i in range(self.struc_size)])).to(device=self.device)

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
            self.q = torch.arange(self.qmin, self.qmax, self.qstep).unsqueeze(-1).to(device=self.device, dtype=torch.float32)
            self.r = torch.arange(self.rmin, self.rmax, self.rstep).unsqueeze(-1).to(device=self.device, dtype=torch.float32)

    def iq(
        self,
        structure: str,
        keep_on_device: bool = False,
        _total_scattering: bool = False,
    ) -> Union[Tuple[np.float32, np.float32], Tuple[torch.FloatTensor, torch.FloatTensor]]:
        """
        Calculate the scattering intensity I(Q) for the given atomic structure.

        Parameters:
            structure (str): Path to the atomic structure file in XYZ format.
            keep_on_device (bool): Flag to keep the results on the class device. Default is False.
            _total_scattering (bool): Flag to return the scattering intensity I(Q) without the self-scattering contribution. Default is False.

        Returns:
            Tuple of torch tensors containing Q-values and scattering intensity I(Q) if keep_on_device is True, otherwise, numpy arrays on CPU.
        """
        self._initialise_structure(structure)
        if self.profile:
            self.profiler.time('Setup structure and form factors')

        # Calculate distances and batch
        if self.batch_size is None:
            self.batch_size = self._max_batch_size
        dists = pdist(self.struc_xyz).split(self.batch_size)
        indices = self.triu_indices.split(self.batch_size, dim=1)
        inverse_indices = self.unique_inverse.split(self.batch_size, dim=1)
        if self.profile:
            self.profiler.time('Batching and Distances')

        # Calculate scattering using Debye Equation
        iq = torch.zeros((len(self.q))).to(device=self.device, dtype=torch.float32)
        for d, inv_idx, idx in zip(dists, inverse_indices, indices):
            mask = d >= self.rthres
            occ_product = self.struc_occupancy[idx[0]] * self.struc_occupancy[idx[1]]
            sinc = torch.sinc(d[mask] * self.q / torch.pi)
            ffp = self.struc_unique_form_factors[inv_idx[0]] * self.struc_unique_form_factors[inv_idx[1]]
            iq += torch.sum(occ_product.unsqueeze(-1) * ffp * sinc.permute(1,0), dim=0)

        # Apply Debye-Weller Isotropic Atomic Displacement
        if self.biso != 0.0:
            iq *= torch.exp(-self.q.squeeze(-1).pow(2) * self.biso/(8*torch.pi**2))
        
        # For total scattering
        if _total_scattering:
            if self.profile:
                self.profiler.time('I(Q)')
            return iq # TODO Times 2

        # Self-scattering contribution
        sinc = torch.ones((self.struc_size, len(self.q))).to(device=self.device)
        iq += torch.sum((self.struc_occupancy.unsqueeze(-1) * self.struc_unique_form_factors[self.struc_inverse])**2 * sinc, dim=0) / 2
        iq *= 2

        if self.profile:
            self.profiler.time('I(Q)')

        if keep_on_device:
            return self.q.squeeze(-1), iq
        else:
            return self.q.squeeze(-1).cpu().numpy(), iq.cpu().numpy()

    def sq(
        self,
        structure: str,
        keep_on_device: bool = False,
    ) -> Union[Tuple[np.float32, np.float32], Tuple[torch.FloatTensor, torch.FloatTensor]]:
        """
        Calculate the structure function S(Q) for the given atomic structure.

        Parameters:
            structure (str): Path to the atomic structure file in XYZ format.
            keep_on_device (bool): Flag to keep the results on the class device. Default is False.

        Returns:
            Tuple of torch tensors containing Q-values and structure function S(Q) if keep_on_device is True, otherwise, numpy arrays on CPU.
        """
        # Calculate Scattering S(Q)
        iq = self.iq(structure, keep_on_device=True, _total_scattering=True)
        sq = iq/self.struc_form_avg_sq/self.struc_size
        if keep_on_device:
            return self.q.squeeze(-1), sq
        else:
            return self.q.squeeze(-1).cpu().numpy(), sq.cpu().numpy()
    
    def fq(
        self,
        structure: str,
        keep_on_device: bool = False,
    ) -> Union[Tuple[np.float32, np.float32], Tuple[torch.FloatTensor, torch.FloatTensor]]:
        """
        Calculate the reduced structure function F(Q) for the given atomic structure.

        Parameters:
            structure (str): Path to the atomic structure file in XYZ format.
            keep_on_device (bool): Flag to keep the results on the class device. Default is False.

        Returns:
            Tuple of torch tensors containing Q-values and reduced structure function F(Q) if keep_on_device is True, otherwise, numpy arrays on CPU.
        """
        # Calculate Scattering S(Q)
        iq = self.iq(structure, keep_on_device=True, _total_scattering=True)
        sq = iq/self.struc_form_avg_sq/self.struc_size
        fq = self.q.squeeze(-1) * sq
        if keep_on_device:
            return self.q.squeeze(-1), fq
        else:
            return self.q.squeeze(-1).cpu().numpy(), fq.cpu().numpy()

    def gr(
        self,
        structure: str,
        keep_on_device: bool = False,
    ) -> Union[Tuple[np.float32, np.float32], Tuple[torch.FloatTensor, torch.FloatTensor]]:
        """
        Calculate the reduced pair distribution function G(r) for the given atomic structure.

        Parameters:
            structure (str): Path to the atomic structure file in XYZ format.
            keep_on_device (bool): Flag to keep the results on the class device. Default is False.

        Returns:
            Tuple of torch tensors containing Q-values and PDF G(r) if keep_on_device is True, otherwise, numpy arrays on CPU.
        """
        # Calculate Scattering I(Q), S(Q), F(Q)
        if self.profile:
            self.profiler.reset()
        iq = self.iq(structure, keep_on_device=True, _total_scattering=True)
        sq = iq/self.struc_form_avg_sq/self.struc_size
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

        if keep_on_device:
            return self.r.squeeze(-1), gr
        else:
            return self.r.squeeze(-1).cpu().numpy(), gr.cpu().numpy()

    def _get_all(
        self,
        structure: str,
        keep_on_device: bool = False,
    ) -> Union[Tuple[np.float32,...], Tuple[torch.FloatTensor,...]]:
        """
        Calculate I(Q), S(Q), F(Q) and G(r) for the given atomic structure and return all.

        Parameters:
            structure (str): Path to the atomic structure file in XYZ format.
            keep_on_device (bool): Flag to keep the results on the class device. Default is False.

        Returns:
            Tuple of torch tensors containing r-values, Q-values and I(Q), S(Q), F(Q) and G(r) if keep_on_device is True, otherwise, numpy arrays on CPU.
        """
        self._initialise_structure(structure)

        # Calculate distances and batch
        if self.batch_size is None:
            self.batch_size = self._max_batch_size
        dists = pdist(self.struc_xyz).split(self.batch_size)
        indices = self.triu_indices.split(self.batch_size, dim=1)
        inverse_indices = self.unique_inverse.split(self.batch_size, dim=1)

        # Calculate scattering using Debye Equation
        iq = torch.zeros((len(self.q))).to(device=self.device, dtype=torch.float32)
        for d, inv_idx, idx in zip(dists, inverse_indices, indices):
            mask = d >= self.rthres
            occ_product = self.struc_occupancy[idx[0]] * self.struc_occupancy[idx[1]]
            sinc = torch.sinc(d[mask] * self.q / torch.pi)
            ffp = self.struc_unique_form_factors[inv_idx[0]] * self.struc_unique_form_factors[inv_idx[1]]
            iq += torch.sum(occ_product.unsqueeze(-1) * ffp * sinc.permute(1,0), dim=0)

        # Apply Debye-Weller Isotropic Atomic Displacement
        if self.biso != 0.0:
            iq *= torch.exp(-self.q.squeeze(-1).pow(2) * self.biso/(8*torch.pi**2))
        
        # Calculate S(Q), F(Q) and G(r)
        sq = iq/self.struc_form_avg_sq/self.struc_size
        fq = self.q.squeeze(-1) * sq
        damp = 1 if self.qdamp == 0.0 else torch.exp(-(self.r.squeeze(-1) * self.qdamp).pow(2) / 2)
        lorch_mod = 1 if self.lorch_mod == None else torch.sinc(self.q * self.lorch_mod*(torch.pi / self.qmax))
        gr = (2 / torch.pi) * torch.sum(fq.unsqueeze(-1) * torch.sin(self.q * self.r.permute(1,0))*self.qstep * lorch_mod, dim=0) * damp

        # Self-scattering contribution / Calculate I(Q)
        sinc = torch.ones((self.struc_size, len(self.q))).to(device=self.device)
        iq += torch.sum((self.struc_occupancy.unsqueeze(-1) * self.struc_unique_form_factors[self.struc_inverse])**2 * sinc, dim=0) / 2
        iq *= 2

        if keep_on_device:
            return self.r.squeeze(-1), self.q.squeeze(-1), iq, sq, fq, gr
        else:
            return self.r.squeeze(-1).cpu().numpy(), self.q.squeeze(-1).cpu().numpy(), iq.cpu().numpy(), sq.cpu().numpy(), fq.cpu().numpy(), gr.cpu().numpy()

    def generate_nanoparticles(
        self,
        structure_path: str,
        radii: Union[List[float], float],
        sort_atoms: bool = True,
        _override_device: bool = True,
    ) -> Tuple[Union[List[Atoms], Atoms, None], Union[List[float], float, None]]:
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
            radii = radii
            single_flag = False
        elif isinstance(radii, float):
            radii = [radii]
            single_flag = True
        elif isinstance(radii, int):
            radii = [float(radii)]
            single_flag = True
        else:
            print('FAILED: Please provide valid radii for generation of nanoparticles')
            return None, None

        # DEV: Override device
        device = 'cpu' if _override_device else self.device

        # Read the input unit cell structure
        unit_cell = read(structure_path)
        cell_dims = np.array(unit_cell.cell.cellpar()[:3])
        r_max = np.amax(radii)
    
        # Create a supercell to encompass the entire range of nanoparticles and center it
        supercell_matrix = np.diag((np.ceil(r_max / cell_dims)) * 2)
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

        # Update the cell positions
        cell.positions = positions.cpu()
    
        # Initialize nanoparticle lists and progress bar
        nanoparticle_list = []
        nanoparticle_sizes = []
        pbar = tqdm(desc=f'Generating nanoparticles in range: [{radii[0]},{radii[-1]}]', leave=False, total=len(radii))
    
        # Generate nanoparticles for each radius
        for r in radii:

            # Mask all atoms within radius
            incl_mask = (center_dists <= r)

            # Find interdistances from all included atoms and all atoms
            interface_dists = cdist(positions, positions[incl_mask])
    
            # Find interface atoms and determine nanoparticle size
            nanoparticle_size = 0
            for i in range(interface_dists.shape[0]):
                # Interface mask: all atoms within the min metal distance from the interface that is not a metal
                interface_mask = (interface_dists[i] <= min_metal_dist) & ~metal_filter[i]

                # If any interface atoms that should be included
                if torch.any(interface_mask):
                    nanoparticle_size = max(nanoparticle_size, center_dists[i] * 2)
                    incl_mask[i] = True
    
            # Append size
            nanoparticle_sizes.append(nanoparticle_size.item())
    
            # Extract the nanoparticle from the supercell
            np_cell = cell[incl_mask.cpu()]

            # Sort the atoms
            if sort_atoms:
                np_cell = ase_sort(np_cell)
                if np_cell.get_chemical_symbols()[0] in ligands:
                    np_cell = np_cell[::-1]
    
            # Append nanoparticle
            nanoparticle_list.append(np_cell)
            pbar.update(1)
        pbar.close()
    
        if single_flag:
            return nanoparticle_list[0], nanoparticle_sizes[0]
        else:
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
            else:
                return False # Other cases
        except NameError:
            return False # Standard Python Interpreter

    def interact(
        self,
        structure_path: Union[str, None] = None,
        _cont_updates: bool = False,
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

        # Check if path is given else Default
        disable_input = False if structure_path is None else True

        qmin = self.qmin
        qmax = self.qmax
        qstep = self.qstep
        qdamp = self.qdamp
        rmin = self.rmin
        rmax = self.rmax
        rstep = self.rstep
        rthres = self.rthres
        biso = self.biso
        device = self.device
        batch_size = self.batch_size
        lorch_mod = self.lorch_mod
        radiation_type = self.radiation_type
        profile = False
        
        # Radiation type button
        radtype_btn = widgets.ToggleButtons(
            options=['xray', 'neutron'],
            value=radiation_type,
            description='Rad. type',
            layout = widgets.Layout(width='900px'),
            button_style='info'
        )

        # Path button
        path_btn = widgets.Text(
            value='',
            placeholder="",
            description='Data directory',
            disabled = disable_input,
        )
        
        # Device button
        device_btn = widgets.ToggleButtons(
            options=['cpu', 'cuda'],
            value=device,
            description='Hardware:',
            button_style='info',
        )
    
        # Device batch_size button
        batch_size_btn = widgets.IntText(
            min = 100,
            max = 10000,
            value=batch_size,
            description='Hardware batch-size',
            style = {'description_width': 'initial'},
        )

        # Q value min/max slider
        qslider = widgets.FloatRangeSlider(
            value=[qmin, qmax],
            min=0.0,
            max=50.0,
            step=0.01,
            description='Qmin/Qmax:',
            continuous_update=_cont_updates,
            orientation='horizontal',
            readout=True,
            style={'font_weight':'bold', 'slider_color': 'white'},
            layout = widgets.Layout(width='900px'),
        )
    
        # r value slider
        rslider = widgets.FloatRangeSlider(
            value=[rmin, rmax],
            min=0,
            max=100.0,
            step=rstep,
            description='rmin/rmax:',
            continuous_update=_cont_updates,
            orientation='horizontal',
            readout=True,
            style={'font_weight':'bold', 'slider_color': 'white'},
            layout = widgets.Layout(width='900px'),
        )
        
        # Qdamp slider
        qdamp_slider = widgets.FloatSlider(
            min=0.00,
            max=0.10,
            value=qdamp, 
            step=0.01,
            description='Qdamp:',
            layout = widgets.Layout(width='900px'),
            continuous_update=_cont_updates,
        )
        
        # Biso slider
        biso_slider = widgets.FloatSlider(
            min=0.00,
            max=1.00,
            value=biso,
            step=0.01,
            description='B-iso:',
            continuous_update=_cont_updates,
            layout = widgets.Layout(width='900px'),
        )
        
        # Qstep button
        qstep_btn = widgets.FloatText(
            min = 0.001,
            max = 1,
            step=0.001,
            value=qstep,
            description='Qstep:',
        )
        
        # rstep button
        rstep_btn = widgets.FloatText(
            min = 0.001,
            max = 1,
            step=0.001,
            value=rstep,
            description='rstep:',
        )
        
        # rthres button
        rthres_btn = widgets.FloatText(
            min = 0.001,
            max = 1,
            step=0.001,
            value=rthres,
            description='rthres:',
        )
        
        # Lorch modification button
        lorch_mod_btn = widgets.Checkbox(
            value=lorch_mod,
            description='Lorch modification',
            style = {'description_width': 'initial'},
        )

        # Scale type button
        scale_type_btn = widgets.ToggleButtons(
            options=['linear', 'logarithmic'],
            value='linear',
            description='Axes scaling:',
            button_style='info'
        )
        
        # Download options
        def create_download_link(filename_prefix, data, header=None):

            # Collect Metadata
            metadata = str({'qmin': qslider.value[0], 'qmax': qslider.value[1], 'qdamp': qdamp_slider.value, 'qstep': qstep_btn.value,
                          'rmin': rslider.value[0], 'rmax': rslider.value[1], 'rstep': rstep_btn.value, 'rthres': rthres_btn.value,
                          'biso': biso_slider.value, 'device': device_btn.value, 'batch_size': batch_size_btn.value, 'lorch_mod': lorch_mod_btn.value,
                          'radiation_type': radtype_btn.value}) + "\n"

            # Content
            content = "\n".join([",".join(map(str, row)) for row in data])

            # Add Header
            if header:
                content = metadata + "\n" + header + "\n" + content
            else:
                content = metadata + "\n" + content

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
            
            # Filename
            filename = filename_prefix + '_' + select_file.value.split('/')[-1].split('.')[0] + '_' + month + day + year + '_' + hours + minutes + seconds + '.csv'
    
            # Make href and return
            href = f'<a href="data:text/csv;base64,{b64}" download="{filename}">Download {filename} (Created: {hours}:{minutes}:{seconds})</a>'
            return href

        def on_download_button_click(button):
            # Try to compile all the data and create html link to download files
            try:
                # Data
                iq_data = np.column_stack([q, iq_values])
                sq_data = np.column_stack([q, sq_values])
                fq_data = np.column_stack([q, fq_values])
                gr_data = np.column_stack([r, gr_values])
            
                # Clear warning message
                sys.stdout.write('\x1b[1A')
                sys.stdout.write('\x1b[1A')
                sys.stdout.write('\x1b[1A')
                sys.stdout.write('\x1b[2K')
    
                # Display download links
                display(HTML(create_download_link('iq', iq_data, "q,I(Q)")))
                display(HTML(create_download_link('sq', sq_data, "q,S(Q)")))
                display(HTML(create_download_link('fq', fq_data, "q,F(Q)")))
                display(HTML(create_download_link('gr', gr_data, "r,G(r)")))

                print('=' * 10)
                
            except Exception as e:
                #raise(e)
                print('WARNING: Data not available', end="\r")
                  
        # Download buttons
        download_button = widgets.Button(description="Download Data")
        download_button.on_click(on_download_button_click)
    
        # Create a color dropdown widget
        folder = widgets.Text(description='Data directory:', placeholder='Provide data directory', disabled=disable_input)
    
        # Create a dropdown menu widget for selection of XYZ file and an output area
        standard_msg = ''
        option_values = standard_msg# if structure_path is None else structure_path
        select_file = widgets.Dropdown(description='Select File:', options=[option_values], value=standard_msg, disabled=True)
    
        # Define a function to update the scattering patterns based on the selected parameters
        def update_options(change):
            folder = change.new
            paths = sorted(glob(os.path.join(folder, '*.xyz')))
            if len(paths):
                select_file.options = ['Select data file'] + paths #[path.split('/')[-1] for path in paths]
                select_file.value = 'Select data file'
                select_file.disabled = False
            else:
                select_file.options = [standard_msg]
                select_file.value = standard_msg
                select_file.disabled = True
    
        # Link the update function to the dropdown widget's value change event
        folder.observe(update_options, names='value')
    
        # Create a function to update the output area
        def update_output(
            folder,
            path,
            device,
            batch_size,
            radtype,
            scale_type,
            qminmax,
            rminmax,
            qdamp,
            biso,
            qstep,
            rstep,
            rthres,
            lorch_mod,
        ):
            global q, r, iq_values, sq_values, fq_values, gr_values  # Declare these variables as global

            try:
                path_ext = path.split('.')[-1]
            except:
                return

            if (path is not None) and path != standard_msg and path_ext in ['xyz', 'cif']:

                try:
                    calculator = DebyeCalculator(device=device, batch_size=batch_size, radiation_type=radtype,
                                                 qmin=qminmax[0], qmax=qminmax[1], qstep=qstep, qdamp=qdamp,
                                                 rmin=rminmax[0], rmax=rminmax[1], rstep=rstep, rthres=rthres, biso=biso,
                                                 lorch_mod=lorch_mod)
    
                    r, q, iq_values, sq_values, fq_values, gr_values = calculator._get_all(path)
    
                    fig, axs = plt.subplots(2, 2, figsize=(12, 8), dpi=75)
                    axs = axs.flatten()
    
                    if scale_type == 'logarithmic':
                        axs[0].set_xscale('log')
                        axs[0].set_yscale('log')
    
                    axs[0].plot(q, iq_values, lw=None)
                    axs[0].set(xlabel='$Q$ [$\AA^{-1}$]', ylabel='$I(Q)$ [counts]')
                    axs[1].axhline(1, alpha=0.5, ls='--', c='g')
                    axs[1].plot(q, sq_values+1, lw=None)
                    axs[1].set(xlabel='$Q$ [$\AA^{-1}$]', ylabel='$S(Q)$')
                    axs[2].axhline(0, alpha=0.5, ls='--', c='g')
                    axs[2].plot(q, fq_values, lw=None)
                    axs[2].set(xlabel='$Q$ [$\AA^{-1}$]', ylabel='$F(Q)$')
                    axs[3].plot(r, gr_values, lw=None)
                    axs[3].set(xlabel='$r$ [$\AA$]', ylabel='$G_r(r)$')
    
                    labels = ['Scattering Intensity, I(Q)',
                              'Structure Function, S(Q)',
                              'Reduced Structure Function, F(Q)',
                              'Reduced Pair Distribution Function, G(r)']
    
                    for ax, label in zip(axs, labels):
                        ax.relim()
                        ax.autoscale_view()
                        ax.set_title(label)
                        ax.grid(alpha=0.2)
    
                    fig.suptitle("XYZ file: " + path.split('/')[-1].split('.')[0])
                    fig.tight_layout()
                except:
                    print(f'WARNING: Could not load data file: {path}', end='\r')
           
    
        # Create an interactive function that triggers when the user-defined parameters changes
        interact(
            update_output, 
            folder = folder,
            path = select_file,
            batch_size = batch_size_btn,
            device = device_btn,
            radtype = radtype_btn,
            lorch_mod = lorch_mod_btn,
            qminmax = qslider,
            qstep = qstep_btn,
            qdamp = qdamp_slider,
            rminmax = rslider,
            rstep = rstep_btn,
            rthres = rthres_btn,
            biso = biso_slider,
            scale_type = scale_type_btn,
        );
        
        # Lastly, display download button
        display(download_button)
