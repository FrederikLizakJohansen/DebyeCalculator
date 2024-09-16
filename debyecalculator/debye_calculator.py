import os
import re
import math
import tempfile
from io import BytesIO
import zipfile
import sys
import base64
import hashlib
import time
import yaml
import importlib.resources
import warnings
from glob import glob
from datetime import datetime, timezone
from typing import Union, Tuple, Any, List, Type
from collections import namedtuple

# Handle import of torch (prerequisite)
try:
    import torch
    from torch import cdist
    from torch.nn.functional import pdist
except ModuleNotFoundError:
    raise ImportError(
        "\n\nDebyeCalculator requires PyTorch, which is not installed. "
        "Please install PyTorch before using DebyeCalculator. "
        "Follow the instructions on the official PyTorch website: "
        "https://pytorch.org/get-started/locally/. "
        "For more information about DebyeCalculator, visit the GitHub repository: "
        "https://github.com/FrederikLizakJohansen/DebyeCalculator"
    )


import numpy as np
import matplotlib.pyplot as plt

from ase import Atoms
from ase.io import read, write
from ase.build import make_supercell
from ase.build.tools import sort as ase_sort

from debyecalculator.utility.profiling import Profiler
from debyecalculator.utility.generate import generate_nanoparticles

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from ipywidgets import HBox, VBox, Layout
from functools import partial
from tqdm.auto import tqdm

try:
    from pymatgen.core import Structure
    pymatgen_available = True
except ImportError:
    pymatgen_available = False

# NamedTuple definitions
StructureTuple = namedtuple('StructureTuple', 'elements size occupancy xyz triu_indices unique_inverse unique_form_factors form_avg_sq structure_inverse')
IqTuple = namedtuple('IqTuple', 'q i')
SqTuple = namedtuple('SqTuple', 'q s')
FqTuple = namedtuple('FqTuple', 'q f')
GrTuple = namedtuple('GrTuple', 'r g')
AllTuple = namedtuple('AllTuple', 'r q i s f g')

ArrayLike = Union[np.ndarray, torch.Tensor]
IntArrayLike = Union[List[int], np.ndarray, torch.Tensor]

if pymatgen_available:
    StructureSourceType = Union[
        Tuple[List[str], ArrayLike],
        Tuple[IntArrayLike, ArrayLike],
        List[Tuple[List[str], ArrayLike]],
        List[Tuple[IntArrayLike, ArrayLike]],
        str,
        List[str],
        Atoms,
        List[Atoms],
        Structure,
        List[Structure],
    ]
else:
    StructureSourceType = Union[
        Tuple[List[str], ArrayLike],
        Tuple[IntArrayLike, ArrayLike],
        List[Tuple[List[str], ArrayLike]],
        List[Tuple[IntArrayLike, ArrayLike]],
        str,
        List[str],
        Atoms,
        List[Atoms],
    ]

class DebyeCalculator:
    """
    Calculate the scattering intensity I(Q) through the Debye scattering equation, the Total Scattering Structure Function S(Q), 
    the Reduced Total Scattering Function F(Q), and the Reduced Atomic Pair Distribution Function G(r) for a given atomic structure.
    """

    def __init__(
        self,
        qmin: float = 1.0,
        qmax: float = 30.0,
        qstep: float = None,
        qdamp: float = 0.04,
        rmin: float = 0.0,
        rmax: float = 20.0,
        rstep: float = 0.01,
        rthres: float = 0.0,
        biso: float = 0.3,
        device: str = 'cuda',
        batch_size: Union[int, None] = 10000,
        lorch_mod: bool = False,
        radiation_type: str = None,
        rad_type: str = None,
        profile: bool = False,
        _max_batch_size: int = 4000,
        _lightweight_mode: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Initialize a DebyeCalculator instance with specified parameters.

        Args:
            qmin (float): Minimum q-value for the scattering calculation. Default is 1.0.
            qmax (float): Maximum q-value for the scattering calculation. Default is 30.0.
            qstep (float): Step size for the q-values in the scattering calculation. Default is None, and is calculated as [pi / (rmax + rstep)] if not provided.
            qdamp (float): Damping parameter caused by the truncated Q-range of the Fourier transformation. Default is 0.04.
            rmin (float): Minimum r-value for the pair distribution function (PDF) calculation. Default is 0.0.
            rmax (float): Maximum r-value for the PDF calculation. Default is 20.0.
            rstep (float): Step size for the r-values in the PDF calculation. Default is 0.01.
            rthres (float): Threshold value for exclusion of distances below this value in the scattering calculation. Default is 0.0.
            biso (float): Debye-Waller isotropic atomic displacement parameter. Default is 0.3.
            device (str): Device to use for computations ('cuda' for CUDA-enabled GPU's or 'cpu' for CPU)
            batch_size (int or None): Batch size for computation. If None, the batch size will be automatically set. Default is None.
            lorch_mod (bool): Flag to enable Lorch modification. Default is False.
            radiation_type (str): Type of radiation for form factor calculations ('xray' or 'neutron'). Default is 'xray'
            rad_type (str): Alias for 'radiation_type'. Default is 'xray'.
            profile (bool): Activate profiler. Default is False.
            dtype (torch.dtype): Data type for tensors. 32-bit and 64-bit floats are currently supported. Default is torch.float32.
        """

        # Handling CUDA availability
        if device == 'cuda' and not torch.cuda.is_available():
            warnings.warn("Warning: Your system might have a CUDA-enabled GPU, but CUDA is not available. Computations will run on the CPU instead. " \
                          "For optimal performance, please install Pytorch with CUDA support. " \
                          "If you do not have a CUDA-enabled GPU, you can surpress this warning by specifying the 'device' argument as 'cpu'", 
                          UserWarning, 
                          stacklevel=2
                         )
            self.device = 'cpu'
        elif device == 'cpu' and torch.cuda.is_available():
            warnings.warn("Warning: Your system has a CUDA-enabled GPU, but CPU was explicitly specified for computations. " \
                          "To utilise GPU acceleration, omit the 'device' argument or specify 'cuda'", 
                          UserWarning,
                          stacklevel=2
                         )
            self.device = 'cpu'
        else:
            self.device = device
            
        # Set parameters
        self.qmin = qmin
        self.qmax = qmax
        self.qstep = qstep
        self.qdamp = qdamp
        self.rmin = rmin
        self.rmax = rmax
        self.rstep = rstep
        self.rthres = rthres
        self.biso = biso
        self.batch_size = batch_size
        self.lorch_mod = lorch_mod
        self.dtype = dtype

        if radiation_type is not None:
            self.radiation_type = radiation_type
        elif rad_type is not None:
            self.radiation_type = rad_type
        else:
            self.radiation_type = 'xray'

        # Set qstep seperately (Nyquist)
        if qstep is None:
            self.qstep = math.pi / (self.rmax + self.rstep)
        else:
            self.qstep = qstep

        # Parameter constraint assertion
        self.parameter_constraint_assertion()

        # Profiler
        self.profile = profile
        if self.profile:
            self.profiler = Profiler()
        
        # Initialise ranges
        self.q = torch.arange(self.qmin, self.qmax, self.qstep).unsqueeze(-1).to(device=self.device, dtype=self.dtype)
        self.r = torch.arange(self.rmin, self.rmax, self.rstep).unsqueeze(-1).to(device=self.device, dtype=self.dtype)

        with importlib.resources.open_text('debyecalculator.utility', 'elements_info.yaml') as yaml_file:
            self.FORM_FACTOR_COEF = yaml.safe_load(yaml_file)

        # Form factor coefficients
        self.atomic_numbers_to_elements = {}
        for i, (key, value) in enumerate(self.FORM_FACTOR_COEF.items()):
            if i > 97:
                break
            self.atomic_numbers_to_elements[value[12]] = key

        # Formfactor retrieval lambda
        for k,v in self.FORM_FACTOR_COEF.items():
            if None in v:
                v = [value if value is not None else np.nan for value in v]
            self.FORM_FACTOR_COEF[k] = torch.tensor(v).to(device=self.device, dtype=self.dtype)
        if self.radiation_type.lower() in ['xray', 'x']:
            self.form_factor_func = lambda p: torch.sum(p[:5] * torch.exp(-1*p[6:11] * (self.q / (4*torch.pi)).pow(2)), dim=1) + p[5]
        elif self.radiation_type.lower() in ['neutron', 'n']:
            self.form_factor_func = lambda p: p[11].unsqueeze(-1)
        else:
            # Should not reach this point, here for safety
            raise ValueError("Invalid radiation type")

        # Max batch size
        self._max_batch_size = _max_batch_size
        
        # Lightweight mode
        self._lightweight_mode = _lightweight_mode

    def __repr__(self):
        return (
            f"{self.__class__.__name__} instance:\n"
            f"{'Q-min:':<12} {self.qmin:5.2f}\n"
            f"{'Q-max:':<12} {self.qmax:5.2f}\n"
            f"{'Q-step:':<12} {self.qstep:5.2f}\n"
            f"{'Q-damp:':<12} {self.qdamp:5.2f}\n"
            f"\n"
            f"{'r-min:':<12} {self.rmin:5.2f}\n"
            f"{'r-max:':<12} {self.rmax:5.2f}\n"
            f"{'r-step:':<12} {self.rstep:5.2f}\n"
            f"{'r-thres:':<12} {self.rthres:5.2f}\n"
            f"\n"
            f"{'B-iso:':<12} {self.biso:5.2f}\n"
            f"\n"
            f"{'rad_type:':<12} {self.radiation_type}\n"
            f"{'lorch_mod:':<12} {self.lorch_mod}\n"
            f"\n"
            f"{'batch_size:':<12} {self.batch_size}\n"
            f"{'device:':<12} {self.device}\n"
            f"{'profile:':<12} {self.profile}\n"
        )

    def parameter_constraint_assertion(
        self,
    ) -> None:
        """
        Assert that all parameters meet valid constraints.
        Provides warning if certain parameters are not within certain constraints

        Raises:
            ValueError: If any of the parameters violate the specified constraints.
            UserWarning: If any of the specified parameters are not within the specified constraints.
        """

        # Positivity constraints
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

        # Batch-size constraints
        if self.batch_size is not None and self.batch_size < 0:
            raise ValueError("batch_size must be non-negative.")

        # Device constraints
        if self.device not in ['cpu', 'cuda']:
            raise ValueError("Invalid device")

        # Radiation type constraints
        if self.radiation_type not in ['xray', 'x', 'neutron', 'n']:
            raise ValueError("Invalid radiation type")

        # Optimal qstep UserWarning
        optimal_qstep = (math.pi / (self.rmax + self.rstep))
        if self.qstep > optimal_qstep:
            warnings.warn(
                f"The qstep that was chosen is too large and might result in unwanted signal artifacts. With rmax={self.rmax} and rstep={self.rstep}, consider using qstep<={optimal_qstep:2.3f}.",
                UserWarning
            )
        
        # Dtype
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError(
                f"Invalid dtype {self.dtype}. Only torch.float32 and torch.float64 are supported."
            )
    
    def update_parameters(
        self,
        **kwargs: Any,
    ) -> None:
        """
        Set or update the parameters of the DebyeCalculator.

        Parameters:
            **kwargs: Arbitrary keyword arguments to update the parameters.

        Raises:
            ValueError: If any of the updated parameters violate the specified constraints.
        """
            
        for k,v in kwargs.items():
            try:
                setattr(self, k, v)
                if k == 'rad_type':
                    setattr(self, 'radiation_type', v)
            except:
                print("Failed to update parameters because of unexpected parameter names")
                return

        # Run constrain assertion
        self.parameter_constraint_assertion()

        # Re-initialise ranges
        if np.any([k in ['qmin','qmax','qstep','rmin', 'rmax', 'rstep', 'device', 'dtype'] for k in kwargs.keys()]):
            self.q = torch.arange(self.qmin, self.qmax, self.qstep).unsqueeze(-1).to(device=self.device, dtype=self.dtype)
            self.r = torch.arange(self.rmin, self.rmax, self.rstep).unsqueeze(-1).to(device=self.device, dtype=self.dtype)
            for key,val in self.FORM_FACTOR_COEF.items():
                self.FORM_FACTOR_COEF[key] = val.to(device=self.device)

        if np.any([k in ['rad_type', 'radiation_type'] for k in kwargs.keys()]):
            if self.radiation_type.lower() in ['xray', 'x']:
                self.form_factor_func = lambda p: torch.sum(p[:5] * torch.exp(-1*p[6:11] * (self.q / (4*torch.pi)).pow(2)), dim=1) + p[5]
            elif self.radiation_type.lower() in ['neutron', 'n']:
                self.form_factor_func = lambda p: p[11].unsqueeze(-1)
            else:
                # Should not reach this point, here for safety
                raise ValueError("Invalid radiation type")


    def _initialise_structure(
        self,
        structure_source: StructureSourceType,
        radii: Union[List[float], float, None] = None,
        disable_pbar: bool = False,
    ) -> Union[None, StructureTuple, List[StructureTuple]]:

        """
        Initialize a single atomic structure and unique elements form factors from an input file or Atoms object.

        Parameters:
            structure_source (StructureSourceType): Atomic structure source in XYZ/CIF format, ASE Atoms object, or as a tuple of (atomic_identities, atomic_positions).
            radii (Union[List[float], float, None]): List/float of radii/radius of particle(s) to generate with parsed CIF.
            disable_pbar (bool): Flag to disable the progress bar during nanoparticle generation. Default is False.

        Returns:
            Union[None, StructureTuple, List[StructureTuple]]: The initialized structure(s) as StructureTuple or list of StructureTuple objects.

        Raises:
            TypeError: If the structure source is of an invalid type.
            IOError: If there is an issue loading the structure from the specified file.
            ValueError: If the file extension is not valid or when providing .cif data file, radii is not provided.
        """

        def is_valid_str_tuple(t: Tuple[List[str], ArrayLike]) -> bool:
            check_len = len(t) == 2
            check_type = isinstance(t[0], list) and isinstance(t[1], (np.ndarray, torch.Tensor))
            check_shape = len(t[1].shape) == 2 and t[1].shape[1] == 3
            
            return check_len and check_type and check_shape

        def is_valid_int_tuple(t: Tuple[IntArrayLike, ArrayLike]) -> bool:
            check_len = len(t) == 2
            check_type = isinstance(t[0], (np.ndarray, torch.Tensor)) and isinstance(t[1], (np.ndarray, torch.Tensor))
            check_shape = len(t[0].shape) == 1 and len(t[1].shape) == 2 and t[1].shape[1] == 3

            return check_len and check_type and check_shape

        def parse_elements(elements, size):
            # Get unique elements and construc form factor stacks
            unique_elements, inverse, counts = np.unique(elements, return_counts=True, return_inverse=True)

            triu_indices = torch.triu_indices(size, size, 1)
            unique_inverse = torch.from_numpy(inverse[triu_indices]).to(device=self.device)
            unique_form_factors = torch.stack([self.form_factor_func(self.FORM_FACTOR_COEF[el]) for el in unique_elements])

            # Calculate average squared form factor and self scattering inverse indices
            counts = torch.from_numpy(counts).to(device=self.device)
            compositional_fractions = counts / torch.sum(counts)
            form_avg_sq = torch.sum(compositional_fractions.reshape(-1,1) * unique_form_factors, dim=0)**2
            structure_inverse = torch.from_numpy(np.array([inverse[i] for i in range(size)])).to(device=self.device)

            return triu_indices, unique_inverse, unique_form_factors, form_avg_sq, structure_inverse

        if isinstance(structure_source, tuple):
            if is_valid_str_tuple(structure_source):
                elements, xyz = structure_source
                size = xyz.shape[0]
                if isinstance(xyz, np.ndarray):
                    xyz = torch.from_numpy(xyz)
                xyz = xyz.to(device=self.device, dtype=self.dtype)
                occupancy = torch.ones(xyz.shape[0]).to(device=self.device, dtype=self.dtype)
                triu_indices, unique_inverse, unique_form_factors, form_avg_sq, structure_inverse = parse_elements(elements, size)

                return StructureTuple(elements, size, occupancy, xyz, triu_indices, unique_inverse, unique_form_factors, form_avg_sq, structure_inverse)

            elif is_valid_int_tuple(structure_source):

                elements, xyz = structure_source
                elements = [self.atomic_numbers_to_elements[e] for e in elements]
                size = xyz.shape[0]
                if isinstance(xyz, np.ndarray):
                    xyz = torch.from_numpy(xyz)
                xyz = xyz.to(device=self.device, dtype=self.dtype)
                occupancy = torch.ones(xyz.shape[0]).to(device=self.device, dtype=self.dtype)
                triu_indices, unique_inverse, unique_form_factors, form_avg_sq, structure_inverse = parse_elements(elements, size)

                return StructureTuple(elements, size, occupancy, xyz, triu_indices, unique_inverse, unique_form_factors, form_avg_sq, structure_inverse)
            else:
                raise TypeError('Encountered an invalid structure source (type: tuple)')
        elif isinstance(structure_source, str):
            try:
                ext = structure_source.split('.')[-1]
            except:
                raise TypeError(f'Encountered invalid file path on {structure_source}')
            if ext == 'xyz':
                try:
                    structure = np.genfromtxt(structure_source, dtype='str', skip_header=2)
                    elements = structure[:,0]
                    size = len(elements)

                    # Append occupancy if nothing is provided
                    if structure.shape[1] == 5:
                        occupancy = torch.from_numpy(structure[:,-1]).to(device=self.device, dtype=self.dtype)
                        xyz = torch.tensor(structure[:,1:-1].astype('float')).to(device=self.device, dtype=self.dtype)
                    else:
                        occupancy = torch.ones((size), dtype=self.dtype).to(device=self.device)
                        xyz = torch.tensor(structure[:,1:].astype('float')).to(device=self.device, dtype=self.dtype)
                except:
                    raise IOError(f'Encountered invalid file format when trying to load structure from {structure_source}')
                    
                triu_indices, unique_inverse, unique_form_factors, form_avg_sq, structure_inverse = parse_elements(elements, size)

                return StructureTuple(elements, size, occupancy, xyz, triu_indices, unique_inverse, unique_form_factors, form_avg_sq, structure_inverse)

            elif ext == 'cif':
                if radii is not None:
                    structures = generate_nanoparticles(structure_source, radii, disable_pbar=disable_pbar, _lightweight_mode=self._lightweight_mode, device=self.device)
                    structure_tuple_list = []
                    for structure in structures:
                        triu_indices, unique_inverse, unique_form_factors, form_avg_sq, structure_inverse = parse_elements(structure.elements, structure.size)
                        structure_tuple_list.append(
                            StructureTuple(
                                elements = structure.elements,
                                size = structure.size,
                                occupancy = structure.occupancy.to(dtype=self.dtype, device=self.device),
                                xyz = structure.xyz.to(dtype=self.dtype, device=self.device),
                                triu_indices = triu_indices,
                                unique_inverse = unique_inverse,
                                unique_form_factors = unique_form_factors,
                                form_avg_sq = form_avg_sq,
                                structure_inverse = structure_inverse
                            )
                        )

                    return structure_tuple_list
                else:
                    raise ValueError('When providing .cif data file, please provide radii (Union[List[float], float]) for the decrete particle generation')
            else:
                raise TypeError(f'Encountered invalid file-extention on {structure_source}, valid extentions include [".xyz", ".cif"]')
        elif isinstance(structure_source, Atoms):
            try:
                elements = structure_source.get_chemical_symbols()
                size = len(elements)
                occupancy = torch.ones((size), dtype=self.dtype).to(device=self.device)
                xyz = torch.tensor(np.array(structure_source.get_positions())).to(device=self.device, dtype=self.dtype)
            except:
                raise ValueError(f'Encountered invalid Atoms object')
                
            triu_indices, unique_inverse, unique_form_factors, form_avg_sq, structure_inverse = parse_elements(elements, size)

            return StructureTuple(elements, size, occupancy, xyz, triu_indices, unique_inverse, unique_form_factors, form_avg_sq, structure_inverse)
        elif isinstance(structure_source, Structure):
            try:
                elements = [site.species_string for site in structure_source.sites]
                size = len(elements)
                occupancy = torch.ones((size), dtype=self.dtype).to(device=self.device)
                xyz = torch.from_numpy(np.array([[site.a, site.b, site.c] for site in structure_source.sites])).to(device=self.device, dtype=self.dtype)
            except:
                raise ValueError(f'Encountered invalid Structure object')
                
            triu_indices, unique_inverse, unique_form_factors, form_avg_sq, structure_inverse = parse_elements(elements, size)

            return StructureTuple(elements, size, occupancy, xyz, triu_indices, unique_inverse, unique_form_factors, form_avg_sq, structure_inverse)
        else:
            raise TypeError('Encountered unknown structure source')

    def generate_partial_masks(
        self,
        structure: StructureTuple,
        partial: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates masks for selecting specific atom pairs in a structure based on a given partial pattern.

        This function creates boolean masks that identify specific atom pairs within a structure, allowing for the calculation of partial scattering patterns. 
        The masks are generated based on a specified pattern indicating two elements (e.g., 'X-Y'), where 'X' and 'Y' represent different elements in the structure.
        The function supports both specified patterns and a default behavior when no pattern is provided.

        Parameters:
            structure (StructureTuple): A tuple representing the atomic structure, containing atomic positions, elements, etc.
            partial (str): A string in the form 'X-Y', where 'X' and 'Y' are element symbols. If provided, masks are created to isolate interactions between these elements. If None, masks select all elements.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - partial_mask_sparse (torch.Tensor): A boolean mask indicating the selected pairs of atoms in the upper triangular portion of the distance matrix.
                - partial_mask_struc (torch.Tensor): A boolean mask for selecting atoms of the first element 'X' in the structure.
                - partial_mask_other (torch.Tensor): A boolean mask for selecting atoms of the second element 'Y' in the structure.

        Raises:
            ValueError: If the partial pattern does not match the expected format 'X-Y', where 'X' and 'Y' are valid element symbols.
        """
            
        # Generate the upper trianfular indices
        N = structure.xyz.size(0)
        triu_indices = torch.triu_indices(N, N, offset=1)

        if partial is not None:

            # Assert that string matches represetation
            re_pattern = r'^([a-zA-Z]+)-([a-zA-Z]+)$'
            match = re.match(re_pattern, partial)
            if not match:
                raise ValueError("'partial' does not match the pattern 'X-Y', of elements 'X' and 'Y'.")

            # Extract elements and make sure they are atoms in the structure
            el1, el2 = match.groups()

            # Convert elements to a numpy array
            elements = np.array(structure.elements)

            if not el1 in elements:
                raise ValueError(f"element {el1} from 'partial' is not present in structure.")
            if not el2 in elements:
                raise ValueError(f"element {el2} from 'partial' is not present in structure.")


            # Construct masks for struc and other using comparison
            partial_mask_struc = torch.from_numpy(np.argwhere(elements == el1)).T.squeeze(0)
            partial_mask_other = torch.from_numpy(np.argwhere(elements == el2)).T.squeeze(0)

            # Construct boolean masks for the sparse represenation
            is_i_in_struc = torch.isin(triu_indices[0], partial_mask_struc)
            is_j_in_other = torch.isin(triu_indices[1], partial_mask_other)
            is_j_in_struc = torch.isin(triu_indices[1], partial_mask_struc)
            is_i_in_other = torch.isin(triu_indices[0], partial_mask_other)
            partial_mask_sparse = (is_i_in_struc & is_j_in_other) | (is_j_in_struc & is_i_in_other)
        else:
            partial_mask_struc = torch.ones((N,), dtype=torch.bool)
            partial_mask_other = torch.ones((N,), dtype=torch.bool)
            partial_mask_sparse = torch.ones((int((N*(N-1))/2),), dtype=torch.bool)

        return partial_mask_sparse, partial_mask_struc, partial_mask_other

    def iq(
        self,
        structure_source: StructureSourceType,
        radii: Union[List[float], float, None] = None,
        partial: str = None,
        keep_on_device: bool = False,
        include_self_scattering: bool = True,
    ) -> Union[IqTuple, List[IqTuple]]:
        """
        Calculate the scattering intensity I(Q) for the given atomic structure(s).

        Parameters:
            structure_source (StructureSourceType): Atomic structure source in XYZ/CIF format, ASE Atoms object, or as a tuple of (atomic_identities, atomic_positions).
            radii (Union[List[float], float, None]): List/float of radii/radius of particle(s) to generate with parsed CIF.
            partial (str): String on the form 'X-Y' where 'X' and 'Y' are elements in either structure. Used for calculating partial scattering patterns. Default is None.
            keep_on_device (bool): Flag to keep the results on the class device. Default is False, and will return numpy arrays on CPU.
            include_self_scattering (bool): Flag to compute self-scattering contribution. Default is True.

        Returns:
            Union[IqTuple, List[IqTuple]]: IqTuple containing Q-values and scattering intensity I(Q) or a list of such tuples.

        Raises:
            TypeError: If the structure source is of an invalid type.
            IOError: If there is an issue loading the structure from the specified file.
            ValueError: If the file extension is not valid or when providing .cif data file, radii is not provided.
        """
        def compute_iq(structure):

            # Calculate distances and batch
            if self.batch_size is None:
                self.batch_size = self._max_batch_size

            # Generate the upper trianfular indices
            N = structure.xyz.size(0)
            triu_indices = torch.triu_indices(N, N, offset=1)

            # Generate all distances, and batch
            dists = torch.norm(structure.xyz[:,None] - structure.xyz, dim=2, p=2)[triu_indices[0], triu_indices[1]].split(self.batch_size)

            # Generate partial masks
            partial_mask_sparse, partial_mask_struc, partial_mask_other = self.generate_partial_masks(structure, partial)

            # Batch mask and other indices
            partial_mask_sparse = partial_mask_sparse.to(device=self.device).split(self.batch_size)
            indices = structure.triu_indices.split(self.batch_size, dim=1)
            inverse_indices = structure.unique_inverse.split(self.batch_size, dim=1)

            if self.profile:
                self.profiler.time('Batching and Distances')

            # Calculate scattering using Debye Equation
            iq = torch.zeros((len(self.q))).to(device=self.device, dtype=self.dtype)
            for d, inv_idx, idx, partial_mask in zip(dists, inverse_indices, indices, partial_mask_sparse):
                
                # Construct mask
                threshold_mask = d >= self.rthres
                mask = threshold_mask & partial_mask

                # Calcualte scattering
                occ_product = structure.occupancy[idx[0]] * structure.occupancy[idx[1]]
                sinc = torch.sinc(d[mask] * self.q / torch.pi)
                ffp = structure.unique_form_factors[inv_idx[0]] * structure.unique_form_factors[inv_idx[1]]
                iq += torch.sum(occ_product.unsqueeze(-1)[mask] * ffp[mask] * sinc.permute(1,0), dim=0)

            # Apply Debye-Weller Isotropic Atomic Displacement
            if self.biso != 0.0:
                iq *= torch.exp(-self.q.squeeze(-1).pow(2) * self.biso/(8*torch.pi**2))
            
            # Self-scattering contribution
            if include_self_scattering:
                self_scattering_mask = torch.zeros((N,), dtype=bool)
                self_scattering_mask[partial_mask_struc] = True
                self_scattering_mask[partial_mask_other] = True

                sinc = torch.ones((structure.size, len(self.q))).to(device=self.device)[self_scattering_mask]
                iq += torch.sum((structure.occupancy[self_scattering_mask].unsqueeze(-1) * structure.unique_form_factors[structure.structure_inverse][self_scattering_mask])**2 * sinc, dim=0) / 2

            if self.profile:
                self.profiler.time('I(Q)')

            return iq
        
        if self.profile:
            self.profiler.reset()
        
        if not isinstance(structure_source, list):
            structure_source = [structure_source]

        structures = []
        for i, item in enumerate(structure_source):
            structure_output = self._initialise_structure(item, radii, disable_pbar = True)

            if isinstance(structure_output, list):
                structures.extend(structure_output)
            else:
                structures.append(structure_output)

        if self.profile:
            self.profiler.time('Setup structures and form factors')

        output = []
        for structure in structures:

            output_tuple = IqTuple(self.q.squeeze(-1), compute_iq(structure))
            if not keep_on_device:
                output_tuple = output_tuple._replace(
                    q = output_tuple.q.cpu().numpy(),
                    i = output_tuple.i.cpu().numpy()
                )
            output.append(output_tuple)

        return output if len(output) > 1 else output[0]

    def sq(
        self,
        structure_source: StructureSourceType,
        radii: Union[List[float], float, None] = None,
        partial: str = None,
        keep_on_device: bool = False,
    ) -> Union[SqTuple, List[SqTuple]]:
        """
        Calculate the structure function S(Q) for the given atomic structure(s)

        Parameters:
            structure_source (StructureSourceType): Atomic structure source in XYZ/CIF format, ASE Atoms object or as a tuple of (atomic_identities, atomic_positions)
            radii (Union[List[float], float, None]): List/float of radii/radius of particle(s) to generate with parsed CIF.
            partial (str): String on the form 'X-Y' where 'X' and 'Y' are elements in either structure. Used for calculating partial scattering patterns. Default is None.
            keep_on_device (bool): Flag to keep the results on the class device. Default is False, and will return numpy arrays on CPU

        Returns:
            SqTuple containing Q-values and structure function S(Q)
        """
        def compute_sq(structure):
            # Calculate distances and batch
            if self.batch_size is None:
                self.batch_size = self._max_batch_size

            # Generate the upper trianfular indices
            N = structure.xyz.size(0)
            triu_indices = torch.triu_indices(N, N, offset=1)

            # Generate all distances, and batch
            dists = torch.norm(structure.xyz[:,None] - structure.xyz, dim=2, p=2)[triu_indices[0], triu_indices[1]].split(self.batch_size)
            
            # Generate partial masks
            partial_mask_sparse, partial_mask_struc, partial_mask_other = self.generate_partial_masks(structure, partial)

            # Batch mask and other indices
            partial_mask_sparse = partial_mask_sparse.to(device=self.device).split(self.batch_size)
            indices = structure.triu_indices.split(self.batch_size, dim=1)
            inverse_indices = structure.unique_inverse.split(self.batch_size, dim=1)
            
            if self.profile:
                self.profiler.time('Batching and Distances')

            # Calculate scattering using Debye Equation
            iq = torch.zeros((len(self.q))).to(device=self.device, dtype=self.dtype)
            for d, inv_idx, idx, partial_mask in zip(dists, inverse_indices, indices, partial_mask_sparse):

                # Construct mask
                threshold_mask = d >= self.rthres
                mask = threshold_mask & partial_mask

                # Calculate scattering
                occ_product = structure.occupancy[idx[0]] * structure.occupancy[idx[1]]
                sinc = torch.sinc(d[mask] * self.q / torch.pi)
                ffp = structure.unique_form_factors[inv_idx[0]] * structure.unique_form_factors[inv_idx[1]]
                iq += torch.sum(occ_product.unsqueeze(-1)[mask] * ffp[mask] * sinc.permute(1,0), dim=0)

            # Apply Debye-Weller Isotropic Atomic Displacement
            if self.biso != 0.0:
                iq *= torch.exp(-self.q.squeeze(-1).pow(2) * self.biso/(8*torch.pi**2))
        
            # Calculate S(Q) and F(Q)
            sq = iq/structure.form_avg_sq/structure.size
            
            if self.profile:
                self.profiler.time('S(Q)')

            return sq
        
        if self.profile:
            self.profiler.reset()
        
        if not isinstance(structure_source, list):
            structure_source = [structure_source]

        structures = []
        for i, item in enumerate(structure_source):
            structure_output = self._initialise_structure(item, radii, disable_pbar = True)

            if isinstance(structure_output, list):
                structures.extend(structure_output)
            else:
                structures.append(structure_output)

        if self.profile:
            self.profiler.time('Setup structures and form factors')

        output = []
        for structure in structures:
            output_tuple = SqTuple(self.q.squeeze(-1), compute_sq(structure))
            if not keep_on_device:
                output_tuple = output_tuple._replace(
                    q = output_tuple.q.cpu().numpy(),
                    s = output_tuple.s.cpu().numpy()
                )
            output.append(output_tuple)

        return output if len(output) > 1 else output[0]

    def fq(
        self,
        structure_source: StructureSourceType,
        radii: Union[List[float], float, None] = None,
        partial: str = None,
        keep_on_device: bool = False,
    ) -> Union[FqTuple, List[FqTuple]]:
        """
        Calculate the structure function S(Q) for the given atomic structure(s).

        Parameters:
            structure_source (StructureSourceType): Atomic structure source in XYZ/CIF format, ASE Atoms object, or as a tuple of (atomic_identities, atomic_positions).
            radii (Union[List[float], float, None]): List/float of radii/radius of particle(s) to generate with parsed CIF.
            partial (str): String on the form 'X-Y' where 'X' and 'Y' are elements in either structure. Used for calculating partial scattering patterns. Default is None.
            keep_on_device (bool): Flag to keep the results on the class device. Default is False, and will return numpy arrays on CPU.

        Returns:
            Union[SqTuple, List[SqTuple]]: SqTuple containing Q-values and structure function S(Q) or a list of such tuples.

        Raises:
            TypeError: If the structure source is of an invalid type.
            IOError: If there is an issue loading the structure from the specified file.
            ValueError: If the file extension is not valid or when providing .cif data file, radii is not provided.
        """
        
        def compute_fq(structure):
            # Calculate distances and batch
            if self.batch_size is None:
                self.batch_size = self._max_batch_size

            # Generate the upper trianfular indices
            N = structure.xyz.size(0)
            triu_indices = torch.triu_indices(N, N, offset=1)

            # Generate all distances, and batch
            dists = torch.norm(structure.xyz[:,None] - structure.xyz, dim=2, p=2)[triu_indices[0], triu_indices[1]].split(self.batch_size)
            
            # Generate partial masks
            partial_mask_sparse, partial_mask_struc, partial_mask_other = self.generate_partial_masks(structure, partial)

            # Batch mask and other indices
            partial_mask_sparse = partial_mask_sparse.to(device=self.device).split(self.batch_size)
            indices = structure.triu_indices.split(self.batch_size, dim=1)
            inverse_indices = structure.unique_inverse.split(self.batch_size, dim=1)
            
            if self.profile:
                self.profiler.time('Batching and Distances')

            # Calculate scattering using Debye Equation
            iq = torch.zeros((len(self.q))).to(device=self.device, dtype=self.dtype)
            for d, inv_idx, idx, partial_mask in zip(dists, inverse_indices, indices, partial_mask_sparse):

                # Construct mask
                threshold_mask = d >= self.rthres
                mask = threshold_mask & partial_mask

                # Calculate scattering
                occ_product = structure.occupancy[idx[0]] * structure.occupancy[idx[1]]
                sinc = torch.sinc(d[mask] * self.q / torch.pi)
                ffp = structure.unique_form_factors[inv_idx[0]] * structure.unique_form_factors[inv_idx[1]]
                iq += torch.sum(occ_product.unsqueeze(-1)[mask] * ffp[mask] * sinc.permute(1,0), dim=0)

            # Apply Debye-Weller Isotropic Atomic Displacement
            if self.biso != 0.0:
                iq *= torch.exp(-self.q.squeeze(-1).pow(2) * self.biso/(8*torch.pi**2))
        
            # Calculate S(Q) and F(Q)
            sq = iq/structure.form_avg_sq/structure.size
            fq = self.q.squeeze(-1) * sq
            
            if self.profile:
                self.profiler.time('F(Q)')

            return fq
        
        if self.profile:
            self.profiler.reset()
        
        if not isinstance(structure_source, list):
            structure_source = [structure_source]

        structures = []
        for i, item in enumerate(structure_source):
            structure_output = self._initialise_structure(item, radii, disable_pbar = True)

            if isinstance(structure_output, list):
                structures.extend(structure_output)
            else:
                structures.append(structure_output)

        if self.profile:
            self.profiler.time('Setup structures and form factors')

        output = []
        for structure in structures:
            output_tuple = FqTuple(self.q.squeeze(-1), compute_fq(structure))
            if not keep_on_device:
                output_tuple = output_tuple._replace(
                    q = output_tuple.q.cpu().numpy(),
                    f = output_tuple.f.cpu().numpy()
                )
            output.append(output_tuple)

        return output if len(output) > 1 else output[0]

    def gr(
        self,
        structure_source: StructureSourceType,
        radii: Union[List[float], float, None] = None,
        partial: str = None,
        keep_on_device: bool = False,
    ) -> Union[GrTuple, List[GrTuple]]:
        """
        Calculate the reduced pair distribution function G(r) for the given atomic structure(s).

        Parameters:
            structure_source (StructureSourceType): Atomic structure source in XYZ/CIF format, ASE Atoms object, or as a tuple of (atomic_identities, atomic_positions).
            radii (Union[List[float], float, None]): List/float of radii/radius of particle(s) to generate with parsed CIF.
            partial (str): String on the form 'X-Y' where 'X' and 'Y' are elements in either structure. Used for calculating partial scattering patterns. Default is None.
            keep_on_device (bool): Flag to keep the results on the class device. Default is False, and will return numpy arrays on CPU.

        Returns:
            Union[GrTuple, List[GrTuple]]: GrTuple containing r-values and reduced pair distribution function G(r) or a list of such tuples.

        Raises:
            TypeError: If the structure source is of an invalid type.
            IOError: If there is an issue loading the structure from the specified file.
            ValueError: If the file extension is not valid or when providing .cif data file, radii is not provided.
        """
        def compute_gr(structure):
            # Calculate distances and batch
            if self.batch_size is None:
                self.batch_size = self._max_batch_size

            # Generate the upper trianfular indices
            N = structure.xyz.size(0)
            triu_indices = torch.triu_indices(N, N, offset=1)

            # Generate all distances, and batch
            dists = torch.norm(structure.xyz[:,None] - structure.xyz, dim=2, p=2)[triu_indices[0], triu_indices[1]].split(self.batch_size)
            
            # Generate partial masks
            partial_mask_sparse, partial_mask_struc, partial_mask_other = self.generate_partial_masks(structure, partial)

            # Batch mask and other indices
            partial_mask_sparse = partial_mask_sparse.to(device=self.device).split(self.batch_size)
            indices = structure.triu_indices.split(self.batch_size, dim=1)
            inverse_indices = structure.unique_inverse.split(self.batch_size, dim=1)
            
            if self.profile:
                self.profiler.time('Batching and Distances')

            # Calculate scattering using Debye Equation
            iq = torch.zeros((len(self.q))).to(device=self.device, dtype=self.dtype)
            for d, inv_idx, idx, partial_mask in zip(dists, inverse_indices, indices, partial_mask_sparse):

                # Construct mask
                threshold_mask = d >= self.rthres
                mask = threshold_mask & partial_mask

                # Calculate scattering
                occ_product = structure.occupancy[idx[0]] * structure.occupancy[idx[1]]
                sinc = torch.sinc(d[mask] * self.q / torch.pi)
                ffp = structure.unique_form_factors[inv_idx[0]] * structure.unique_form_factors[inv_idx[1]]
                iq += torch.sum(occ_product.unsqueeze(-1)[mask] * ffp[mask] * sinc.permute(1,0), dim=0)

            # Apply Debye-Weller Isotropic Atomic Displacement
            if self.biso != 0.0:
                iq *= torch.exp(-self.q.squeeze(-1).pow(2) * self.biso/(8*torch.pi**2))
        
            # Calculate S(Q), F(Q) and G(r)
            sq = iq/structure.form_avg_sq/structure.size
            fq = self.q.squeeze(-1) * sq

            damp = 1 if self.qdamp == 0.0 else torch.exp(-(self.r.squeeze(-1) * self.qdamp).pow(2) / 2)
            lorch_mod = 1 if self.lorch_mod == None else torch.sinc(self.q * self.lorch_mod*(torch.pi / self.qmax))
            gr = (2 / torch.pi) * torch.sum(fq.unsqueeze(-1) * torch.sin(self.q * self.r.permute(1,0))*self.qstep * lorch_mod, dim=0) * damp
            
            if self.profile:
                self.profiler.time('G(r)')
            
            return gr
        
        if self.profile:
            self.profiler.reset()
        
        if not isinstance(structure_source, list):
            structure_source = [structure_source]

        structures = []
        for i, item in enumerate(structure_source):
            structure_output = self._initialise_structure(item, radii, disable_pbar = True)

            if isinstance(structure_output, list):
                structures.extend(structure_output)
            else:
                structures.append(structure_output)

        if self.profile:
            self.profiler.time('Setup structures and form factors')

        output = []
        for structure in structures:
            output_tuple = GrTuple(self.r.squeeze(-1), compute_gr(structure))
            if not keep_on_device:
                output_tuple = output_tuple._replace(
                    r = output_tuple.r.cpu().numpy(),
                    g = output_tuple.g.cpu().numpy()
                )
            output.append(output_tuple)

        return output if len(output) > 1 else output[0]

    def _get_all(
        self,
        structure_source: StructureSourceType,
        radii: Union[List[float], float, None] = None,
        partial: str = None,
        keep_on_device: bool = False,
        include_self_scattering: bool = True,
    ) -> Union[AllTuple, List[AllTuple]]:
        """
        Calculate I(Q), S(Q), F(Q), and G(r) for the given atomic structure(s).

        Parameters:
            structure_source (StructureSourceType): Atomic structure source in XYZ/CIF format, ASE Atoms object, or as a tuple of (atomic_identities, atomic_positions).
            radii (Union[List[float], float, None]): List/float of radii/radius of particle(s) to generate with parsed CIF.
            partial (str): String on the form 'X-Y' where 'X' and 'Y' are elements in either structure. Used for calculating partial scattering patterns. Default is None.
            keep_on_device (bool): Flag to keep the results on the class device. Default is False, and will return numpy arrays on CPU.
            include_self_scattering (bool): Flag to compute self-scattering contribution. Default is True.

        Returns:
            Union[AllTuple, List[AllTuple]]: AllTuple containing r-values, Q-values, I(Q), S(Q), F(Q), and G(r) or a list of such tuples.

        Raises:
            TypeError: If the structure source is of an invalid type.
            IOError: If there is an issue loading the structure from the specified file.
            ValueError: If the file extension is not valid or when providing .cif data file, radii is not provided.
        """
        def compute_all(structure):
            # Calculate distances and batch
            if self.batch_size is None:
                self.batch_size = self._max_batch_size

            # Generate the upper trianfular indices
            N = structure.xyz.size(0)
            triu_indices = torch.triu_indices(N, N, offset=1)

            # Generate all distances, and batch
            dists = torch.norm(structure.xyz[:,None] - structure.xyz, dim=2, p=2)[triu_indices[0], triu_indices[1]].split(self.batch_size)
            
            # Generate partial masks
            partial_mask_sparse, partial_mask_struc, partial_mask_other = self.generate_partial_masks(structure, partial)

            # Batch mask and other indices
            partial_mask_sparse = partial_mask_sparse.to(device=self.device).split(self.batch_size)
            indices = structure.triu_indices.split(self.batch_size, dim=1)
            inverse_indices = structure.unique_inverse.split(self.batch_size, dim=1)
            
            if self.profile:
                self.profiler.time('Batching and Distances')

            # Calculate scattering using Debye Equation
            iq = torch.zeros((len(self.q))).to(device=self.device, dtype=self.dtype)
            for d, inv_idx, idx, partial_mask in zip(dists, inverse_indices, indices, partial_mask_sparse):

                # Construct mask
                threshold_mask = d >= self.rthres
                mask = threshold_mask & partial_mask

                # Calculate scattering
                occ_product = structure.occupancy[idx[0]] * structure.occupancy[idx[1]]
                sinc = torch.sinc(d[mask] * self.q / torch.pi)
                ffp = structure.unique_form_factors[inv_idx[0]] * structure.unique_form_factors[inv_idx[1]]
                iq += torch.sum(occ_product.unsqueeze(-1)[mask] * ffp[mask] * sinc.permute(1,0), dim=0)

            # Apply Debye-Weller Isotropic Atomic Displacement
            if self.biso != 0.0:
                iq *= torch.exp(-self.q.squeeze(-1).pow(2) * self.biso/(8*torch.pi**2))
        
            # Calculate S(Q), F(Q) and G(r)
            sq = iq/structure.form_avg_sq/structure.size
            fq = self.q.squeeze(-1) * sq

            damp = 1 if self.qdamp == 0.0 else torch.exp(-(self.r.squeeze(-1) * self.qdamp).pow(2) / 2)
            lorch_mod = 1 if self.lorch_mod == None else torch.sinc(self.q * self.lorch_mod*(torch.pi / self.qmax))
            gr = (2 / torch.pi) * torch.sum(fq.unsqueeze(-1) * torch.sin(self.q * self.r.permute(1,0))*self.qstep * lorch_mod, dim=0) * damp
            
            # Self-scattering contribution
            if include_self_scattering:
                self_scattering_mask = torch.zeros((N,), dtype=bool)
                self_scattering_mask[partial_mask_struc] = True
                self_scattering_mask[partial_mask_other] = True

                sinc = torch.ones((structure.size, len(self.q))).to(device=self.device)[self_scattering_mask]
                iq += torch.sum((structure.occupancy[self_scattering_mask].unsqueeze(-1) * structure.unique_form_factors[structure.structure_inverse][self_scattering_mask])**2 * sinc, dim=0) / 2
            
            if self.profile:
                self.profiler.time('All')

            return iq, sq, fq, gr
        
        if self.profile:
            self.profiler.reset()
        
        if not isinstance(structure_source, list):
            structure_source = [structure_source]

        structures = []
        for i, item in enumerate(structure_source):
            structure_output = self._initialise_structure(item, radii, disable_pbar = True)

            if isinstance(structure_output, list):
                structures.extend(structure_output)
            else:
                structures.append(structure_output)

        if self.profile:
            self.profiler.time('Setup structures and form factors')

        output = []
        for structure in structures:
            iq, sq, fq, gr = compute_all(structure)
            output_tuple = AllTuple(
                self.r.squeeze(-1),
                self.q.squeeze(-1),
                iq,
                sq,
                fq,
                gr
            )
            if not keep_on_device:
                output_tuple = output_tuple._replace(
                    r = output_tuple.r.cpu().numpy(),
                    q = output_tuple.q.cpu().numpy(),
                    i = output_tuple.i.cpu().numpy(),
                    s = output_tuple.s.cpu().numpy(),
                    f = output_tuple.f.cpu().numpy(),
                    g = output_tuple.g.cpu().numpy()
                )
            output.append(output_tuple)

        return output if len(output) > 1 else output[0]

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

        with importlib.resources.open_binary('debyecalculator.display_assets', 'choose_hardware.png') as f:
            choose_hardware_img = f.read()
        with importlib.resources.open_binary('debyecalculator.display_assets', 'batch_size.png') as f:
            batch_size_img = f.read()
        
        """ Utility widgets """

        # Spacing widget
        spacing_10px = widgets.Text(description='', layout=Layout(visibility='hidden', height='10px'), disabled=True)
        spacing_5px = widgets.Text(description='', layout=Layout(visibility='hidden', height='5px'), disabled=True)

        """ File Selection Tab """

        # Load examples
        load_examples = widgets.Button(description='Load Example Data')
        
        @load_examples.on_click
        def update_example_files(change):
            global global_file_widgets
            global upload
        
            file_widgets = [HBox([
                widgets.Text("Incl.", disabled=True, layout=Layout(width='7%')),
                widgets.Text("File Name", disabled=True, layout=Layout(width='70%')),
                widgets.Text("Particle Radius [Å]", disabled=True, layout=Layout(width='23%')),
            ], layout=Layout(justify_content='space-around'))]
            
            file_list = '<strong>Uploaded Files:</strong><br>'

            # CIF Example
            with importlib.resources.open_binary('debyecalculator.data', 'AntiFluorite_Co2O.cif') as f:
                example_dict = {}
                example_dict['content'] = f.read()
                example_dict['name'] = 'AntiFluorite_Co2O.cif'
                example_dict['type'] = ''
                example_dict['size'] = 1209
                example_dict['last_modified'] = datetime(2023, 8, 2, 13, 7, 38, 424000, tzinfo=timezone.utc)
                upload = widgets.FileUpload(value=(example_dict,), accept='.xyz,.cif', multiple=True)
                create_file_widgets(example_dict, file_widgets)

            # XYZ Example
            with importlib.resources.open_binary('debyecalculator.data', 'AntiFluorite_Co2O_r10.xyz') as f:
                example_dict = {}
                example_dict['content'] = f.read()
                example_dict['name'] = 'AntiFluorite_Co2O_r10.xyz'
                example_dict['type'] = ''
                example_dict['size'] = 1209
                example_dict['last_modified'] = datetime(2023, 8, 2, 13, 7, 38, 424000, tzinfo=timezone.utc)
                upload.value += (example_dict,)
                create_file_widgets(example_dict, file_widgets)

            upload_text.value = file_list

            global_file_widgets = file_widgets

            upload.observe(update_uploaded_files, names='value')
            
            clear_output(wait=True)
            display_tabs()

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
        
        # Upload files
        # Specify the allowed file extensions
        allowed_extensions = ['.cif', '.xyz']

        # Create FileUpload widget with the specified extensions
        global upload
        upload = widgets.FileUpload(accept=','.join(allowed_extensions), multiple=True)
        upload_text = widgets.HTML()

        # Dictionary to store uploaded files and associated float values
        radii_inputs = {}
        file_names = {}
        checkboxes = {}

        global global_file_widgets
        global_file_widgets = [HBox([
            widgets.Text("Incl.", disabled=True, layout=Layout(width='7%')),
            widgets.Text("File Name", disabled=True, layout=Layout(width='70%')),
            widgets.Text("Particle Radius [Å]", disabled=True, layout=Layout(width='23%')),
        ], layout=Layout(justify_content='space-around'))]

        # File selection Tab
        file_tab = VBox([
            VBox([HBox([upload, load_examples], layout=Layout(justify_content='space-around')), upload_text],
                 layout=Layout(justify_content='flex-start')),
            VBox(global_file_widgets),
        ], layout = file_tab_layout)
        
        """ Scattering Options Tab """

        # Load display_assets
        with importlib.resources.open_binary('debyecalculator.display_assets', 'qslider.png') as f:
            qslider_img = f.read()
        with importlib.resources.open_binary('debyecalculator.display_assets', 'rslider.png') as f:
            rslider_img = f.read()
        with importlib.resources.open_binary('debyecalculator.display_assets', 'qdamp.png') as f:
            qdamp_img = f.read()
        with importlib.resources.open_binary('debyecalculator.display_assets', 'global_biso.png') as f:
            global_biso_img = f.read()
        with importlib.resources.open_binary('debyecalculator.display_assets', 'a.png') as f:
            a_img = f.read()
        with importlib.resources.open_binary('debyecalculator.display_assets', 'a_inv.png') as f:
            a_inv_img = f.read()
        with importlib.resources.open_binary('debyecalculator.display_assets', 'a_sq.png') as f:
            a_sq_img = f.read()
        with importlib.resources.open_binary('debyecalculator.display_assets', 'qstep.png') as f:
            qstep_img = f.read()
        with importlib.resources.open_binary('debyecalculator.display_assets', 'rstep.png') as f:
            rstep_img = f.read()
        with importlib.resources.open_binary('debyecalculator.display_assets', 'rthres.png') as f:
            rthres_img = f.read()
        with importlib.resources.open_binary('debyecalculator.display_assets', 'radiation_type.png') as f:
            radiation_type_img = f.read()
        with importlib.resources.open_binary('debyecalculator.display_assets', 'scattering_parameters.png') as f:
            scattering_parameters_img = f.read()
        with importlib.resources.open_binary('debyecalculator.display_assets', 'presets.png') as f:
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
        with importlib.resources.open_binary('debyecalculator.display_assets', 'iq_scaling.png') as f:
            iq_scaling_img = f.read()
        with importlib.resources.open_binary('debyecalculator.display_assets', 'show_hide.png') as f:
            show_hide_img = f.read()
        with importlib.resources.open_binary('debyecalculator.display_assets', 'max_norm.png') as f:
            max_norm_img = f.read()
        with importlib.resources.open_binary('debyecalculator.display_assets', 'iq.png') as f:
            iq_img = f.read()
        with importlib.resources.open_binary('debyecalculator.display_assets', 'sq.png') as f:
            sq_img = f.read()
        with importlib.resources.open_binary('debyecalculator.display_assets', 'fq.png') as f:
            fq_img = f.read()
        with importlib.resources.open_binary('debyecalculator.display_assets', 'gr.png') as f:
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
        download_button = DownloadButton(zip_filename='DebyeCalculator_output.zip', description='Download data',
                                         layout=Layout(width='50%', height='90%'), button_style='success', icon='fa-download')
        
        # Tab index observer
        global selected_tab_idx
        selected_tab_idx = 0

        def on_tab_change(change):
            global selected_tab_idx
            selected_tab_idx = change['new']

        def display_tabs():
            global global_file_widgets
            global upload

            file_tab = VBox([
            VBox([HBox([upload, load_examples], layout=Layout(justify_content='space-around')), upload_text],
                 layout=Layout(justify_content='flex-start')),
            VBox(global_file_widgets),
        ], layout = file_tab_layout)

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
            
            tabs.selected_index = selected_tab_idx

            display(VBox([tabs, HBox([plot_button, download_button], layout=Layout(width='100%', height='50px'))]))
        
            tabs.observe(on_tab_change, names='selected_index')

        """ Observer utility """

        # Upload observer
        def create_file_widgets(file_info, file_widgets):

            file_name = file_info['name']

            # Add FloatText widget for ".cif" files
            is_cif = file_name.lower().endswith('.cif')
            checkboxes[file_name] = widgets.ToggleButton(
                value=True,
                description='',
                disabled=False,
                button_style='success',
                tooltip='',
                icon='check-square-o',
                layout=Layout(width='7%')
            )
            file_names[file_name] = widgets.Text(file_name, disabled=True, layout=Layout(width='70%'))
            radii_inputs[file_name] = widgets.FloatText(value=5.0, disabled=not is_cif, 
                                                layout=Layout(width='23%', visibility = 'hidden' if not is_cif else 'visible'))
            
            file_widgets.append(HBox([checkboxes[file_name], file_names[file_name], radii_inputs[file_name]],
                                layout=Layout(justify_content='space-around')))
            
            # Add an observer for each ToggleButton
            checkboxes[file_name].observe(partial(callback_toggle_button, file_name=file_name), names='value')

        def callback_toggle_button(change, file_name):
            if change['owner'].value:
                change['owner'].button_style = 'success'
                change['owner'].icon = 'check-square-o'
            else:
                change['owner'].button_style = ''
                change['owner'].icon = 'minus'

        def update_uploaded_files(change):
            global global_file_widgets
            global upload
        
            file_widgets = [HBox([
                widgets.Text("Incl.", disabled=True, layout=Layout(width='7%')),
                widgets.Text("File Name", disabled=True, layout=Layout(width='70%')),
                widgets.Text("Particle Radius [Å]", disabled=True, layout=Layout(width='23%')),
            ], layout=Layout(justify_content='space-around'))]
            
            num_files = len(upload.value)
            #upload.value = change['new']

            if num_files > 0:
                file_list = '<strong>Uploaded Files:</strong><br>'
                for file_info in upload.value:
                    create_file_widgets(file_info, file_widgets)

            upload_text.value = file_list

            global_file_widgets = file_widgets
            
            clear_output(wait=True)
            display_tabs()
                              
        upload.observe(update_uploaded_files, names='value')

        tabs.observe(on_tab_change, names='selected_index')

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
            qstep_box.value = 0.05
        
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
            
            labels = [out[0] for out in debye_outputs]
            for do in [out[1] for out in debye_outputs]:
                if show_iq_button.value:
                    axis_ids.append(0)
                    xseries.append(do[1]) # q
                    iq_ = do[2] if not normalize_iq.value else do[2]/max(do[2]) 
                    yseries.append(iq_) # iq
                    xlabels.append('$Q$ [$Å^{-1}$]')
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
                    xlabels.append('$Q$ [$Å^{-1}$]')
                    ylabels.append('$S(Q)$' + normalize_sq_text)
                    scales.append('linear')
                    titles.append('Structure Function, S(Q)')
                if show_fq_button.value:
                    axis_ids.append(2)
                    xseries.append(do[1]) # q
                    fq_ = do[4] if not normalize_fq.value else do[4]/max(do[4]) 
                    yseries.append(fq_) # fq
                    xlabels.append('$Q$ [$Å^{-1}$]')
                    ylabels.append('$F(Q)$'+ normalize_fq_text)
                    scales.append('linear')
                    titles.append('Reduced Structure Function, F(Q)')
                if show_gr_button.value:
                    axis_ids.append(3)
                    xseries.append(do[0]) # r
                    gr_ = do[5] if not normalize_gr.value else do[5]/max(do[5]) 
                    yseries.append(gr_) # gr
                    xlabels.append('$r$ [$Å$]')
                    ylabels.append('$G(r)$' + normalize_gr_text)
                    scales.append('linear')
                    titles.append('Reduced Pair Distribution Function, G(r)')

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

            fig.tight_layout()

        @plot_button.on_click
        def update_parameters(b=None):
            global debye_outputs
            global upload

            debye_outputs = []
            download_button.reset()

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

            # Loop thorugh all the files
            for i, upload_file in enumerate(upload.value):
                name = upload_file['name']
                suffix = name.split('.')[-1]
                if checkboxes[name].value:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.' + suffix, mode = 'w') as temp_file:
                        temp_content = BytesIO(upload_file['content']).read().decode()
                        temp_filename = temp_file.name
                        temp_file.write(temp_content)
                        temp_file.flush()
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
                            if radii_inputs[name].layout.visibility == 'visible' and radii_inputs[name].value > 8:
                                print(f'Generating nanoparticle of radius {radii_inputs[name].value} Å, using {name} ...')
                            if suffix == 'cif':
                                temp_structure = generate_nanoparticles(temp_filename, radii=radii_inputs[name].value, disable_pbar=True,
                                                                        _return_ase=True, device=hardware_button.value)[0].ase_structure
                                download_button.add_file_structure(ase_structure = temp_structure, filename='structure.xyz', subfolder=name)
                                all_tuple = debye_calc._get_all(temp_structure)
                            else:
                                all_tuple = debye_calc._get_all(temp_filename)

                            debye_outputs.append((name, all_tuple))

                            download_button.add_file_csv(x=all_tuple.q, y=all_tuple.i, filename='I(Q).csv', subfolder=name, metadata=metadata)
                            download_button.add_file_csv(x=all_tuple.q, y=all_tuple.s, filename='S(Q).csv', subfolder=name, metadata=metadata)
                            download_button.add_file_csv(x=all_tuple.q, y=all_tuple.f, filename='F(Q).csv', subfolder=name, metadata=metadata)
                            download_button.add_file_csv(x=all_tuple.r, y=all_tuple.g, filename='G(r).csv', subfolder=name, metadata=metadata)

                        except Exception as e:
                            print(f'FAILED: Could not load data file: {name}', end='\r')
                            raise e

            # Clear and display
            clear_output(wait=True)
            display_tabs()
            
            #print(debye_outputs)

            if len(debye_outputs) < 1:
                print('FAILED: Please select data file(s)', end="\r")
                return

            update_figure(debye_outputs)

        # Display tabs when function is called
        display_tabs()

class DownloadButton(widgets.Button):
    def __init__(self, zip_filename: str, **kwargs):
        super(DownloadButton, self).__init__(**kwargs)
        self.files = {}  # Dictionary to store filenames and contents by subfolder
        self.zip_filename = zip_filename
        self.on_click(self.__on_click)

    def reset(self):
         self.files = {}

    def add_file_csv(self, x, y, filename: str, subfolder: str, metadata: dict):
        output = ''
        content = "\n".join([",".join(map(str, np.around(row,len(str(metadata['qstep']))+5))) for row in np.stack([x, y]).T])
        for k,v in metadata.items():
            output += f'{k}:{v}\n'
        output += '\n'
        output += content

        if subfolder not in self.files:
            self.files[subfolder] = []
        self.files[subfolder].append((filename, output))

    def add_file_structure(self, ase_structure, filename: str, subfolder: str):
        # Get atomic properties
        positions = ase_structure.get_positions()
        elements = ase_structure.get_chemical_symbols()
        num_atoms = len(ase_structure)
        
        # Make header
        header = str(num_atoms) + "\n\n"
        
        # Join content 
        output = header + "\n".join([el + '\t' + "\t".join(map(str,np.around(row, 5))) for row, el in zip(positions, elements)])

        if subfolder not in self.files:
            self.files[subfolder] = []
        self.files[subfolder].append((filename, output))

    def __on_click(self, b):
        if not self.files:
            raise ValueError("Files must be added before clicking the button.")

        zip_buffer = BytesIO()

        with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
            for subfolder, file_list in self.files.items():
                for filename, content in file_list:
                    zip_file.writestr(f"{subfolder}/{filename}", content)

        zip_buffer.seek(0)
        zip_payload = base64.b64encode(zip_buffer.read()).decode()
        zip_digest = hashlib.md5(zip_buffer.getvalue()).hexdigest()  # bypass browser cache
        zip_id = f'dl_{zip_digest}'

        display(
            HTML(
                f"""
                    <html>
                    <body>
                    <a id="{zip_id}" download="{self.zip_filename}" href="data:application/zip;base64,{zip_payload}" download>
                    </a>

                    <script>
                    (function download() {{
                    document.getElementById('{zip_id}').click();
                    }})()
                    </script>

                    </body>
                    </html>
                """
            )
        )
