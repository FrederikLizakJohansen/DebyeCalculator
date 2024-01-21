# Handle import of torch (prerequisite)
try:
    import torch
    from torch import cdist
except ModuleNotFoundError:
    raise ImportError(
        "\n\nDebyeCalculator (and generate_nanoparticles) requires PyTorch, which is not installed. "
        "Please install PyTorch before using DebyeCalculator. "
        "Follow the instructions on the official PyTorch website: "
        "https://pytorch.org/get-started/locally/. "
        "For more information about DebyeCalculator, visit the GitHub repository: "
        "https://github.com/FrederikLizakJohansen/DebyeCalculator"
    )
import numpy as np
from ase.io import read
from ase.build import make_supercell
from ase.build.tools import sort as ase_sort
from typing import Union, List
from collections import namedtuple
import yaml
import pkg_resources
import warnings
from tqdm.auto import tqdm

NanoParticle = namedtuple('NanoParticle', 'elements size occupancy xyz')
NanoParticleASE = namedtuple('NanoParticleASE', 'ase_structure np_size')
NanoParticleASEGraph = namedtuple('NanoParticleASEGraph', 'ase_structure np_size edges distances')
NanoParticleType = Union[
    List[NanoParticle],
    NanoParticle,
    List[NanoParticleASE],
    NanoParticleASE,
    List[NanoParticleASEGraph],
    NanoParticleASEGraph
]

def get_default_atoms(
    atom_type: str,
    output_type: str = 'number'
) -> List[str]:
    """
    Get the default atoms based on the atom type and output type.

    Parameters:
    - atom_type (str): The type of atoms to retrieve. Accepts either "metal" or "ligand".
    - output_type (str): The type of output to retrieve. Accepts either "number" or "symbol". Defaults to "number".

    Returns:
    - atoms (list): The list of default atoms based on the atom type and output type.

    Raises:
    - ValueError: If the atom_type is not "metal" or "ligand".
    - ValueError: If the output_type is not "number" or "symbol".
    """
    METAL_SYMBOLS = [
        'Li', 'Be', 'B', 'Na', 'Mg', 'Al', 'Si', 'K',
        'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co',
        'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Rb', 'Sr',
        'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
        'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'Cs', 'Ba',
        'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
        'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf',
        'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        'Tl', 'Pb', 'Bi', 'Ra'
    ]

    LIGAND_SYMBOLS = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Se', 'Br', 'I']

    METAL_NUMBERS = [
        3, 4, 5, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 33, 37, 38, 39,
        40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
        52, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
        66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
        78, 79, 80, 81, 82, 83, 88
    ]

    LIGAND_NUMBERS = [1, 6, 7, 8, 9, 15, 16, 17, 34, 35, 53]

    if atom_type == 'metal':
        atoms = METAL_NUMBERS if output_type == 'number' else METAL_SYMBOLS
    elif atom_type == 'ligand':
        atoms = LIGAND_NUMBERS if output_type == 'number' else LIGAND_SYMBOLS
    else:
        raise ValueError(f'Invalid atom_type, accepts only "metal" or "ligand", got {atom_type}')

    if output_type not in ['number', 'symbol']:
        raise ValueError(f'Invalid output_type, accepts only "number" or "symbol", got {output_type}')

    return atoms
                    
def generate_nanoparticles(
    cif_file: str,
    radii: Union[List[float], float],
    metals: Union[List[float], List[str], str] = 'Default',
    ligands: Union[List[float], List[str], str] = 'Default', 
    sort_atoms: bool = True,
    disable_pbar: bool = False,
    return_graph_elements: bool = False,
    device: str = 'cuda',
    _override_device: bool = False,
    _lightweight_mode: bool = False,
    _return_ase: bool = False,
    _reverse_order: bool = True,
    _benchmarking: bool = False,
) -> NanoParticleType:
    """
    Generate spherical nanoparticles from a given CIF and radii.

    Args:
        cif_file (str): Input CIF file.
        radii (Union[List[float], float]): List of floats or float of radii for nanoparticles to be generated.
        metals (Union[List[float], List[str], str]): List of metals, their symbols, or 'Default' for default metal atoms.
        ligands (Union[List[float], List[str], str]): List of ligands, their symbols, or 'Default' for default ligand atoms.
        sort_atoms (bool, optional): Whether to sort atoms in the nanoparticle. Defaults to True.
        disable_pbar (bool, optional): Whether to disable the progress bar. Defaults to False.
        return_graph_elements (bool, optional): Whether to return graph elements. Defaults to False.
        device (str): Device to use for computations ('cuda' for CUDA-enabled GPU's or 'cpu' for CPU)
        _override_device (bool): Ignore object device and run on CPU.
        _lightweight_mode (bool): Whether to use lightweight mode. Defaults to False.
        _return_ase (bool): Whether to return ASE objects. Defaults to False.
        _reverse_order (bool): Whether to generate particles in reverse radii order
        _benchmarking (bool): Stripped down version for benchmarking

    Returns:
        NanoParticleType: List of nanoparticle tuples or ASE objects.
    """
        
    # Handling CUDA availability
    if _override_device:
        device = 'cpu'
    else:
        if device == 'cuda' and not torch.cuda.is_available():
            warnings.warn("Warning: Your system might have a CUDA-enabled GPU, but CUDA is not available. Computations will run on the CPU instead. " \
                          "For optimal performance, please install Pytorch with CUDA support. " \
                          "If you do not have a CUDA-enabled CPU, you can surpress this warning by specifying the 'device' argument as 'cpu'", stacklevel=2)
            device = 'cpu'
        elif device == 'cpu' and torch.cuda.is_available():
            warnings.warn("Warning: Your system has a CUDA-enabled GPU, but CPU was explicitly specified for computations. " \
                          "To utilise GPU acceleration, omit the 'device' argument or specify 'cuda'", stacklevel=2)
            device = 'cpu'
        else:
            device = device

    # Fetch atomic numbers and radii
    with open(pkg_resources.resource_filename(__name__, 'elements_info.yaml'), 'r') as yaml_file:
        elements_info = yaml.safe_load(yaml_file)

    # Fix radii type
    if isinstance(radii, list):
        radii = [float(r) for r in radii]
    elif isinstance(radii, float):
        radii = [radii]
    elif isinstance(radii, int):
        radii = [float(radii)]
    else:
        raise ValueError('FAILED: Please provide valid radii for generation of nanoparticles')

    # Handle metals and ligands
    if metals == 'Default':
        metals = get_default_atoms('metal', output_type='number')
    elif isinstance(metals, list):
        if isinstance(metals[0], str):
            try:
                metals = [elements_info(elm)[12] for elm in metals]
            except ImportError:
                raise ImportError('FAILED: Invalid element found')
    else:
        raise ValueError('FAILED: Please provide valid metals for generation of nanoparticles')
    
    if ligands == 'Default':
        ligands = get_default_atoms('ligand', output_type='number')
    elif isinstance(ligands, list):
        if isinstance(ligands[0], str):
            try:
                ligands = [elements_info(elm)[12] for elm in ligands]
            except ImportError:
                raise ImportError('FAILED: Invalid element found')

    # Read the input unit cell structure
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        unit_cell = read(cif_file)
    cell_dims = np.array(unit_cell.cell.cellpar()[:3])
    r_max = np.amax(radii)

    # Create a supercell to encompass the entire range of nanoparticles and center it
    size_check = np.array([False, False, False])
    padding = np.array([-2,-2,-2])
    while not all(size_check):
        padding[~size_check] += 2 # Symmetric padding to ensure the particle does not exceed the supercell boundary
        supercell_matrix = np.diag((np.ceil(r_max / cell_dims)) * 2 + padding)
        cell = make_supercell(prim=unit_cell, P=supercell_matrix)
        size_check = cell.get_positions().max(axis=0) >= (r_max * 2 + 5) # Check if the supercell is larger than diameter of largest particle + 5 Angstroms of padding
        
    cell.center(about=0.) # Center the supercell

    # Convert positions to torch and send to device
    positions = torch.from_numpy(cell.get_positions()).to(dtype = torch.float32, device = device)
    
    # Benchmarking
    if _benchmarking:
        nanoparticle_tuple_list = []
        for r in sorted(radii, reverse=_reverse_order):
            cell_norms = torch.norm(positions, p=2, dim=-1).cpu()
            np_cell = cell[cell_norms <= r]
            elements = np_cell.get_chemical_symbols()
            try:
                occupancy = np_cell.info['occupancy']
            except:
                occupancy = torch.ones((np_cell.get_global_number_of_atoms()), dtype=torch.float32)
            nanoparticle_tuple_list.append(
                NanoParticle(
                    elements = elements,
                    size = len(elements),
                    occupancy = occupancy,
                    xyz = torch.from_numpy(np_cell.get_positions()).to(device=device)
                )
            )
        return nanoparticle_tuple_list

    # Find atomic radii
    atomic_radii = torch.tensor(np.array([
        elements_info[elm][13]
        for elm in cell.get_chemical_symbols()
        ], dtype='float'), device=device)

    if _lightweight_mode:
        center_dists = torch.norm(positions, dim=1)
    else:
        # Find all metals and center around the nearest metal
        metal_filter = torch.BoolTensor([a in metals for a in cell.get_atomic_numbers()]).to(device = device)

        # Find the most central metal atom and center the cell around it
        center_dists = torch.norm(positions, dim=1)
        positions -= positions[metal_filter][torch.argmin(center_dists[metal_filter])]
        center_dists = torch.norm(positions, dim=1)

        # Update the cell positions
        cell.positions = positions.cpu()

    # Calculate distance matrix
    cell_dists = cdist(positions, positions)

    # Create mask of threshold for bonds
    bond_threshold = torch.zeros_like(cell_dists, device=device)
    for i, r1 in enumerate(atomic_radii):
        bond_threshold[i,:] = (r1 + atomic_radii) * 1.25
    bond_threshold.fill_diagonal_(0.)

    # Find edges
    direction = torch.argwhere(cell_dists < bond_threshold).T

    # Handle case with no edges
    if len(direction[0]) == 0:
        min_dist = torch.amin(cell_dists[cell_dists > 0])
        direction = torch.argwhere(cell_dists < min_dist * 1.1).T

    # Initialize nanoparticle lists and progress bar
    nanoparticle_tuple_list = []
    pbar = tqdm(desc=f'Generating nanoparticles in range: [{np.amin(radii)},{np.amax(radii)}]', leave=False, total=len(radii), disable=disable_pbar)

    # Generate nanoparticles for each radius
    for r in sorted(radii, reverse=_reverse_order):
        if _lightweight_mode:
            # Mask all atoms within radius
            incl_mask = (center_dists <= r)
            
            # Get indices of atoms to be included
            incl_indices = torch.nonzero(incl_mask).flatten()

            # Get edges to be included
            included_edges = direction[:,~(torch.isin(direction[0], ~incl_indices) + torch.isin(direction[1], ~incl_indices))]
            
            # Get included atoms
            included_atoms = included_edges.unique()
            
        else:
            # Mask all metal atoms outside of the radius
            excl_mask = (center_dists > r) & metal_filter
            # Mask all metal atoms within the radius
            incl_mask = (center_dists <= r) & metal_filter

            # Get indices of atoms to be included and excluded
            excl_indices = torch.nonzero(excl_mask).flatten()
            incl_indices = torch.nonzero(incl_mask).flatten()

            # Get edges to be included
            included_edges = direction[:,(torch.isin(direction[0], incl_indices) + torch.isin(direction[1], incl_indices))]

            # Remove edges to be excluded
            included_edges = included_edges[:,~(torch.isin(included_edges[0], excl_indices) + torch.isin(included_edges[1], excl_indices))]
            
            # Get included atoms
            included_atoms = included_edges.unique()

        # Get Atoms object for the NP
        np_cell = cell[included_atoms.cpu()]

        # Remove NPs with only one atom
        if len(np_cell) == 1:
            pbar.update(1)
            continue

        # Determine NP size
        nanoparticle_size = torch.amax(center_dists[included_atoms.cpu()]) * 2
        
        # Sort the atoms
        if sort_atoms:
            np_cell = ase_sort(np_cell)
            if np_cell.get_chemical_symbols()[0] in ligands:
                np_cell = np_cell[::-1]
        
        # Get occupancy (if any)
        try:
            occupancy = np_cell.info['occupancy']
        except:
            occupancy = torch.ones((np_cell.get_global_number_of_atoms()), dtype=torch.float32)

        # Append nanoparticle
        if not _return_ase:
            elements = np_cell.get_chemical_symbols()
            nanoparticle_tuple_list.append(
                NanoParticle(
                    elements = elements,
                    size = len(elements),
                    occupancy = occupancy,
                    xyz = torch.from_numpy(np_cell.get_positions()).to(device=device)
                )
            )
        else:
            if return_graph_elements:
                if sort_atoms:
                    raise NotImplementedError('FAILED: return_graph_elements is not yet implemented for sorted atoms')
        
                # Get included distances
                np_dists = cell_dists[included_edges[0], included_edges[1]]

                # Reorganise the included edges
                reorganised_edges = transform_edge_indices(included_edges)

                nanoparticle_tuple_list.append(
                    NanoParticleASEGraph(
                        ase_structure = np_cell,
                        np_size = nanoparticle_size.item(),
                        edges = reorganised_edges,
                        distances = np_dists
                    )
                )
            else:
                nanoparticle_tuple_list.append(
                    NanoParticleASE(
                        ase_structure = np_cell,
                        np_size = nanoparticle_size.item()
                    )
                )

        pbar.update(1)
    pbar.close()
    return nanoparticle_tuple_list

def transform_edge_indices(edge_indices):

    # Extract unique nodes from the edge indices
    edge_indices = edge_indices.T
    unique_nodes = torch.unique(edge_indices)

    # Create a mapping from old indices to new indices
    node_mapping = {old_index.item(): new_index for new_index, old_index in enumerate(unique_nodes)}

    # Transform edge indices using the mapping
    transformed_edges = torch.tensor([[node_mapping[edge[0].item()], node_mapping[edge[1].item()]] for edge in edge_indices])

    return transformed_edges.T
