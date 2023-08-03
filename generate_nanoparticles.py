"""
generate_nanoparticles.py

This file contains a function to generate nanoparticle structures based on a given unit cell structure and a set of radii.

Function:
    generate_nanoparticle:
        Generates nanoparticle structures by replicating and manipulating a given unit cell.

Parameters:
    structure_path (str): Path to the file containing the unit cell structure in a format supported by ASE (Atomic Simulation Environment).
    radii (list): List of radii to generate nanoparticles with different sizes.
    sort_atoms (bool): Flag to sort atoms in the generated nanoparticle structures. Default is True.

Returns:
    tuple: Tuple containing two lists: 
        - The first list contains the generated nanoparticle structures as ASE Atom objects.
        - The second list contains the corresponding nanoparticle sizes (diameters) for each generated nanoparticle.

The `generate_nanoparticle` function reads the unit cell structure from the provided file and creates a supercell by replicating the unit cell based on the given radii. It then manipulates the positions of the atoms to center them within the supercell.

The function iteratively generates nanoparticle structures by including atoms within a certain distance from the center of the supercell. The size of each nanoparticle is determined by the largest distance between metal atoms (if present) within the specified radius.

The resulting nanoparticle structures are returned in sorted order (if `sort_atoms` is True) based on the atom type. The sorting is performed to ensure consistent orientation and to align the metal atoms towards the center of the nanoparticle.

Note: The ASE (Atomic Simulation Environment) library is used for reading and manipulating the atomic structures. Make sure to install ASE and its dependencies before using this function.

Author: Johansen & Anker et. al.
Date: August 2023
"""

import torch
import numpy as np
from tqdm.auto import tqdm
from torch import cdist
from ase.io import read
from ase.build import make_supercell
from ase.build.tools import sort as ase_sort

def generate_nanoparticle(
    self,
    structure_path,
    radii,
    sort_atoms = True,
):
    unit_cell = read(structure_path)
    cell_dims = np.array(unit_cell.cell.cellpar()[:3])
    r_max = np.amax(radii)
    P = np.diag((np.ceil(r_max / cell_dims)) * 2)
    cell = make_supercell(prim=unit_cell, P=P)

    positions = torch.from_numpy(cell.get_positions()).to(dtype=torch.float32, device=self.device)
    positions -= torch.mean(positions, dim=0)
    metal_filter = torch.BoolTensor([a not in self.ligand_list for a in cell.get_chemical_symbols()]).to(device=self.device)
    center_dists = torch.norm(positions, dim=1)
    positions -= positions[metal_filter][torch.argmin(center_dists[metal_filter])]
    center_dists = torch.norm(positions, dim=1)
    min_metal_dist = torch.min(pdist(positions[metal_filter]))
    cell.positions = positions.cpu()

    nano_particle_list = []
    nano_particle_sizes = []
    pbar = tqdm(desc='Generating nanoparticles', leave=False, total=len(radii))
    for r in radii:
        incl_mask = (center_dists <= r)
        interface_dists = cdist(positions, positions[incl_mask])
        nano_particle_size = 0
        for i in range(interface_dists.shape[0]):
            interface_mask = (interface_dists[i] <= (min_metal_dist)) & ~metal_filter[i]
            if torch.any(interface_mask):
                nano_particle_size = max(nano_particle_size, center_dists[i]*2)
                incl_mask[i] = True
        nano_particle_sizes.append(nano_particle_size)
        
        np_cell = cell[incl_mask.cpu()]
        if sort_atoms:
            np_cell = ase_sort(np_cell)
            if np_cell.get_chemical_symbols()[0] in self.ligand_list:
                np_cell = np_cell[::-1]
        nano_particle_list.append(np_cell)
        pbar.update(1)

    return nano_particle_list, nano_particle_sizes
