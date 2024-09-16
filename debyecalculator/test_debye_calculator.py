import pytest, torch
from debyecalculator import DebyeCalculator
from debyecalculator.utility.generate import generate_nanoparticles
import numpy as np
from ase.io import read
import importlib.resources
import yaml
import sys
import math

try:
    from pymatgen.core import Structure
    pymatgen_available = True
except ImportError:
    pymatgen_available = False

# Elements to atomic numbers map (fixture for reuse)
@pytest.fixture(scope="module")
def element_to_atomic_number():
    with importlib.resources.open_text('debyecalculator.utility', 'elements_info.yaml') as yaml_file:
        element_info = yaml.safe_load(yaml_file)
    return {key: value[12] for i, (key, value) in enumerate(element_info.items()) if i <= 97}

# Fixture to provide a fresh instance of DebyeCalculator
@pytest.fixture
def debye_calc():
    return DebyeCalculator()

# Helper function to normalize data
def normalize_counts(data, eps=1e-16):
    return data / (np.max(data) + eps)

@pytest.mark.parametrize("filename, qstep, gr_qstep, radii, dtype, expected_iq, expected_sq, expected_fq, expected_gr", [
    (
        'AntiFluorite_Co2O.cif', 
        0.05, 
        0.05,
        10.0,
        torch.float32,
        'iq_AntiFluorite_Co2O_radius10.0.dat', 
        'sq_AntiFluorite_Co2O_radius10.0.dat', 
        'fq_AntiFluorite_Co2O_radius10.0.dat', 
        'gr_AntiFluorite_Co2O_radius10.0.dat'
    ),
    
    (
        'AntiFluorite_Co2O.cif', 
        0.05, 
        0.05,
        10.0,
        torch.float64,
        'iq_AntiFluorite_Co2O_radius10.0.dat', 
        'sq_AntiFluorite_Co2O_radius10.0.dat', 
        'fq_AntiFluorite_Co2O_radius10.0.dat', 
        'gr_AntiFluorite_Co2O_radius10.0.dat'
    ),

    (
        'icsd_001504_cc_r6_lc_2.85_6_tetragonal.xyz', 
        0.1,
        0.1,
        None, 
        torch.float32,
        'icsd_001504_cc_r6_lc_2.85_6_tetragonal_Iq.dat', 
        'icsd_001504_cc_r6_lc_2.85_6_tetragonal_Sq.dat', 
        'icsd_001504_cc_r6_lc_2.85_6_tetragonal_Fq.dat',
        'icsd_001504_cc_r6_lc_2.85_6_tetragonal_Gr.dat'
    ),

    (
        'icsd_001504_cc_r6_lc_2.85_6_tetragonal.xyz', 
        0.1,
        0.1,
        None, 
        torch.float64,
        'icsd_001504_cc_r6_lc_2.85_6_tetragonal_Iq.dat', 
        'icsd_001504_cc_r6_lc_2.85_6_tetragonal_Sq.dat', 
        'icsd_001504_cc_r6_lc_2.85_6_tetragonal_Fq.dat',
        'icsd_001504_cc_r6_lc_2.85_6_tetragonal_Gr.dat'
    )
])
def test_scattering(debye_calc, filename, qstep, gr_qstep, radii, dtype, expected_iq, expected_sq, expected_fq, expected_gr):

    # Set dtype of calculator
    debye_calc.update_parameters(dtype=dtype)

    # Load the expected PDF
    expected_data = np.genfromtxt(f'debyecalculator/unittests_files/{expected_gr}', delimiter=',', skip_header=1)
    r_gr_expected, gr_expected = expected_data[:, 0], expected_data[:, 1]

    # Load the expected XRD
    expected_data = np.genfromtxt(f'debyecalculator/unittests_files/{expected_iq}', delimiter=',', skip_header=1)
    q_iq_expected, iq_expected = expected_data[:, 0], expected_data[:, 1]
    expected_data = np.genfromtxt(f'debyecalculator/unittests_files/{expected_sq}', delimiter=',', skip_header=1)
    q_sq_expected, sq_expected = expected_data[:, 0], expected_data[:, 1]
    expected_data = np.genfromtxt(f'debyecalculator/unittests_files/{expected_fq}', delimiter=',', skip_header=1)
    q_fq_expected, fq_expected = expected_data[:, 0], expected_data[:, 1]

    # Calculate the PDF
    debye_calc.update_parameters(qstep=gr_qstep)
    r_gr, gr = debye_calc.gr(f'debyecalculator/unittests_files/{filename}', radii=radii)

    # Calculate the XRD
    debye_calc.update_parameters(qstep=qstep)
    q_iq, iq = debye_calc.iq(f'debyecalculator/unittests_files/{filename}', radii=radii)
    q_sq, sq = debye_calc.sq(f'debyecalculator/unittests_files/{filename}', radii=radii)
    q_fq, fq = debye_calc.fq(f'debyecalculator/unittests_files/{filename}', radii=radii)

    # Normalize counts
    iq_expected = normalize_counts(iq_expected)
    iq = normalize_counts(iq)

    # Assertions
    assert np.allclose(q_iq, q_iq_expected, atol=1e-04, rtol=1e-03), f"Mismatch in Q for {expected_iq} when calculating for {filename}"
    assert np.allclose(iq, iq_expected, atol=1e-04, rtol=1e-03), f"Mismatch in I(Q) for {expected_iq} when calculating for {filename}"

    assert np.allclose(q_sq, q_sq_expected, atol=1e-04, rtol=1e-03), f"Mismatch in Q for {expected_sq} when calculating for {filename}"
    assert np.allclose(sq, sq_expected, atol=1e-04, rtol=1e-03), f"Mismatch in S(Q) for {expected_sq} when calculating for {filename}"

    assert np.allclose(q_fq, q_fq_expected, atol=1e-04, rtol=1e-03), f"Mismatch in Q for {expected_fq} when calculating for {filename}"
    assert np.allclose(fq, fq_expected, atol=1e-04, rtol=1e-03), f"Mismatch in F(Q) for {expected_fq} when calculating for {filename}"

    assert np.allclose(r_gr, r_gr_expected, atol=1e-04, rtol=1e-03), f"Mismatch in r for {expected_gr} when calculating for {filename}"
    assert np.allclose(gr, gr_expected, atol=1e-04, rtol=1e-03), f"Mismatch in G(r) for {expected_gr} when calculating for {filename}"

def test_invalid_input(debye_calc):
    # Test that DebyeCalculator raises a FileNotFoundError when given a non-existent file
    with pytest.raises(IOError):
        debye_calc.iq('non_existent_file.xyz')

    # Test invalid parameter updates (parametrized for different invalid values)
    invalid_params = [
        {'qmin': -1.0}, {'qmax': -1.0}, {'qstep': -1.0}, {'qdamp': -1.0},
        {'rmin': -1.0}, {'rmax': -1.0}, {'rstep': -1.0}, {'rthres': -1.0},
        {'biso': -1.0}, {'batch_size': -1}, {'device': 'x'}, {'radiation_type': 'x'},
        {'dtype': torch.float16}
    ]
    
    for param in invalid_params:
        with pytest.raises(ValueError):
            debye_calc.update_parameters(**param)

def test_generate_nanoparticle():
    # Load structure from xyz
    ase_structure = read('debyecalculator/unittests_files/structure_AntiFluorite_Co2O_radius10.0.xyz')
    xyz = ase_structure.get_positions()
    elements = ase_structure.get_chemical_symbols()

    # Generate xyz from utility function
    structure = generate_nanoparticles('debyecalculator/data/AntiFluorite_Co2O.cif', radii=10.0)[0]

    # Assertions
    assert np.allclose(xyz, structure.xyz.cpu(), atol=1e-04, rtol=1e-03), "Mismatch in generated xyz"
    assert elements == structure.elements, "Mismatch in generated elements"

@pytest.mark.parametrize("filename, qstep, gr_qstep", [
    ('icsd_001504_cc_r6_lc_2.85_6_tetragonal.xyz', 0.05, 0.1),
    ('structure_AntiFluorite_Co2O_radius10.0.xyz', 0.05, 0.05)
])
@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="Requires Python 3.10 or above"
)
def test_ase_and_pymatgen_input(debye_calc, filename, qstep, gr_qstep):
    # Load structure from ASE
    ase_structure = read(f'debyecalculator/unittests_files/{filename}')
    
    # Create Pymatgen structure from ASE
    cell = ase_structure.get_cell()
    positions = ase_structure.get_positions()
    elements = ase_structure.get_chemical_symbols()

    pymatgen_structure = Structure(cell, elements, positions)

    # I(Q)
    debye_calc.update_parameters(qstep=qstep)
    q_ase, iq_ase = debye_calc.iq(ase_structure)
    q_pmg, iq_pmg = debye_calc.iq(pymatgen_structure)

    # Normalize counts
    iq_ase = normalize_counts(iq_ase)
    iq_pmg = normalize_counts(iq_pmg)

    assert np.allclose(q_ase, q_pmg, atol=1e-04, rtol=1e-03), "Mismatch in Q between ASE and Pymatgen"
    assert np.allclose(iq_ase, iq_pmg, atol=1e-04, rtol=1e-03), "Mismatch in I(Q) between ASE and Pymatgen"

    # S(Q)
    q_ase, sq_ase = debye_calc.sq(ase_structure)
    q_pmg, sq_pmg = debye_calc.sq(pymatgen_structure)
    assert np.allclose(q_ase, q_pmg, atol=1e-04, rtol=1e-03), "Mismatch in Q between ASE and Pymatgen for S(Q)"
    assert np.allclose(sq_ase, sq_pmg, atol=1e-04, rtol=1e-03), "Mismatch in S(Q) between ASE and Pymatgen"

    # F(Q)
    q_ase, fq_ase = debye_calc.fq(ase_structure)
    q_pmg, fq_pmg = debye_calc.fq(pymatgen_structure)
    assert np.allclose(q_ase, q_pmg, atol=1e-04, rtol=1e-03), "Mismatch in Q between ASE and Pymatgen for F(Q)"
    assert np.allclose(fq_ase, fq_pmg, atol=1e-04, rtol=1e-03), "Mismatch in F(Q) between ASE and Pymatgen"

    # G(r)
    debye_calc.update_parameters(qstep=gr_qstep)
    r_ase, gr_ase = debye_calc.gr(ase_structure)
    r_pmg, gr_pmg = debye_calc.gr(pymatgen_structure)
    assert np.allclose(r_ase, r_pmg, atol=1e-04, rtol=1e-03), "Mismatch in r between ASE and Pymatgen"
    assert np.allclose(gr_ase, gr_pmg, atol=1e-04, rtol=1e-03), "Mismatch in G(r) between ASE and Pymatgen"

# Partials
def test_partials(debye_calc):
    """
    Testing whether the partials add up correctly for I(Q), S(Q), F(Q), and G(r)
    when include_self_scattering is taken into account.
    """
    calc = debye_calc
    cif_path = "debyecalculator/data/AntiFluorite_Co2O.cif"
    radii = 5

    # I(Q)
    _, iq = calc.iq(cif_path, radii=radii, partial=None)
    iq_parts = [calc.iq(cif_path, radii=radii, partial=p, include_self_scattering=is_self)[1]
                for p, is_self in [("O-O", True), ("Co-O", False), ("Co-Co", True)]]
    iq_sum = sum(iq_parts)
    assert np.allclose(iq, iq_sum, atol=1e-04, rtol=1e-03), "Partials are not matching for I(Q) calculations"

    # S(Q)
    _, sq = calc.sq(cif_path, radii=radii, partial=None)
    sq_parts = [calc.sq(cif_path, radii=radii, partial=p)[1]
                for p in ["O-O", "Co-O", "Co-Co"]]
    sq_sum = sum(sq_parts)
    assert np.allclose(sq, sq_sum, atol=1e-04, rtol=1e-03), "Partials are not matching for S(Q) calculations"

    # F(Q)
    _, fq = calc.fq(cif_path, radii=radii, partial=None)
    fq_parts = [calc.fq(cif_path, radii=radii, partial=p)[1]
                for p in ["O-O", "Co-O", "Co-Co"]]
    fq_sum = sum(fq_parts)
    assert np.allclose(fq, fq_sum, atol=1e-04, rtol=1e-03), "Partials are not matching for F(Q) calculations"

    # G(r)
    _, gr = calc.gr(cif_path, radii=radii, partial=None)
    gr_parts = [calc.gr(cif_path, radii=radii, partial=p)[1]
                for p in ["O-O", "Co-O", "Co-Co"]]
    gr_sum = sum(gr_parts)
    assert np.allclose(gr, gr_sum, atol=1e-04, rtol=1e-03), "Partials are not matching for G(r) calculations"

# Optimal qstep
def test_optimal_qstep(debye_calc):
    """
    Test that the calculated qstep is optimal based on the rmax and rstep values.
    """
    calc = debye_calc
    # Optimal qstep is π / (rmax + rstep), with a small tolerance for computational uncertainty
    optimal_qstep = math.pi / (calc.rmax + calc.rstep) + 1e-5
    assert calc.qstep <= optimal_qstep, f"Expected qstep <= {optimal_qstep}, but got qstep = {calc.qstep}"
