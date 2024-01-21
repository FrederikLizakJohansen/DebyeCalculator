import pytest, torch
from debyecalculator import DebyeCalculator
from debyecalculator.utility.generate import generate_nanoparticles
import numpy as np
from ase.io import read
import pkg_resources
import yaml

# Elements to atomic numbers map
with open(pkg_resources.resource_filename(__name__, 'utility/elements_info.yaml'), 'r') as yaml_file:
    element_info = yaml.safe_load(yaml_file)
element_to_atomic_number = {}
for i, (key, value) in enumerate(element_info.items()):
    if i > 97:
        break
    element_to_atomic_number[key] = value[12]

# Global calculator object
calc = DebyeCalculator()

def test_init_defaults():
    #calc = DebyeCalculator()
    assert calc.qmin == 1.0, f"Expected qmin to be 1.0, but got {calc.qmin}"
    assert calc.qmax == 30.0, f"Expected qmax to be 30.0, but got {calc.qmax}"
    assert calc.qstep == 0.05, f"Expected qstep to be 0.05, but got {calc.qstep}"
    assert calc.qdamp == 0.04, f"Expected qdamp to be 0.04, but got {calc.qdamp}"
    assert calc.rmin == 0.0, f"Expected rmin to be 0.0, but got {calc.rmin}"
    assert calc.rmax == 20.0, f"Expected rmax to be 20.0, but got {calc.rmax}"
    assert calc.rstep == 0.01, f"Expected rstep to be 0.01, but got {calc.rstep}"
    assert calc.rthres == 0.0, f"Expected rthres to be 0.0, but got {calc.rthres}"
    assert calc.biso == 0.3, f"Expected biso to be 0.3, but got {calc.biso}"
    assert calc.device == 'cuda' if torch.cuda.is_available() else 'cpu', f"Expected device to be 'cuda' or 'cpu', but got {calc.device}"
    assert calc.batch_size == 10000, f"Expected batch_size to be 10000, but got {calc.batch_size}"
    assert calc.lorch_mod == False, f"Expected lorch_mod to be False, but got {calc.lorch_mod}"
    assert calc.radiation_type == 'xray', f"Expected radiation_type to be 'xray', but got {calc.radiation_type}"
    assert calc.profile == False, f"Expected profile to be False, but got {calc.profile}"

def test_iq_xyz():
    # Load the expected scattering intensity from a file
    ph = np.loadtxt('debyecalculator/unittests_files/icsd_001504_cc_r6_lc_2.85_6_tetragonal_Iq.dat')
    q_expected, iq_expected = ph[:,0], ph[:,1]

    # Calculate the scattering intensity using the DebyeCalculator
    calc.update_parameters(qstep=0.1)
    q, iq = calc.iq('debyecalculator/unittests_files/icsd_001504_cc_r6_lc_2.85_6_tetragonal.xyz')

    # Check that the calculated scattering intensity matches the expected value
    assert np.allclose(q, q_expected, atol=1e-04, rtol=1e-03), f"Expected Q to be {q_expected}, but got {q}"
    assert np.allclose(iq, iq_expected, atol=1e-04, rtol=1e-03), f"Expected I(Q) to be {iq_expected}, but got {iq}"

def test_iq_cif():

    # Load expected scattering intensity generated from cif
    ph = np.genfromtxt('debyecalculator/unittests_files/iq_AntiFluorite_Co2O_radius10.0.dat', delimiter=',', skip_header=15)
    q_expected, iq_expected = ph[:,0], ph[:,1]

    # Calculate the scatering intensity using DebyeCalculator
    calc.update_parameters(qstep=0.05)
    q, iq = calc.iq('data/AntiFluorite_Co2O.cif', radii=10.0)
    
    # Check that the calculated scattering intensity matches the expected value
    assert np.allclose(q, q_expected, atol=1e-04, rtol=1e-03), f"Expected Q to be {q_expected}, but got {q}"
    assert np.allclose(iq, iq_expected, atol=1e-04, rtol=1e-03), f"Expected I(Q) to be {iq_expected}, but got {iq}"

def test_iq_tuple():

    # Load expected scattering intensity
    ph = np.genfromtxt('debyecalculator/unittests_files/iq_AntiFluorite_Co2O_radius10.0.dat', delimiter=',', skip_header=15)
    q_expected, iq_expected = ph[:,0], ph[:,1]

    # calculate the scattering intensity using tuple
    ase_structure = read('debyecalculator/unittests_files/structure_AntiFluorite_Co2O_radius10.0.xyz')
    ase_elements = ase_structure.get_chemical_symbols()
    ase_xyz = ase_structure.get_positions()

    structure_tuple_str = (ase_elements, ase_xyz)
    structure_tuple_int = (np.array([element_to_atomic_number[e] for e in ase_elements]), ase_xyz)

    calc.update_parameters(qstep=0.05)
    q_str, iq_str = calc.iq(structure_tuple_str)
    q_int, iq_int = calc.iq(structure_tuple_int)

    # Check that the calculated scattering intensity matches the expected value
    assert np.allclose(q_str, q_expected, atol=1e-04, rtol=1e-03), f"Expected Q to be {q_expected}, but got {q_str}"
    assert np.allclose(q_int, q_expected, atol=1e-04, rtol=1e-03), f"Expected Q to be {q_expected}, but got {q_int}"
    assert np.allclose(iq_str, iq_expected, atol=1e-04, rtol=1e-03), f"Expected I(Q) to be {iq_expected}, but got {iq_str}"
    assert np.allclose(iq_int, iq_expected, atol=1e-04, rtol=1e-03), f"Expected I(Q) to be {iq_expected}, but got {iq_int}"

def test_sq_xyz():
    # Load the expected structure factor from a file
    ph = np.loadtxt('debyecalculator/unittests_files/icsd_001504_cc_r6_lc_2.85_6_tetragonal_Sq.dat')
    q_expected, sq_expected = ph[:,0], ph[:,1]

    # Calculate the structure factor using the DebyeCalculator
    calc.update_parameters(qstep=0.1)
    q, sq = calc.sq('debyecalculator/unittests_files/icsd_001504_cc_r6_lc_2.85_6_tetragonal.xyz')

    # Check that the calculated structure factor matches the expected value
    assert np.allclose(q, q_expected, atol=1e-04, rtol=1e-03), f"Expected Q to be {q_expected}, but got {q}"
    assert np.allclose(sq, sq_expected, atol=1e-04, rtol=1e-03), f"Expected S(Q) to be {sq_expected}, but got {sq}"

def test_sq_cif():
    # Load the expected structure factor from a file
    ph = np.genfromtxt('debyecalculator/unittests_files/sq_AntiFluorite_Co2O_radius10.0.dat', delimiter=',', skip_header=15)
    q_expected, sq_expected = ph[:,0], ph[:,1]

    # Calculate the structure factor using the DebyeCalculator
    calc.update_parameters(qstep=0.05)
    q, sq = calc.sq('data/AntiFluorite_Co2O.cif', radii=10.0)

    # Check that the calculated structure factor matches the expected value
    assert np.allclose(q, q_expected, atol=1e-04, rtol=1e-03), f"Expected Q to be {q_expected}, but got {q}"
    assert np.allclose(sq, sq_expected, atol=1e-04, rtol=1e-03), f"Expected S(Q) to be {sq_expected}, but got {sq}"

def test_sq_tuple():

    ph = np.genfromtxt('debyecalculator/unittests_files/sq_AntiFluorite_Co2O_radius10.0.dat', delimiter=',', skip_header=15)
    q_expected, sq_expected = ph[:,0], ph[:,1]

    ase_structure = read('debyecalculator/unittests_files/structure_AntiFluorite_Co2O_radius10.0.xyz')
    ase_elements = ase_structure.get_chemical_symbols()
    ase_xyz = ase_structure.get_positions()

    structure_tuple_str = (ase_elements, ase_xyz)
    structure_tuple_int = (np.array([element_to_atomic_number[e] for e in ase_elements]), ase_xyz)

    calc.update_parameters(qstep=0.05)
    q_str, sq_str = calc.sq(structure_tuple_str)
    q_int, sq_int = calc.sq(structure_tuple_int)

    assert np.allclose(q_str, q_expected, atol=1e-04, rtol=1e-03), f"Expected Q to be {q_expected}, but got {q_str}"
    assert np.allclose(q_int, q_expected, atol=1e-04, rtol=1e-03), f"Expected Q to be {q_expected}, but got {q_int}"
    assert np.allclose(sq_str, sq_expected, atol=1e-04, rtol=1e-03), f"Expected S(Q) to be {sq_expected}, but got {sq_str}"
    assert np.allclose(sq_int, sq_expected, atol=1e-04, rtol=1e-03), f"Expected S(Q) to be {sq_expected}, but got {sq_int}"

def test_fq_xyz():
    # Load the expected atomic form factor from a file
    ph = np.loadtxt('debyecalculator/unittests_files/icsd_001504_cc_r6_lc_2.85_6_tetragonal_Fq.dat')
    q_expected, fq_expected = ph[:,0], ph[:,1]

    # Calculate the atomic form factor using the DebyeCalculator
    calc.update_parameters(qstep=0.1)
    q, fq = calc.fq('debyecalculator/unittests_files/icsd_001504_cc_r6_lc_2.85_6_tetragonal.xyz')

    # Check that the calculated atomic form factor matches the expected value
    assert np.allclose(q, q_expected, atol=1e-04, rtol=1e-03), f"Expected Q to be {q_expected}, but got {q}"
    assert np.allclose(fq, fq_expected, atol=1e-04, rtol=1e-03), f"Expected F(Q) to be {fq_expected}, but got {fq}"

def test_fq_cif():
    # Load the expected structure factor from a file
    ph = np.genfromtxt('debyecalculator/unittests_files/fq_AntiFluorite_Co2O_radius10.0.dat', delimiter=',', skip_header=15)
    q_expected, fq_expected = ph[:,0], ph[:,1]

    # Calculate the structure factor using the DebyeCalculator
    calc.update_parameters(qstep=0.05)
    q, fq = calc.fq('data/AntiFluorite_Co2O.cif', radii=10.0)

    # Check that the calculated structure factor matches the expected value
    assert np.allclose(q, q_expected, atol=1e-04, rtol=1e-03), f"Expected Q to be {q_expected}, but got {q}"
    assert np.allclose(fq, fq_expected, atol=1e-04, rtol=1e-03), f"Expected F(Q) to be {fq_expected}, but got {fq}"

def test_fq_tuple():

    ph = np.genfromtxt('debyecalculator/unittests_files/fq_AntiFluorite_Co2O_radius10.0.dat', delimiter=',', skip_header=15)
    q_expected, fq_expected = ph[:,0], ph[:,1]

    ase_structure = read('debyecalculator/unittests_files/structure_AntiFluorite_Co2O_radius10.0.xyz')
    ase_elements = ase_structure.get_chemical_symbols()
    ase_xyz = ase_structure.get_positions()

    structure_tuple_str = (ase_elements, ase_xyz)
    structure_tuple_int = (np.array([element_to_atomic_number[e] for e in ase_elements]), ase_xyz)

    calc.update_parameters(qstep=0.05)
    q_str, fq_str = calc.fq(structure_tuple_str)
    q_int, fq_int = calc.fq(structure_tuple_int)

    assert np.allclose(q_str, q_expected, atol=1e-04, rtol=1e-03), f"Expected Q to be {q_expected}, but got {q_str}"
    assert np.allclose(q_int, q_expected, atol=1e-04, rtol=1e-03), f"Expected Q to be {q_expected}, but got {q_int}"
    assert np.allclose(fq_str, fq_expected, atol=1e-04, rtol=1e-03), f"Expected F(Q) to be {fq_expected}, but got {fq_str}"
    assert np.allclose(fq_int, fq_expected, atol=1e-04, rtol=1e-03), f"Expected F(Q) to be {fq_expected}, but got {fq_int}"

def test_gr_xyz():
    # Load the expected radial distribution function from a file
    ph = np.loadtxt('debyecalculator/unittests_files/icsd_001504_cc_r6_lc_2.85_6_tetragonal_Gr.dat')
    r_expected, gr_expected = ph[:,0], ph[:,1]

    # Calculate the radial distribution function using the DebyeCalculator
    calc.update_parameters(qstep=0.1)
    r, gr = calc.gr('debyecalculator/unittests_files/icsd_001504_cc_r6_lc_2.85_6_tetragonal.xyz')

    # Check that the calculated radial distribution function matches the expected value
    assert np.allclose(r, r_expected, atol=1e-04, rtol=1e-03), f"Expected r to be {r_expected}, but got {r}"
    assert np.allclose(gr, gr_expected, atol=1e-04, rtol=1e-03), f"Expected G(r) to be {gr_expected}, but got {gr}"

def test_gr_cif():
    # Load the expected structure factor from a file
    ph = np.genfromtxt('debyecalculator/unittests_files/gr_AntiFluorite_Co2O_radius10.0.dat', delimiter=',', skip_header=15)
    r_expected, gr_expected = ph[:,0], ph[:,1]

    # Calculate the structure factor using the DebyeCalculator
    calc.update_parameters(qstep=0.05)
    r, gr = calc.gr('data/AntiFluorite_Co2O.cif', radii=10.0)

    # Check that the calculated structure factor matches the expected value
    assert np.allclose(r, r_expected, atol=1e-04, rtol=1e-03), f"Expected r to be {r_expected}, but got {r}"
    assert np.allclose(gr, gr_expected, atol=1e-04, rtol=1e-03), f"Expected G(r) to be {gr_expected}, but got {gr}"

def test_gr_tuple():

    ph = np.genfromtxt('debyecalculator/unittests_files/gr_AntiFluorite_Co2O_radius10.0.dat', delimiter=',', skip_header=15)
    r_expected, gr_expected = ph[:,0], ph[:,1]

    ase_structure = read('debyecalculator/unittests_files/structure_AntiFluorite_Co2O_radius10.0.xyz')
    ase_elements = ase_structure.get_chemical_symbols()
    ase_xyz = ase_structure.get_positions()

    structure_tuple_str = (ase_elements, ase_xyz)
    structure_tuple_int = (np.array([element_to_atomic_number[e] for e in ase_elements]), ase_xyz)

    calc.update_parameters(qstep=0.05)
    r_str, gr_str = calc.gr(structure_tuple_str)
    r_int, gr_int = calc.gr(structure_tuple_int)

    assert np.allclose(r_str, r_expected, atol=1e-04, rtol=1e-03), f"Expected r to be {r_expected}, but got {r_str}"
    assert np.allclose(r_int, r_expected, atol=1e-04, rtol=1e-03), f"Expected r to be {r_expected}, but got {r_int}"
    assert np.allclose(gr_str, gr_expected, atol=1e-04, rtol=1e-03), f"Expected G(r) to be {gr_expected}, but got {gr_str}"
    assert np.allclose(gr_int, gr_expected, atol=1e-04, rtol=1e-03), f"Expected G(r) to be {gr_expected}, but got {gr_int}"

def test_get_all_xyz():
    # Calculate Iq, Fq, Sq, and Gr using the DebyeCalculator
    calc.update_parameters(qstep=0.1)
    r, q, iq, sq, fq, gr = calc._get_all('debyecalculator/unittests_files/icsd_001504_cc_r6_lc_2.85_6_tetragonal.xyz')

    # Check that the calculated Iq matches the expected value
    ph = np.loadtxt('debyecalculator/unittests_files/icsd_001504_cc_r6_lc_2.85_6_tetragonal_Iq.dat')
    q_expected, iq_expected = ph[:,0], ph[:,1]
    assert np.allclose(q, q_expected, atol=1e-04, rtol=1e-03), f"Expected Q to be {q_expected}, but got {q}"
    assert np.allclose(iq, iq_expected, atol=1e-04, rtol=1e-03), f"Expected I(Q) to be {iq_expected}, but got {iq}"

    # Check that the calculated Sq matches the expected value
    ph = np.loadtxt('debyecalculator/unittests_files/icsd_001504_cc_r6_lc_2.85_6_tetragonal_Sq.dat')
    q_expected, sq_expected = ph[:,0], ph[:,1]
    assert np.allclose(q, q_expected, rtol=1e-03), f"Expected Q to be {q_expected}, but got {q}"
    assert np.allclose(sq, sq_expected, rtol=1e-03), f"Expected S(Q) to be {sq_expected}, but got {sq}"

    # Check that the calculated Fq matches the expected value
    ph = np.loadtxt('debyecalculator/unittests_files/icsd_001504_cc_r6_lc_2.85_6_tetragonal_Fq.dat')
    q_expected, fq_expected = ph[:,0], ph[:,1]
    assert np.allclose(q, q_expected, atol=1e-04, rtol=1e-03), f"Expected Q to be {q_expected}, but got {q}"
    assert np.allclose(fq, fq_expected, atol=1e-04, rtol=1e-03), f"Expected F(Q) to be {fq_expected}, but got {fq}"

    # Check that the calculated Gr matches the expected value
    ph = np.loadtxt('debyecalculator/unittests_files/icsd_001504_cc_r6_lc_2.85_6_tetragonal_Gr.dat')
    r_expected, gr_expected = ph[:,0], ph[:,1]
    assert np.allclose(r, r_expected, atol=1e-04, rtol=1e-03), f"Expected r to be {r_expected}, but got {r}"
    assert np.allclose(gr, gr_expected, atol=1e-04, rtol=1e-03), f"Expected G(r) to be {gr_expected}, but got {gr}"

def test_get_all_cif():

    # Calculate Iq, Fq, Sq, and Gr using the DebyeCalculator
    calc.update_parameters(qstep=0.05)
    r, q, iq, sq, fq, gr = calc._get_all('data/AntiFluorite_Co2O.cif', radii=10.0)

    # Check that the calculated Iq matches the expected value
    ph = np.genfromtxt('debyecalculator/unittests_files/iq_AntiFluorite_Co2O_radius10.0.dat', delimiter=',', skip_header=15)
    q_expected, iq_expected = ph[:,0], ph[:,1]
    assert np.allclose(q, q_expected, atol=1e-04, rtol=1e-03), f"Expected Q to be {q_expected}, but got {q}"
    assert np.allclose(iq, iq_expected, atol=1e-04, rtol=1e-03), f"Expected I(Q) to be {iq_expected}, but got {iq}"

    # Check that the calculated Sq matches the expected value
    ph = np.genfromtxt('debyecalculator/unittests_files/sq_AntiFluorite_Co2O_radius10.0.dat', delimiter=',', skip_header=15)
    q_expected, sq_expected = ph[:,0], ph[:,1]
    assert np.allclose(q, q_expected, atol=1e-04, rtol=1e-03), f"Expected Q to be {q_expected}, but got {q}"
    assert np.allclose(sq, sq_expected, atol=1e-04, rtol=1e-03), f"Expected S(Q) to be {sq_expected}, but got {sq}"

    # Check that the calculated Fq matches the expected value
    ph = np.genfromtxt('debyecalculator/unittests_files/fq_AntiFluorite_Co2O_radius10.0.dat', delimiter=',', skip_header=15)
    q_expected, fq_expected = ph[:,0], ph[:,1]
    assert np.allclose(q, q_expected, atol=1e-04, rtol=1e-03), f"Expected Q to be {q_expected}, but got {q}"
    assert np.allclose(fq, fq_expected, atol=1e-04, rtol=1e-03), f"Expected F(Q) to be {fq_expected}, but got {fq}"

    # Check that the calculated Gr matches the expected value
    ph = np.genfromtxt('debyecalculator/unittests_files/gr_AntiFluorite_Co2O_radius10.0.dat', delimiter=',', skip_header=15)
    r_expected, gr_expected = ph[:,0], ph[:,1]
    assert np.allclose(r, r_expected, atol=1e-04, rtol=1e-03), f"Expected r to be {r_expected}, but got {r}"
    assert np.allclose(gr, gr_expected, atol=1e-04, rtol=1e-03), f"Expected G(r) to be {gr_expected}, but got {gr}"

def test_get_all_tuple():
    
    # calculate the scattering intensity using tuple
    ase_structure = read('debyecalculator/unittests_files/structure_AntiFluorite_Co2O_radius10.0.xyz')
    ase_elements = ase_structure.get_chemical_symbols()
    ase_xyz = ase_structure.get_positions()

    structure_tuple_str = (ase_elements, ase_xyz)
    structure_tuple_int = (np.array([element_to_atomic_number[e] for e in ase_elements]), ase_xyz)
    
    calc.update_parameters(qstep=0.05)
    r_str, q_str, iq_str, sq_str, fq_str, gr_str = calc._get_all(structure_tuple_str) 
    r_int, q_int, iq_int, sq_int, fq_int, gr_int = calc._get_all(structure_tuple_int) 
    
    # I(Q)
    ph = np.genfromtxt('debyecalculator/unittests_files/iq_AntiFluorite_Co2O_radius10.0.dat', delimiter=',', skip_header=15)
    q_expected, iq_expected = ph[:,0], ph[:,1]
    assert np.allclose(q_str, q_expected, atol=1e-04, rtol=1e-03), f"Expected Q to be {q_expected}, but got {q_str}"
    assert np.allclose(q_int, q_expected, atol=1e-04, rtol=1e-03), f"Expected Q to be {q_expected}, but got {q_int}"
    assert np.allclose(iq_str, iq_expected, atol=1e-04, rtol=1e-03), f"Expected I(Q) to be {iq_expected}, but got {iq_str}"
    assert np.allclose(iq_int, iq_expected, atol=1e-04, rtol=1e-03), f"Expected I(Q) to be {iq_expected}, but got {iq_int}"
    
    # S(Q)
    ph = np.genfromtxt('debyecalculator/unittests_files/sq_AntiFluorite_Co2O_radius10.0.dat', delimiter=',', skip_header=15)
    q_expected, sq_expected = ph[:,0], ph[:,1]
    assert np.allclose(q_str, q_expected, atol=1e-04, rtol=1e-03), f"Expected Q to be {q_expected}, but got {q_str}"
    assert np.allclose(q_int, q_expected, atol=1e-04, rtol=1e-03), f"Expected Q to be {q_expected}, but got {q_int}"
    assert np.allclose(sq_str, sq_expected, atol=1e-04, rtol=1e-03), f"Expected S(Q) to be {sq_expected}, but got {sq_str}"
    assert np.allclose(sq_int, sq_expected, atol=1e-04, rtol=1e-03), f"Expected S(Q) to be {sq_expected}, but got {sq_int}"

    # F(Q)
    ph = np.genfromtxt('debyecalculator/unittests_files/fq_AntiFluorite_Co2O_radius10.0.dat', delimiter=',', skip_header=15)
    q_expected, fq_expected = ph[:,0], ph[:,1]
    assert np.allclose(q_str, q_expected, atol=1e-04, rtol=1e-03), f"Expected Q to be {q_expected}, but got {q_str}"
    assert np.allclose(q_int, q_expected, atol=1e-04, rtol=1e-03), f"Expected Q to be {q_expected}, but got {q_int}"
    assert np.allclose(fq_str, fq_expected, atol=1e-04, rtol=1e-03), f"Expected F(Q) to be {fq_expected}, but got {fq_str}"
    assert np.allclose(fq_int, fq_expected, atol=1e-04, rtol=1e-03), f"Expected F(Q) to be {fq_expected}, but got {fq_int}"
    
    # G(r)
    ph = np.genfromtxt('debyecalculator/unittests_files/gr_AntiFluorite_Co2O_radius10.0.dat', delimiter=',', skip_header=15)
    r_expected, gr_expected = ph[:,0], ph[:,1]
    assert np.allclose(r_str, r_expected, atol=1e-04, rtol=1e-03), f"Expected r to be {r_expected}, but got {r_str}"
    assert np.allclose(r_int, r_expected, atol=1e-04, rtol=1e-03), f"Expected r to be {r_expected}, but got {r_int}"
    assert np.allclose(gr_str, gr_expected, atol=1e-04, rtol=1e-03), f"Expected G(r) to be {gr_expected}, but got {gr_str}"
    assert np.allclose(gr_int, gr_expected, atol=1e-04, rtol=1e-03), f"Expected G(r) to be {gr_expected}, but got {gr_int}"

def test_generate_nanoparticle():

    # Load structure from xyz:
    ase_structure = read('debyecalculator/unittests_files/structure_AntiFluorite_Co2O_radius10.0.xyz')
    xyz = ase_structure.get_positions()
    elements = ase_structure.get_chemical_symbols()

    # Generate xyz from utility function
    structure = generate_nanoparticles('data/AntiFluorite_Co2O.cif', radii=10.0)[0]

    # Assert
    assert np.allclose(xyz, structure.xyz.cpu(), atol=1e-04, rtol=1e-03), f"Expected xyz to be {xyz}, but got {structure.xyz}"
    assert elements == structure.elements, f"Expected elements to be {elements}, but got {structure.elements}"

def test_invalid_input():
    # Test that the DebyeCalculator raises a FileNotFoundError when given a non-existent file
    with pytest.raises(IOError):
        calc.iq('non_existent_file.xyz')

    # Test invalid update of parameters
    with pytest.raises(ValueError):
        calc.update_parameters(qmin=-1.0)
    with pytest.raises(ValueError):
        calc.update_parameters(qmin=-1.0)
    with pytest.raises(ValueError):
        calc.update_parameters(qmax=-1.0)
    with pytest.raises(ValueError):
        calc.update_parameters(qstep=-1.0)
    with pytest.raises(ValueError):
        calc.update_parameters(qdamp=-1.0)
    with pytest.raises(ValueError):
        calc.update_parameters(rmin=-1.0)
    with pytest.raises(ValueError):
        calc.update_parameters(rmax=-1.0)
    with pytest.raises(ValueError):
        calc.update_parameters(rstep=-1.0)
    with pytest.raises(ValueError):
        calc.update_parameters(rthres=-1.0)
    with pytest.raises(ValueError):
        calc.update_parameters(biso=-1.0)
    with pytest.raises(ValueError):
        calc.update_parameters(batch_size = -1)
    with pytest.raises(ValueError):
        calc.update_parameters(device = 'x')
    with pytest.raises(ValueError):
        calc.update_parameters(radiation_type = 'x')

