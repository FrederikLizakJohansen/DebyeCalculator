import pytest, torch
from debyecalculator.debye_calculator import DebyeCalculator
import numpy as np

def test_init_defaults():
    calc = DebyeCalculator()
    assert calc.qmin == 1.0, f"Expected qmin to be 1.0, but got {calc.qmin}"
    assert calc.qmax == 30.0, f"Expected qmax to be 30.0, but got {calc.qmax}"
    assert calc.qstep == 0.1, f"Expected qstep to be 0.1, but got {calc.qstep}"
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

def test_iq():
    # Load the expected scattering intensity from a file
    ph = np.loadtxt('debyecalculator/unittests_files/icsd_001504_cc_r6_lc_2.85_6_tetragonal_Iq.dat')
    Q_expected, Iq_expected = ph[:,0], ph[:,1]

    # Calculate the scattering intensity using the DebyeCalculator
    calc = DebyeCalculator()
    Q, Iq = calc.iq('debyecalculator/unittests_files/icsd_001504_cc_r6_lc_2.85_6_tetragonal.xyz')

    # Check that the calculated scattering intensity matches the expected value
    assert np.allclose(Q, Q_expected, rtol=1e-03), f"Expected Q to be {Q_expected}, but got {Q}"
    assert np.allclose(Iq, Iq_expected, rtol=1e-03), f"Expected Iq to be {Iq_expected}, but got {Iq}"

def test_sq():
    # Load the expected structure factor from a file
    ph = np.loadtxt('debyecalculator/unittests_files/icsd_001504_cc_r6_lc_2.85_6_tetragonal_Sq.dat')
    Q_expected, sq_expected = ph[:,0], ph[:,1]

    # Calculate the structure factor using the DebyeCalculator
    calc = DebyeCalculator()
    Q, sq = calc.sq('debyecalculator/unittests_files/icsd_001504_cc_r6_lc_2.85_6_tetragonal.xyz')

    # Check that the calculated structure factor matches the expected value
    assert np.allclose(Q, Q_expected, rtol=1e-03), f"Expected Q to be {Q_expected}, but got {Q}"
    assert np.allclose(sq, sq_expected, rtol=1e-03), f"Expected Sq to be {sq_expected}, but got {sq}"

def test_fq():
    # Load the expected atomic form factor from a file
    ph = np.loadtxt('debyecalculator/unittests_files/icsd_001504_cc_r6_lc_2.85_6_tetragonal_Fq.dat')
    Q_expected, fq_expected = ph[:,0], ph[:,1]

    # Calculate the atomic form factor using the DebyeCalculator
    calc = DebyeCalculator()
    Q, fq = calc.fq('debyecalculator/unittests_files/icsd_001504_cc_r6_lc_2.85_6_tetragonal.xyz')

    # Check that the calculated atomic form factor matches the expected value
    assert np.allclose(Q, Q_expected, rtol=1e-03), f"Expected Q to be {Q_expected}, but got {Q}"
    assert np.allclose(fq, fq_expected, rtol=1e-03), f"Expected fq to be {fq_expected}, but got {fq}"

def test_gr():
    # Load the expected radial distribution function from a file
    ph = np.loadtxt('debyecalculator/unittests_files/icsd_001504_cc_r6_lc_2.85_6_tetragonal_Gr.dat')
    r_expected, gr_expected = ph[:,0], ph[:,1]

    # Calculate the radial distribution function using the DebyeCalculator
    calc = DebyeCalculator()
    r, gr = calc.gr('debyecalculator/unittests_files/icsd_001504_cc_r6_lc_2.85_6_tetragonal.xyz')

    # Check that the calculated radial distribution function matches the expected value
    assert np.allclose(r, r_expected, rtol=1e-03), f"Expected r to be {r_expected}, but got {r}"
    assert np.allclose(gr, gr_expected, rtol=1e-03), f"Expected Gr to be {gr_expected}, but got {gr}"

def test_get_all():
    # Calculate Iq, Fq, Sq, and Gr using the DebyeCalculator
    calc = DebyeCalculator()
    r, q, iq, sq, fq, gr = calc._get_all('debyecalculator/unittests_files/icsd_001504_cc_r6_lc_2.85_6_tetragonal.xyz')

    # Check that the calculated Iq matches the expected value
    ph = np.loadtxt('debyecalculator/unittests_files/icsd_001504_cc_r6_lc_2.85_6_tetragonal_Iq.dat')
    q_expected, iq_expected = ph[:,0], ph[:,1]
    assert np.allclose(q, q_expected, rtol=1e-03), f"Expected q to be {q_expected}, but got {q}"
    assert np.allclose(iq, iq_expected, rtol=1e-03), f"Expected Iq to be {iq_expected}, but got {iq}"

    # Check that the calculated Sq matches the expected value
    ph = np.loadtxt('debyecalculator/unittests_files/icsd_001504_cc_r6_lc_2.85_6_tetragonal_Sq.dat')
    q_expected, sq_expected = ph[:,0], ph[:,1]
    assert np.allclose(q, q_expected, rtol=1e-03), f"Expected q to be {q_expected}, but got {q}"
    assert np.allclose(sq, sq_expected, rtol=1e-03), f"Expected Sq to be {sq_expected}, but got {sq}"

    # Check that the calculated Fq matches the expected value
    ph = np.loadtxt('debyecalculator/unittests_files/icsd_001504_cc_r6_lc_2.85_6_tetragonal_Fq.dat')
    q_expected, fq_expected = ph[:,0], ph[:,1]
    assert np.allclose(q, q_expected, rtol=1e-03), f"Expected q to be {q_expected}, but got {q}"
    assert np.allclose(fq, fq_expected, rtol=1e-03), f"Expected Fq to be {fq_expected}, but got {fq}"

    # Check that the calculated Gr matches the expected value
    ph = np.loadtxt('debyecalculator/unittests_files/icsd_001504_cc_r6_lc_2.85_6_tetragonal_Gr.dat')
    r_expected, gr_expected = ph[:,0], ph[:,1]
    assert np.allclose(gr, gr_expected, rtol=1e-03), f"Expected r to be {gr_expected}, but got {gr}"
    assert np.allclose(gr, gr_expected, rtol=1e-03), f"Expected Gr to be {gr_expected}, but got {gr}"

def test_invalid_input():
    # Test that the DebyeCalculator raises a FileNotFoundError when given a non-existent file
    with pytest.raises(FileNotFoundError):
        calc = DebyeCalculator()
        calc.iq('non_existent_file.xyz')

    # Test that the DebyeCalculator raises a ValueError when given invalid input parameters in the iq method
    with pytest.raises(ValueError):
        calc = DebyeCalculator()
        calc.update_parameters(qmin=-1.0)
        calc.iq('debyecalculator/unittests_files/icsd_001504_cc_r6_lc_2.85_6_tetragonal.xyz')
