import os
import yaml
import torch
import numpy as np
from torch.nn.functional import pdist
from profiling import Profiler
from ase import Atoms

class DebyeCalculator:
    """
    Calculate Debye scattering intensity and pair distribution function (PDF) for a given atomic structure.

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
        verbose (int): Verbosity level (0: no messages, 1: show profiling). Default is 0.
    """
    def __init__(
        self,
        qmin = 0.0,
        qmax = 30.0,
        qstep = 0.1,
        qdamp = 0.0,
        rmin = 0.0,
        rmax = 20.0,
        rstep = 0.01,
        rthres = 0.0,
        biso = 0.0,
        device = 'cuda',
        batch_size = None,
        lorch_mod = False,
        radiation_type = 'xray',
        verbose = 0,
    ):
        self.profiler = Profiler()
        self.verbose = verbose
        
        # Initial parameters
        self.device = device
        self.batch_size = batch_size
        self.lorch_mod = lorch_mod
        
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
        with open('form_factor_coef.yaml', 'r') as yaml_file:
            self.FORM_FACTOR_COEF = yaml.safe_load(yaml_file)

        # Formfactor retrieval lambda
        for k,v in self.FORM_FACTOR_COEF.items():
            if None in v:
                v = [value if value is not None else np.nan for value in v]
            self.FORM_FACTOR_COEF[k] = torch.FloatTensor(v).to(device=self.device)
        if radiation_type.lower() in ['xray', 'x']:
            self.form_factor_func = lambda p: torch.sum(p[:5] * torch.exp(-1*p[6:11] * (self.q / (4*torch.pi)).pow(2)), dim=1) + p[5]
        elif radiation_type.lower() in ['neutron', 'n']:
            self.form_factor_func = lambda p: p[11].unsqueeze(-1)

    def _initialise_structure(
        self,
        structure_path,
    ):
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
                    self.struc_occupancy = torch.from_numpy(struc[:,-1]).to(device=self.device)
                    self.struc_xyz = torch.FloatTensor(struc[:,1:-1].astype('float')).to(device=self.device)
                else:
                    self.struc_occupancy = torch.ones((self.struc_size), dtype=torch.float32).to(device=self.device)
                    self.struc_xyz = torch.FloatTensor(struc[:,1:].astype('float')).to(device=self.device)
            else:
                raise NotImplementedError('Structure File Extention Not Supported')
        elif isinstance(structure_path, Atoms):
            self.struc_elements = structure_path.get_chemical_symbols()
            self.struc_size = len(self.struc_elements)
            self.num_pairs = self.struc_size * (self.struc_size - 1) // 2
            self.struc_occupancy = torch.ones((self.struc_size), dtype=torch.float32).to(device=self.device)
            self.struc_xyz = torch.FloatTensor(np.array(structure_path.get_positions())).to(device=self.device)
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
        #self.struc_fsa = torch.sum(compositional_fractions.reshape(-1,1) * self.struc_unique_form_factors**2, dim=0)
        self.struc_fsa = torch.sum(compositional_fractions.reshape(-1,1) * self.struc_unique_form_factors, dim=0)**2 # FAS

        # self scattering
        self.struc_inverse = torch.from_numpy(np.array([inverse[i] for i in range(self.struc_size)])).to(device=self.device)

    def update_parameters(
        self,
        **kwargs,
    ):
        """
        Set or update the parameters of the DebyeCalculator.

        Parameters:
            **kwargs: Arbitrary keyword arguments to update the parameters.
        """
        for k,v in kwargs.items():
            setattr(self, k, v)
            
        if np.any([k in ['qmin','qmax','qstep','rmin', 'rmax', 'rstep'] for k in kwargs.keys()]):
            # Re-initialise ranges
            self.q = torch.arange(self.qmin, self.qmax, self.qstep).unsqueeze(-1).to(device=self.device)
            self.r = torch.arange(self.rmin, self.rmax, self.rstep).unsqueeze(-1).to(device=self.device)

    def iq(
        self,
        structure,
        _keep_on_device = False,
        _for_total_scattering = False,
    ):
        """
        Calculate the scattering intensity I(q) for the given atomic structure.

        Parameters:
            structure (str): Path to the atomic structure file in XYZ format.
            _keep_on_device (bool): Flag to keep the results on the class device. Default is False.
            _for_total_scattering (bool): Flag to return the scattering intensity I(q) without the self-scattering contribution. Default is False.

        Returns:
            tuple or numpy.ndarray: Tuple containing q-values and scattering intensity I(q) if _keep_on_device is True, otherwise, numpy arrays on CPU.
        """
        self._initialise_structure(structure)
        if self.verbose > 0:
            self.profiler.time('Setup structure and form factors')

        # Calculate distances and batch
        if self.batch_size is None:
            self.batch_size = self.num_pairs
        dists = pdist(self.struc_xyz).split(self.batch_size)
        indices = self.triu_indices.split(self.batch_size, dim=1)
        inverse_indices = self.unique_inverse.split(self.batch_size, dim=1)
        if self.verbose > 0:
            self.profiler.time('Batching and Distances')

        # Calculate scattering using Debye Equation
        iq = torch.zeros((len(self.q))).to(device=self.device)
        for d, inv_idx, idx in zip(dists, inverse_indices, indices):
            mask = d >= self.rthres
            occ_product = self.struc_occupancy[idx[0]] * self.struc_occupancy[idx[1]]
            sinc = torch.sinc(d[mask] * self.q / torch.pi)
            ffp = self.struc_unique_form_factors[inv_idx[0]] * self.struc_unique_form_factors[inv_idx[1]]
            iq += torch.sum(occ_product.unsqueeze(-1) * ffp * sinc.permute(1,0), dim=0)

        # Apply Debye-Weller Isotropic Atomic Displacement
        DW = 1 if self.biso == 0.0 else torch.exp(-self.q.squeeze(-1).pow(2) * self.biso/(8*torch.pi**2))
        iq *= DW
        
        # For total scattering
        if _for_total_scattering:
            if self.verbose > 0:
                self.profiler.time('I(Q)')
            return iq

        # Self-scattering contribution
        self_sinc = torch.ones((self.struc_size, len(self.q))).to(device=self.device)
        iq += torch.sum((self.struc_occupancy.unsqueeze(-1) * self.struc_unique_form_factors[self.struc_inverse])**2 * self_sinc, dim=0) / 2
        iq *= 2
        if self.verbose > 0:
            self.profiler.time('I(Q)')

        if _keep_on_device:
            return self.q.squeeze(-1), iq
        else:
            return self.q.squeeze(-1).cpu().numpy(), iq.cpu().numpy()

    def sq(
        self,
        structure,
        _keep_on_device = False,
    ):
        # Calculate Scattering S(Q)
        iq = self.iq(structure, _keep_on_device=True, _for_total_scattering=True)
        sq = iq/self.struc_fsa/self.struc_size
        if _keep_on_device:
            return self.q.squeeze(-1), sq
        else:
            return self.q.squeeze(-1).cpu().numpy(), sq.cpu().numpy()
    
    def fq(
        self,
        structure,
        _keep_on_device = False,
    ):
        # Calculate Scattering S(Q)
        iq = self.iq(structure, _keep_on_device=True, _for_total_scattering=True)
        sq = iq/self.struc_fsa/self.struc_size
        fq = self.q.squeeze(-1) * sq
        if _keep_on_device:
            return self.q.squeeze(-1), fq
        else:
            return self.q.squeeze(-1).cpu().numpy(), fq.cpu().numpy()

    def gr(
        self,
        structure,
        _keep_on_device = False,
    ):
        """
        Calculate the pair distribution function G(r) for the given atomic structure.

        Parameters:
            structure (str): Path to the atomic structure file in XYZ format.
            _keep_on_device (bool): Flag to keep the results on the class device. Default is False.

        Returns:
            tuple or numpy.ndarray: Tuple containing r-values and pair distribution function G(r) if _keep_on_device is True, otherwise, numpy arrays on CPU.
        """
        # Calculate Scattering I(Q), S(Q), F(Q)
        if self.verbose > 0:
            self.profiler.reset()
        iq = self.iq(structure, _keep_on_device=True, _for_total_scattering=True)
        sq = iq/self.struc_fsa/self.struc_size
        if self.verbose > 0:
            self.profiler.time('S(Q)')
        fq = self.q.squeeze(-1) * sq
        if self.verbose > 0:
            self.profiler.time('F(Q)')
        
        # Calculate total scattering, G(r)
        damp = 1 if self.qdamp == 0.0 else torch.exp(-(self.r.squeeze(-1) * self.qdamp).pow(2) / 2)
        lorch_mod = 1 if self.lorch_mod == None else torch.sinc(self.q * self.lorch_mod*(torch.pi / self.qmax))
        if self.verbose > 0:
            self.profiler.time('Modifications, Qdamp/Lorch')
        gr = (2 / torch.pi) * torch.sum(fq.unsqueeze(-1) * torch.sin(self.q * self.r.permute(1,0))*self.qstep * lorch_mod, dim=0) * damp
        if self.verbose > 0:
            self.profiler.time('G(r)')

        if _keep_on_device:
            return self.r.squeeze(-1), gr
        else:
            return self.r.squeeze(-1).cpu().numpy(), gr.cpu().numpy()
