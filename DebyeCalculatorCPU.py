import torch
import numpy as np
from torch.nn.functional import pdist
from profiling import Timings
from formfactor_coef import ff_coef

class DebyeCalculatorGPU:
    def __init__(
        self,
        qmin = 0.0,
        qmax = 30.0,
        qstep = 0.1,
        qdamp = 0.0,
        rmin = 0.0,
        rmax = 20.0,
        rstep = 0.01,
        rthres = 1.0,
        biso = 0.0,
        device = 'cuda',
        batch_size = None,
        lorch_mod = False,
        radiation_type = 'xray',
        verbose = 0,
        profiling = False,
    ):
        self.profiling = profiling
        self.timings = Timings()
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

        # Formfactor retrieval lambda
        self.ff_coef = ff_coef.copy()
        for k,v in self.ff_coef.items():
            self.ff_coef[k] = torch.FloatTensor(v).to(device=self.device)
        if radiation_type.lower() in ['xray', 'x']:
            self.formfactor_func = lambda p: torch.sum(p[:5] * torch.exp(-1*p[6:11] * (self.q / (4*torch.pi))**2), dim=1) + p[5]
        elif radiation_type.lower() in ['neutron', 'n']:
            self.formfactor_func = lambda p: p[11].unsqueeze(-1)

        # Ligand list
        self.ligand_list = ['O', 'H', 'Cl']

    def _init_structure_formfactors(
        self,
    ):    
        # 
        unique_elements, inverse, counts = np.unique(self.struc_elements, return_counts=True, return_inverse=True)
        self.triu_indices = torch.triu_indices(self.struc_size, self.struc_size, 1)
        self.unique_inverse = torch.from_numpy(inverse[self.triu_indices]).to(device=self.device)
        self.struc_unique_formfactors = torch.stack([self.formfactor_func(self.ff_coef[el]) for el in unique_elements])
        
        # Get f_avg_sqrd and f_sqrd_avg
        counts = torch.from_numpy(counts).to(device=self.device)
        compositional_fractions = counts / torch.sum(counts)
        self.struc_fsa = torch.sum(compositional_fractions.reshape(-1,1) * self.struc_unique_formfactors**2, dim=0)
        #self.struc_fas = torch.sum(compositional_fractions.reshape(-1,1) * self.struc_unique_formfactors, dim=0)**2

        # self scattering
        self.struc_inverse = torch.from_numpy(np.array([inverse[i] for i in range(self.struc_size)])).to(device=self.device)

    def _init_structure(
        self,
        structure,
    ):
        # XYZ file
        if isinstance(structure, str):
            path_ext = structure.split('.')[-1]
            if path_ext == 'xyz':
                struc = np.genfromtxt(structure, dtype='str', skip_header=2)
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

    def set_parameters(
        self,
        **kwargs,
    ):
        for k,v in kwargs.items():
            setattr(self, k, v)
        
        # Re-initialise ranges
        self.q = torch.arange(self.qmin, self.qmax, self.qstep).unsqueeze(-1).to(device=self.device)
        self.r = torch.arange(self.rmin, self.rmax, self.rstep).unsqueeze(-1).to(device=self.device)

    def iq(
        self,
        structure,
        _keep_on_device = False,
        _for_total_scattering = False,
    ):
        # Setup structure
        self.timings.reset()
        self._init_structure(structure)
        if self.profiling: self.timings.time('Setup Structure')

        # Setup atomic formfactors
        self._init_structure_formfactors()
        if self.profiling: self.timings.time('Setup Formfactors')

        # Calculate distances and batch
        if self.batch_size is None:
            self.batch_size = self.num_pairs
        dists = pdist(self.struc_xyz).split(self.batch_size)
        indices = self.triu_indices.split(self.batch_size, dim=1)
        inverse_indices = self.unique_inverse.split(self.batch_size, dim=1)

        # Calculate scattering using Debye Equation
        iq = torch.zeros((len(self.q))).to(device=self.device)
        for d, inv_idx, idx in zip(dists, inverse_indices, indices):
            if d < self.rthres:
                continue
            occ_product = self.struc_occupancy[idx[0]] * self.struc_occupancy[idx[1]]
            sinc = torch.sinc(d * self.q / torch.pi)
            ffp = self.struc_unique_formfactors[inv_idx[0]] * self.struc_unique_formfactors[inv_idx[1]]
            iq += torch.sum(occ_product.unsqueeze(-1) * ffp * sinc.permute(1,0), dim=0)

        # Apply Debye-Weller Isotropic Atomic Displacement
        DW = 1 if self.biso == 0.0 else torch.exp(-(self.q.squeeze(-1))**2 * self.biso/(8*torch.pi**2))
        iq *= DW
        
        # For total scattering
        if _for_total_scattering:
            if self.profiling: self.timings.time('I(Q)')
            return iq

        # Self-scattering contribution
        self_sinc = torch.ones((self.struc_size, len(self.q))).to(device=self.device)
        iq += torch.sum((self.struc_occupancy.unsqueeze(-1) * self.struc_unique_formfactors[self.struc_inverse])**2 * self_sinc, dim=0)
        if self.profiling: self.timings.time('I(Q)')

        if _keep_on_device:
            return self.q.squeeze(-1), iq
        else:
            return self.q.squeeze(-1).cpu().numpy(), iq_out.cpu().numpy()

    def gr(
        self,
        structure,
        _keep_on_device = False,
    ):
        # Calculate Scattering I(Q), S(Q), F(Q)
        iq = self.iq(structure, _keep_on_device=True, _for_total_scattering=True)
        self.sq = iq/self.struc_fsa/self.struc_size
        self.timings.time('S(Q)')
        self.fq = self.q.squeeze(-1) * self.sq
        self.timings.time('F(Q)')
        
        # Calculate total scattering, G(r)
        damp = 1 if self.qdamp == 0.0 else torch.exp(-(self.r.squeeze(-1) * self.qdamp)**2 / 2)
        lorch_mod = 1 if self.lorch_mod == None else torch.sinc(self.q * self.lorch_mod*(torch.pi / self.qmax))
        self.timings.time('Modifications, Qdamp/Lorch')
        gr = (2 / torch.pi) * torch.sum(self.fq.unsqueeze(-1) * torch.sin(self.q * self.r.permute(1,0))*self.qstep * lorch_mod, dim=0) * damp
        self.timings.time('G(r)')

        if _keep_on_device:
            return self.r.squeeze(-1), gr
        else:
            return self.r.squeeze(-1).cpu().numpy(), gr.cpu().numpy()

    def _diffpy_gr(self, path):
        self.timings.reset()
        path_ext = path.split('.')[-1]
        if path_ext == 'xyz':
            # Load structure with Diffpy
            struc = loadStructure(path)
            self.timings.time('Load Structure')
            xyz_device = torch.tensor(struc.xyz_cartn).to(device='cuda')
            if self.profiling: self.timings.time('Pull Structure from Device')
    
            struc.B11 = self.biso
            struc.B22 = self.biso
            struc.B33 = self.biso
            struc.B12 = 0
            struc.B13 = 0
            struc.B23 = 0
            dbc = DebyePDFCalculator(
                qmin = self.qmin,
                qmax = self.qmax,
                qdamp = self.qdamp,
                rmin = self.rmin,
                rmax = self.rmax,
                rstep = self.rstep,
                delta2 = self.delta2,
            )
            if self.profiling: self.timings.time('Setup DebyePDFCalculator')
            r, gr = dbc(struc)
            self.timings.time('G(r)')
            gr = torch.from_numpy(gr).to(dtype=torch.float32, device=self.device)
            if self.profiling: self.timings.time('Push PDF to Device')
        else:
            raise NotImplementedError('Structure File Extention Not Supported')
