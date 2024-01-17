from debyecalculator import DebyeCalculator
import torch 

# Initialise calculator object
calc = DebyeCalculator(qmin=1.0, qmax=8.0, qstep=0.01, device='cpu')

# Define structure sources
structure_tuple = (
    ["Fe", "Fe", "O", "O"],
    torch.tensor(
        [[0.5377, 0.7068, 0.8589],
         [0.1576, 0.1456, 0.8799],
         [0.5932, 0.0204, 0.6759],
         [0.6946, 0.4114, 0.4869]]
))

# Create a mask for oxygen atoms
mask = [atom == "O" for atom in structure_tuple[0]]

# Apply the mask to the structure_tuple
structure_tuple = (
    [atom for atom, m in zip(structure_tuple[0], mask) if m],
    structure_tuple[1][mask]
)

Q_truncatedStructure, I_truncatedStructure = calc.iq(structure_source=structure_tuple)