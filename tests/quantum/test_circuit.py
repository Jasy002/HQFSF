"""
Unit Test for HQFSFCircuit.
"""

import numpy as np

from quantum.circuit import HQFSFCircuit

sample = np.array([
    0.2,
    0.5,
    0.9,
    0.1,
])

circuit = HQFSFCircuit(
    n_qubits=4,
    layers=2,
    encoding="ry",
    entanglement="linear",
)

qc = circuit.build(sample)

qc = circuit.measure(qc)

print(qc.draw())