"""
Unit Test for QuantumMeasurement.
"""

import numpy as np

from quantum.circuit import HQFSFCircuit
from quantum.backend import QuantumBackend
from quantum.measurement import QuantumMeasurement

sample = np.array([
    0.2,
    0.5,
    0.9,
    0.1,
])

# Build circuit
builder = HQFSFCircuit(
    n_qubits=4,
    layers=2,
    encoding="ry",
    entanglement="linear",
)

qc = builder.build(sample)

qc = builder.measure(qc)

# Backend
backend = QuantumBackend()

# Measurement
measurement = QuantumMeasurement(
    backend=backend,
    shots=1024,
)

measurement.summary()

counts = measurement.counts(qc)

print("Measurement Counts\n")
print(counts)

print()

print("Probabilities\n")
print(measurement.probabilities(qc))