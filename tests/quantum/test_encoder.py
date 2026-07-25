"""
Unit test for QuantumEncoder.
"""

import numpy as np

from quantum.encoder import QuantumEncoder


sample = np.array([
    0.2,
    0.5,
    0.9,
    0.1,
])

encoder = QuantumEncoder(n_qubits=4)

for encoding_method in ["ry", "rx", "rz"]:

    print("=" * 50)
    print(f"{encoding_method.upper()} Encoding")
    print("=" * 50)

    qc = encoder.encode(
        sample,
        encoding_method=encoding_method
    )

    print(qc.draw())
    print()