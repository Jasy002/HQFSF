import numpy as np

from quantum.encoder import QuantumEncoder

sample = np.array([
    0.2,
    0.5,
    0.9,
    0.1,
])

encoder = QuantumEncoder(4)

for method in ["ry", "rx", "rz"]:

    print("=" * 50)

    print(method.upper())

    qc = encoder.encode(sample, method=method)

    print(qc.draw())