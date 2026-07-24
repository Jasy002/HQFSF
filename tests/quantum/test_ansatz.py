from quantum.ansatz import VariationalAnsatz

print("=" * 60)

ansatz = VariationalAnsatz(
    n_qubits=4,
    layers=2,
    entanglement="linear",
)

qc = ansatz.build()

print(qc.draw())

print()

print("=" * 60)

ansatz = VariationalAnsatz(
    n_qubits=4,
    layers=2,
    entanglement="circular",
)

qc = ansatz.build()

print(qc.draw())