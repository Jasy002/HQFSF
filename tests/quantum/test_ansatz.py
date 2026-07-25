"""
Unit Test for VariationalAnsatz.
"""

from quantum.ansatz import VariationalAnsatz

print("=" * 70)
print("LINEAR ENTANGLEMENT")
print("=" * 70)

linear_ansatz = VariationalAnsatz(
    n_qubits=4,
    layers=2,
    entanglement="linear",
)

linear_ansatz.summary()

linear_circuit = linear_ansatz.build()

print(linear_circuit.draw())

print()

print("Trainable Parameters")
print(linear_ansatz.trainable_parameters)

print()

print("=" * 70)
print("CIRCULAR ENTANGLEMENT")
print("=" * 70)

circular_ansatz = VariationalAnsatz(
    n_qubits=4,
    layers=2,
    entanglement="circular",
)

circular_ansatz.summary()

circular_circuit = circular_ansatz.build()

print(circular_circuit.draw())

print()

print("Trainable Parameters")
print(circular_ansatz.trainable_parameters)