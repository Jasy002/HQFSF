"""
Unit Test for QuantumBackend.
"""

from quantum.backend import QuantumBackend

print("=" * 60)
print("Aer Simulator")
print("=" * 60)

backend = QuantumBackend(
    backend_type="aer_simulator"
)

backend.summary()

print("Backend Name:")
print(backend.backend_name())

print()

print("=" * 60)
print("Statevector Simulator")
print("=" * 60)

backend = QuantumBackend(
    backend_type="statevector"
)

backend.summary()

print("Backend Name:")
print(backend.backend_name())