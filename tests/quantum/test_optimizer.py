"""
Unit Test for QuantumOptimizer.
"""

from quantum.optimizer import QuantumOptimizer

print("=" * 70)
print("SUPPORTED OPTIMIZERS")
print("=" * 70)

print(QuantumOptimizer.available_optimizers())

print()

for optimizer_name in [
    "cobyla",
    "spsa",
    "slsqp",
]:

    print("=" * 70)
    print(f"{optimizer_name.upper()} OPTIMIZER")
    print("=" * 70)

    optimizer = QuantumOptimizer(
        optimizer=optimizer_name,
        maxiter=200,
    )

    optimizer.summary()

    opt = optimizer.get_optimizer()

    print(opt)
    print()