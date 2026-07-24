from quantum.optimizer import QuantumOptimizer

for name in ["cobyla", "spsa", "slsqp"]:

    print("=" * 60)

    optimizer = QuantumOptimizer(
        optimizer=name,
        maxiter=200,
    )

    opt = optimizer.get_optimizer()

    print(opt)