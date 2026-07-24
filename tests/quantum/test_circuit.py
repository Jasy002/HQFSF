from quantum.circuit import HQFSFCircuit

qc = HQFSFCircuit(4)

qc.add_barrier()

qc.measure_all()

print(qc.draw())