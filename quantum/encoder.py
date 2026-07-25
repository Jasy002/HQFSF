class QuantumEncoder:
    SUPPORTED_METHODS = ("ry", "rx", "rz")

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits

    def encode(
        self,
        features: np.ndarray,
        encoding_method: str = "ry",
    ) -> QuantumCircuit:

        features = np.asarray(features, dtype=float)

        if len(features) < self.n_qubits:
            raise ValueError(
                f"Expected at least {self.n_qubits} features, got {len(features)}."
            )

        encoding_method = encoding_method.lower()

        if encoding_method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported encoding '{encoding_method}'. "
                f"Supported methods: {self.SUPPORTED_METHODS}"
            )

        qc = QuantumCircuit(self.n_qubits)

        for qubit in range(self.n_qubits):
            angle = float(features[qubit])

            if encoding_method == "ry":
                qc.ry(angle, qubit)

            elif encoding_method == "rx":
                qc.rx(angle, qubit)

            elif encoding_method == "rz":
                qc.rz(angle, qubit)

        logger.info(
            "%s encoding applied (%d qubits).",
            encoding_method.upper(),
            self.n_qubits,
        )

        return qc