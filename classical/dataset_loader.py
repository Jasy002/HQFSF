class DatasetLoader:

    def load_builtin(self, dataset_name: str):
        ...

    def load_csv(self, file_path: str):
        ...

    def load_excel(self, file_path: str):
        ...