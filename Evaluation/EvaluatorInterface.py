class EvaluatorInterface:

    def setup_model(self):
        raise NotImplemented(f"Methode 'setup_model' not implemented")

    def setup_dataloader(self):
        raise NotImplemented(f"Methode 'setup_dataloader' not implemented")

    def run_batch(self, batch):
        raise NotImplemented(f"Methode 'run_batch' not implemented")

    def evaluate_metrics(self):
        raise NotImplemented(f"Methode 'evaluate_metrics' not implemented")

    def run_evaluation(self):
        self.setup_model()
        self.setup_dataloader()
        self.evaluate_metrics()
