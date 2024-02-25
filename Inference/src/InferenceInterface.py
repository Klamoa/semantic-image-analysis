class InferenceInterface:

    def _preprocessing(self, image_path: str):
        raise NotImplemented(f"Methode '_preprocessing' not implemented")

    def _predicting(self, image):
        raise NotImplemented(f"Methode '_predicting' not implemented")

    def _postprocessing(self, result):
        raise NotImplemented(f"Methode '_postprocessing' not implemented")

    def run_inference(self, image_path: str):
        return self._postprocessing(self._predicting(self._preprocessing(image_path)))
