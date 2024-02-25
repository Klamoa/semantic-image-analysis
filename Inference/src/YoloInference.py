import torch

from overrides import override
from ultralytics import YOLO

from .InferenceInterface import InferenceInterface


class YoloInference(InferenceInterface):
    __config = None
    __device = None

    __model = None

    def __init__(self, config):
        self.__config = config

        # set device
        self.__device = torch.device(f"cuda:{0}")
        torch.cuda.set_device(0)

        # setup model
        self.__model = YOLO(self.__config["path2model"])

    @override
    def _preprocessing(self, image_path: str):
        return image_path

    @override
    def _predicting(self, image):
        return self.__model.predict(image, conf=self.__config["confThreshold"], verbose=False)

    @override
    def _postprocessing(self, result):
        result = result[0]
        names = result.names  # dictionary key: number, value: name

        objects = []
        for obj in result.boxes:
            res = {
                "cls_name": (names[obj.cls[0].item()]).lower(),
                "score": obj.conf[0].item(),
                "xyxy": obj.xyxy[0].tolist(),
            }
            objects.append(res)

        return {
            "objects": objects
        }
