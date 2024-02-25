from overrides import override
from pycocotools.coco import COCO

from .InferenceInterface import InferenceInterface
from .utils import *


class CocoVal2017(InferenceInterface):
    __config = None

    __coco = None

    def __init__(self, config):
        self.__config = config

        disbalePrint()
        # setup model
        self.__coco = COCO(self.__config["path2ann"])
        enablePrint()

    @override
    def _preprocessing(self, image_path: str):
        return image_path

    @override
    def _predicting(self, image):
        return image

    @override
    def _postprocessing(self, result):
        # get id from image_path
        imgId = int(os.path.splitext(os.path.basename(result))[0])

        # get annotations
        annIds = self.__coco.getAnnIds(imgIds=imgId)

        # load annotations
        objects = []
        for ann in self.__coco.loadAnns(ids=annIds):
            bbox = ann["bbox"]
            res = {
                "cls_name": self.__coco.cats[ann["category_id"]]["name"],
                "score": 1.0,
                "xyxy": [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            }
            objects.append(res)

        return {
            "objects": objects
        }
