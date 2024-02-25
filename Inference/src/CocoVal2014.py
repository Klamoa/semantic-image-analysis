from overrides import override
from pycocotools.coco import COCO

from .InferenceInterface import InferenceInterface
from .utils import *


class CocoVal2014(InferenceInterface):
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
        file_name = result.split("_")[-1]
        imgId = int(os.path.splitext(os.path.basename(file_name))[0])

        # get annotations
        annIds = self.__coco.getAnnIds(imgIds=imgId)

        return {
            "caption": [ann["caption"] for ann in self.__coco.loadAnns(ids=annIds)]
        }
