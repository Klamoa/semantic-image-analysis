import io

from overrides import override
from google.cloud import vision
from PIL import Image

from .InferenceInterface import InferenceInterface


class GoogleCloudVisionInference(InferenceInterface):
    __config = None
    __client = None
    __image_size = None

    def __init__(self, config):
        self.__config = config

        # get client
        self.__client = vision.ImageAnnotatorClient()

    @override
    def _preprocessing(self, image_path: str):
        with open(image_path, 'rb') as image_file:
            content = image_file.read()

        self.__image_size = Image.open(io.BytesIO(content)).size

        return content

    @override
    def _predicting(self, image):
        request = vision.AnnotateImageRequest(
            image=vision.Image(content=image),
            features=[vision.Feature(type_=vision.Feature.Type.OBJECT_LOCALIZATION)]
        )
        return self.__client.annotate_image(request=request)

    @override
    def _postprocessing(self, result):
        objects = []
        for obj in result.localized_object_annotations:
            if obj.score >= self.__config["confThreshold"]:
                bbox = obj.bounding_poly.normalized_vertices
                res = {
                    "cls_name": obj.name.lower(),
                    "score": obj.score,
                    "xyxy": [
                        bbox[0].x * self.__image_size[0],
                        bbox[0].y * self.__image_size[1],
                        bbox[2].x * self.__image_size[0],
                        bbox[2].y * self.__image_size[1],
                    ],
                }
                objects.append(res)

        return {
            "objects": objects
        }
