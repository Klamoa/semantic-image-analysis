from overrides import override
import azure.ai.vision as sdk

from .InferenceInterface import InferenceInterface


class AzureAiVisionInference(InferenceInterface):
    __config = None
    __service_options = None
    __analysis_options = None

    def __init__(self, config):
        self.__config = config

        # service options
        self.__service_options = sdk.VisionServiceOptions(self.__config["vision_endpoint"], self.__config["vision_key"])

        # analysis options
        self.__analysis_options = sdk.ImageAnalysisOptions()
        self.__analysis_options.features = [sdk.ImageAnalysisFeature.OBJECTS, sdk.ImageAnalysisFeature.CAPTION]
        self.__analysis_options.language = "en"

    @override
    def _preprocessing(self, image_path: str):
        return image_path

    @override
    def _predicting(self, image):
        image_analyzer = sdk.ImageAnalyzer(
            self.__service_options,
            sdk.VisionSource(filename=image),
            self.__analysis_options
        )
        return image_analyzer.analyze()

    @override
    def _postprocessing(self, result):
        objects = []
        for obj in result.objects:
            if obj.confidence >= self.__config["confThreshold"]:
                bbox = obj.bounding_box
                res = {
                    "cls_name": obj.name.lower(),
                    "score": obj.confidence,
                    "xyxy": [
                        bbox.x,
                        bbox.y,
                        bbox.x + bbox.w,
                        bbox.y + bbox.h
                    ],
                }
                objects.append(res)

        return {
            "objects": objects,
            "caption": result.caption.content
        }
