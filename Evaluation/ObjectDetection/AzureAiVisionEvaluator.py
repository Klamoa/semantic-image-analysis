import argparse
import asyncio
import json

import azure.ai.vision as sdk
from overrides import override

from ObjectDetectionEvaluator import ObjectDetectionEvaluator


class AzureAiVisionEvaluator(ObjectDetectionEvaluator):

    __service_options = None
    __analysis_options = None
    __vocab = None

    def __init__(self, config, seed=42):
        super().__init__(config, seed)

    @override
    def setup_model(self):
        super().setup_model()

        # service options
        self.__service_options = sdk.VisionServiceOptions(self._config.vision_endpoint, self._config.vision_key)

        # analysis options
        self.__analysis_options = sdk.ImageAnalysisOptions()
        self.__analysis_options.features = sdk.ImageAnalysisFeature.OBJECTS
        self.__analysis_options.language = "en"

        # read the vocabulary
        with open(self._config.vocab) as file:
            self.__vocab = json.load(file)

    def run_instance_async(self, instance):
        image_analyzer = sdk.ImageAnalyzer(
            self.__service_options,
            sdk.VisionSource(filename=instance),
            self.__analysis_options
        )
        return image_analyzer.analyze_async()

    async def run_batch_async(self, batch):
        tasks = [self.run_instance_async(image_instance) for i, image_instance in enumerate(batch['path'])]
        out = await asyncio.gather(*tasks)

        # get result from output
        results = []
        for i, res in enumerate(out):
            for obj in res.objects:
                id = int(batch['id'][i].item()) # id
                coords = obj.bounding_box # coordinates in x, y, w, h
                score = obj.confidence # confidence score

                # get name and set it to the corresponding class of coco
                name = obj.name.lower()
                if self.__vocab.get(name):
                    name = self.__vocab.get(name)

                cls_id = self._coco_gt.getCatIds(catNms=[name]) # class id from coco

                # check if class name was found in coco
                if cls_id:
                    results.append([
                        id,
                        coords.x, coords.y, coords.w, coords.h,
                        score,
                        cls_id[0]
                    ])

        return results

    @override
    def run_batch(self, batch):
        return asyncio.get_event_loop().run_until_complete(self.run_batch_async(batch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vision_endpoint', type=str, default='', help='vision endpoint from azure')
    parser.add_argument('--vision_key', type=str, default='', help='vision key from azure')
    parser.add_argument('--path2img_root', type=str, default='../../ms-coco', help='path to image root')
    parser.add_argument('--path2ann_root', type=str, default='../../ms-coco/annotations', help='path to annotation file')
    parser.add_argument('--vocab', type=str, default='./datasets/AzureAiVision_vocab.json', help='path to vocabulary file')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size number for dataloader')
    parser.add_argument('--num_workers', type=int, default=2, help='number of worker for dataloader')
    parser.add_argument('--split', type=str, default='val2017', help='what split should be used')

    config = parser.parse_args()

    evaluator = AzureAiVisionEvaluator(config)
    evaluator.run_evaluation()
