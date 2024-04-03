import argparse
import asyncio

import azure.ai.vision as sdk
from overrides import override

from ImageCaptionEvaluator import ImageCaptionEvaluator

from datasets.field import ImageField, TextField
from datasets import build_coco_dataloaders


class AzureAiVisionImgCapEvaluator(ImageCaptionEvaluator):

    __text_field = None
    __service_options = None
    __analysis_options = None

    def __init__(self, config, seed=42):
        super().__init__(config, seed)

    @override
    def setup_model(self):
        super().setup_model()

        # service options
        self.__service_options = sdk.VisionServiceOptions(self._config.vision_endpoint, self._config.vision_key)

        # analysis options
        self.__analysis_options = sdk.ImageAnalysisOptions()
        self.__analysis_options.features = sdk.ImageAnalysisFeature.CAPTION
        self.__analysis_options.language = "en"

    @override
    def setup_dataloader(self):
        # get text field
        self.__text_field = TextField(use_vocab=False, build_vocab=True)

        # build coco dataloaders
        self._dataloaders = build_coco_dataloaders(
            image_field=ImageField(only_path=True),
            config=self._config,
            mode='finetune',
            device=self._device
        )

    def run_instance_async(self, instance):
        image_analyzer = sdk.ImageAnalyzer(
            self.__service_options,
            sdk.VisionSource(filename=instance),
            self.__analysis_options
        )
        return image_analyzer.analyze_async()

    async def run_batch_async(self, batch):
        tasks = [self.run_instance_async(image_instance) for i, image_instance in enumerate(batch['samples'])]
        results = await asyncio.gather(*tasks)
        tokens = [self.__text_field.preprocess(result.caption.content) for result in results]
        return zip(batch['captions'], tokens)

    @override
    def run_batch(self, batch):
        return asyncio.get_event_loop().run_until_complete(self.run_batch_async(batch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vision_endpoint', type=str, default='', help='vision endpoint from azure')
    parser.add_argument('--vision_key', type=str, default='', help='vision key from azure')
    parser.add_argument('--path2img_root', type=str, default='../../ms-coco/', help='path to image root')
    parser.add_argument('--path2ann_root', type=str, default='../../ms-coco/annotations/', help='path to annotation root')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size number for dataloader')
    parser.add_argument('--num_workers', type=int, default=2, help='number of worker for dataloader')
    parser.add_argument('--split', type=str, default='valid', help='what split should be used')

    config = parser.parse_args()

    evaluator = AzureAiVisionImgCapEvaluator(config)
    evaluator.run_evaluation()
