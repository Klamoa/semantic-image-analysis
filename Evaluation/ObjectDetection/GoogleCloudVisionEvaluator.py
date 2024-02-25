import argparse
import asyncio
import json

from google.cloud import vision
from overrides import override
from PIL import Image
import io

from ObjectDetectionEvaluator import ObjectDetectionEvaluator


class GoogleCloudVisionEvaluator(ObjectDetectionEvaluator):

    __client = None
    __vocab = None

    def __init__(self, config, seed=42):
        super().__init__(config, seed)

    @override
    def setup_model(self):
        super().setup_model()

        # get client
        self.__client = vision.ImageAnnotatorClient()

        # read the vocabulary
        with open(self._config.vocab) as file:
            self.__vocab = json.load(file)

    @override
    def run_batch(self, batch):
        requests = [] # request list
        sizes = [] # size list for calculating x, y, w, h because google cloud vision returns normalized coordinates
        for image_path in batch['path']:
            with open(image_path, 'rb') as image_file:
                content = image_file.read()

            sizes.append(Image.open(io.BytesIO(content)).size)

            image = vision.Image(content=content)
            request = vision.AnnotateImageRequest(image=image, features=[vision.Feature(type_=vision.Feature.Type.OBJECT_LOCALIZATION)])
            requests.append(request)

        batch_request = vision.BatchAnnotateImagesRequest(requests=requests)
        response = self.__client.batch_annotate_images(request=batch_request)

        results = []
        for i, (size, image_response) in enumerate(zip(sizes, response.responses)):
            for obj in image_response.localized_object_annotations:
                x = obj.bounding_poly.normalized_vertices[0].x * size[0]
                y = obj.bounding_poly.normalized_vertices[0].y * size[1]
                width = (obj.bounding_poly.normalized_vertices[2].x - obj.bounding_poly.normalized_vertices[0].x) * size[0]
                height = (obj.bounding_poly.normalized_vertices[2].y - obj.bounding_poly.normalized_vertices[0].y) * size[1]

                id = int(batch['id'][i].item())  # id

                # get name and set it to the corresponding class of coco
                name = obj.name.lower()
                if self.__vocab.get(name):
                    name = self.__vocab.get(name)

                cls_id = self._coco_gt.getCatIds(catNms=[name])  # class id from coco

                # check if class name was found in coco
                if cls_id:
                    results.append([
                        id,
                        x, y, width, height,
                        obj.score,
                        cls_id[0]
                    ])

        return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path2img_root', type=str, default='../../ms-coco', help='path to image root')
    parser.add_argument('--path2ann_root', type=str, default='../../ms-coco/annotations', help='path to annotation file')
    parser.add_argument('--vocab', type=str, default='./datasets/GoogleCloudVision_vocab.json', help='path to vocabulary file')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size number for dataloader')
    parser.add_argument('--num_workers', type=int, default=2, help='number of worker for dataloader')
    parser.add_argument('--split', type=str, default='val2017', help='what split should be used')

    config = parser.parse_args()

    evaluator = GoogleCloudVisionEvaluator(config)
    evaluator.run_evaluation()
