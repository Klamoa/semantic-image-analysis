import argparse

from overrides import override
from ultralytics import YOLO

from ObjectDetectionEvaluator import ObjectDetectionEvaluator


class YoloObjDetEvaluator(ObjectDetectionEvaluator):

    __model = None

    def __init__(self, config, seed=42):
        super().__init__(config, seed)

    @override
    def setup_model(self):
        super().setup_model()
        
        # setup model
        self.__model = YOLO(self._config.path2model).to(self._device)

    @override
    def run_batch(self, batch):
        out = self.__model.predict(batch['path'], verbose=False)

        # get result from output
        results = []
        for i, res in enumerate(out):
            for box in res.boxes:
                id = int(batch['id'][i].item()) # id
                coords = box.xyxy[0].tolist() # coordinates in x1, y1, x2, y2
                score = box.conf[0].item() # confidence score
                cls_name = res.names[box.cls[0].item()] # get class name, because yolo uses other ids than coco
                cls_id = self._coco_gt.getCatIds(catNms=[cls_name])[0]

                results.append([
                    id,
                    coords[0], coords[1], coords[2] - coords[0], coords[3] - coords[1],
                    score,
                    cls_id
                ])

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path2model', type=str, default='yolov8m.pt', help='path to model or the name of the model to download')
    parser.add_argument('--path2img_root', type=str, default='../../ms-coco', help='path to image root')
    parser.add_argument('--path2ann_root', type=str, default='../../ms-coco/annotations', help='path to annotation file')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size number for dataloader')
    parser.add_argument('--num_workers', type=int, default=2, help='number of worker for dataloader')
    parser.add_argument('--split', type=str, default='val2017', help='what split should be used')

    config = parser.parse_args()

    evaluator = YoloObjDetEvaluator(config)
    evaluator.run_evaluation()
