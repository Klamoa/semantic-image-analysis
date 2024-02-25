import os
import sys
import time

import numpy as np
from torch.utils.data import DataLoader

from tqdm import tqdm

from overrides import override

import torch

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from datasets.coco import COCODataset

sys.path.append('../')
from EvaluatorInterface import EvaluatorInterface


class ObjectDetectionEvaluator(EvaluatorInterface):
    _config = None
    _device = None
    _coco_gt = None
    __coco_dt = None
    __dataloader = None

    def __init__(self, config=None, seed=42):
        self._config = config

    @override
    def setup_model(self):
        # get device
        self._device = torch.device(f"cuda:{0}")
        torch.cuda.set_device(0)

    @override
    def setup_dataloader(self):
        self._coco_gt = COCO(os.path.join(self._config.path2ann_root, f"instances_{self._config.split}.json"))
        dataset = COCODataset(self._coco_gt, os.path.join(self._config.path2img_root, self._config.split))
        self.__dataloader = DataLoader(
            dataset=dataset,
            batch_size=self._config.batch_size,
            num_workers=self._config.num_workers,
            shuffle=False
        )

    @override
    def evaluate_metrics(self):
        res = []

        counter = 0
        times = []
        with tqdm(desc=f'Evaluation on {self._config.split}', unit='it', total=len(self.__dataloader)) as pbar:

            for it, batch in enumerate(iter(self.__dataloader)):
                counter += 1
                start_it = time.time()
                bs = len(batch['id'])

                # run the evaluation on a batch, needs to be changed according to the model
                result = self.run_batch(batch)

                end_it = time.time()
                times.append(end_it - start_it)

                if it > 0 and it % 100 == 0:
                    print(
                        f"\nNumber of iterations: {counter}, batch_size={bs}, Total time per 1 batch: {sum(times) / counter:0.5f}s"
                    )

                # append results to res
                for entry in result:
                    res.append(entry)

                pbar.update()

        avg_time = sum(times) / counter
        print(f"Iters: {counter}\nTotal time per 1 batch: {avg_time:0.5f}s")

        # load responses
        self.__coco_dt = self._coco_gt.loadRes(np.array(res))

        # run evaluation
        coco_eval = COCOeval(self._coco_gt, self.__coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
