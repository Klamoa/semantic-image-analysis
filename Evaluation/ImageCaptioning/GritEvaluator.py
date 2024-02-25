import os
import sys
import argparse
from overrides import override

from omegaconf import OmegaConf

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ImageCaptionEvaluator import ImageCaptionEvaluator

from datasets.transforms import get_transform
from datasets.field import ImageField, TextField
from datasets import build_coco_dataloaders

sys.path.append('../../grit')
from models.caption import Transformer
from models.caption.detector import build_detector
from engine.utils import nested_tensor_from_tensor_list


class GritEvaluator(ImageCaptionEvaluator):

    __model = None
    __text_field = None
    __image_field = None

    __hydra_config = None

    def __init__(self, config, seed=42):
        super().__init__(config, seed)

        # load hydra config
        self.__hydra_config = OmegaConf.load(self._config.path2config)

    @override
    def setup_model(self):
        super().setup_model()

        # extract reg features + initial grid features
        detector = build_detector(self.__hydra_config).to(self._device)

        self.__model = Transformer(detector=detector, config=self.__hydra_config)
        self.__model.load_state_dict(torch.load(self._config.path2model)['state_dict'], strict=False)
        self.__model = self.__model.to(self._device)

    @override
    def setup_dataloader(self):
        # get text field
        self.__text_field = TextField(vocab_path=self.__hydra_config.dataset.vocab_path)

        # get image field
        transform = get_transform(self.__hydra_config.dataset.transform_cfg)
        self.__image_field = ImageField(transform=transform['valid'])

        # build coco dataloaders
        self._dataloaders = build_coco_dataloaders(
            image_field=self.__image_field,
            config=self._config,
            mode='finetune',
            device=self._device
        )

    @override
    def run_batch(self, batch):
        input = nested_tensor_from_tensor_list(batch['samples']).to(self._device)
        with torch.no_grad():
            out, _ = self.__model(
                input,
                seq=None,
                use_beam_search=True,
                max_len=self.__hydra_config.model.beam_len,
                eos_idx=self.__hydra_config.model.eos_idx,
                beam_size=self.__hydra_config.model.beam_size,
                out_size=1,
                return_probs=False,
            )
        torch.cuda.synchronize()

        caps_gen = self.__text_field.decode(out, join_words=False)
        return zip(batch['captions'], caps_gen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path2model', type=str, default='../../grit/checkpoint/grit_checkpoint_vg.pth', help='path to model')
    parser.add_argument('--path2config', type=str, default='../../grit/configs/caption/coco_config.yaml', help='path to config.yaml')
    parser.add_argument('--path2img_root', type=str, default='../../ms-coco/', help='path to image root')
    parser.add_argument('--path2ann_root', type=str, default='../../ms-coco/annotations/', help='path to annotation root')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size number for dataloader')
    parser.add_argument('--num_workers', type=int, default=2, help='number of worker for dataloader')
    parser.add_argument('--split', type=str, default='valid', help='what split should be used')

    config = parser.parse_args()

    evaluator = GritEvaluator(config)
    evaluator.run_evaluation()
