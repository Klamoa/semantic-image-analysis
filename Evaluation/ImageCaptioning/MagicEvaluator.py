import os
import sys
import argparse

from overrides import override

import torch

from ImageCaptionEvaluator import ImageCaptionEvaluator

from datasets.field import ImageField, TextField
from datasets import build_coco_dataloaders

sys.path.append('../../MAGIC')
from image_captioning.clip.clip import CLIP
from image_captioning.language_model.simctg import SimCTG


class MagicEvaluator(ImageCaptionEvaluator):

    __generation_model = None
    __clip = None
    __text_field = None

    __input_ids = None
    __k = 45
    __alpha = 0.1
    __beta = 2.0
    __decoding_len = 16

    def __init__(self, config, seed=42):
        super().__init__(config, seed)

    @override
    def setup_model(self):
        super().setup_model()

        # load language model
        sos_token, pad_token = r'<-start_of_text->', r'<-pad->'
        self.__generation_model = SimCTG(self._config.path2language, sos_token, pad_token).to(self._device)
        self.__generation_model.eval()

        # load image model
        self.__clip = CLIP(self._config.path2clip).to(self._device)
        self.__clip.eval()

        # setup input ids
        start_token = self.__generation_model.tokenizer.tokenize(sos_token)
        start_token_id = self.__generation_model.tokenizer.convert_tokens_to_ids(start_token)
        self.__input_ids = torch.LongTensor(start_token_id).view(1, -1).to(self._device)

    @override
    def setup_dataloader(self):
        # get text field
        self.__text_field = TextField(use_vocab=False, build_vocab=True)

        # build coco dataloaders
        self._dataloaders = build_coco_dataloaders(
            image_field=ImageField(),
            config=self._config,
            mode='finetune',
            device=self._device
        )

    @override
    def run_batch(self, batch):
        caps_gen = []
        for i, image_instance in enumerate(batch['samples']):
            out = self.__generation_model.magic_search(
                self.__input_ids,
                self.__k,
                self.__alpha,
                self.__decoding_len,
                self.__beta,
                image_instance,
                self.__clip,
                60)

            # split into words and remove punctuations
            tokens = self.__text_field.preprocess(out)
            caps_gen.append((batch['captions'][i], tokens))

        return caps_gen


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path2language', type=str, default='cambridgeltl/magic_mscoco', help='path to language model')
    parser.add_argument('--path2clip', type=str, default='openai/clip-vit-base-patch32', help='path to clip model')
    parser.add_argument('--path2img_root', type=str, default='../../ms-coco/', help='path to image root')
    parser.add_argument('--path2ann_root', type=str, default='../../ms-coco/annotations/', help='path to annotation root')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size number for dataloader')
    parser.add_argument('--num_workers', type=int, default=2, help='number of worker for dataloader')
    parser.add_argument('--split', type=str, default='valid', help='what split should be used')

    config = parser.parse_args()

    evaluator = MagicEvaluator(config)
    evaluator.run_evaluation()
