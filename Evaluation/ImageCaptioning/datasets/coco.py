# ------------------------------------------------------------------------
# Modified from Meshed Memory Transformer
# https://github.com/aimagelab/meshed-memory-transformer
# ------------------------------------------------------------------------
import os
# import json
import numpy as np
# from PIL import Image
from tqdm import tqdm

# from .transforms import *
# from .transforms import get_transform
# from .transforms.utils import MinMaxResize
# from .field import ImageField, TextField

from .example import Example
from pycocotools.coco import COCO as pyCOCO
# from engine.utils import nested_tensor_from_tensor_list

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler, BatchSampler

OVERFIT_SIZE = 64


class DictionaryCollator:

    def __init__(self, img_field, device='cpu'):
        self.img_field = img_field
        self.device = device

    def __call__(self, batch):
        imgs = [item[0] for item in batch]
        captions = [item[1] for item in batch]
        image_ids = [item[2] for item in batch]

        outputs = {}
        outputs['samples'] = imgs

        outputs['captions'] = captions
        outputs['image_id'] = image_ids
        return outputs


class PairedCollator(DictionaryCollator):

    def __init__(self, img_field, device='cpu', max_len=54, pad_idx=1, bos_idx=2, eos_idx=3):
        super().__init__(img_field, device)
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

    def __call__(self, batch):
        b = super().__call__(batch)

        # truncate
        captions = [c[:self.max_len] for c in b['captions']]
        max_len = max([len(c) for c in b['captions']])

        padded = []
        for c in captions:
            caption = [self.bos_idx] + c + [self.eos_idx] + [self.pad_idx] * (max_len - len(c))
            padded.append(caption)

        padded = [torch.Tensor(caption).long() for caption in padded]
        padded = pad_sequence(padded, batch_first=True).to(self.device)

        b['captions'] = padded
        return b


class CPairedDataset:

    def __init__(self, examples, image_field, overfit=False):
        self.examples = examples
        self.image_field = image_field
        self.overfit = overfit

    def __getitem__(self, idx):
        example = self.examples[idx]
        img_path, caption = example.image, example.tokens
        image_id = example.image_id
        img = self.image_field.preprocess(img_path)
        return img, caption, image_id

    def __len__(self):
        if self.overfit:
            return OVERFIT_SIZE
        return len(self.examples)


class CDictionaryDataset:

    def __init__(self, examples, image_field, overfit=False):
        self.overfit = overfit
        self.image_field = image_field
        self.img2captions = {}
        self.img2image_id = {}

        for example in examples:
            if example.image not in self.img2captions:
                self.img2captions[example.image] = []
            self.img2captions[example.image].append(example.text)
            self.img2image_id[example.image] = example.image_id
        self.img_paths = list(self.img2captions.keys())

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image_id = self.img2image_id[img_path]
        captions = self.img2captions[img_path]
        img = self.image_field.preprocess(img_path)
        return img, captions, image_id

    def __len__(self):
        if self.overfit:
            return OVERFIT_SIZE
        return len(self.img_paths)


class COCO(CPairedDataset):

    def __init__(
        self,
        image_field,
        text_field,
        img_root,
        ann_root,
        use_restval=True,
        cut_validation=False,
        overfit=False,
        **kwargs,
    ):
        self.image_field = image_field
        self.text_field = text_field
        self.overfit = overfit

        roots = {}
        roots['train'] = {
            'img': os.path.join(img_root, 'train2014'),
            'cap': os.path.join(ann_root, 'captions_train2014.json')
        }
        roots['valid'] = {
            'img': os.path.join(img_root, 'val2014'),
            'cap': os.path.join(ann_root, 'captions_val2014.json')
        }
        roots['test'] = {
            'img': os.path.join(img_root, 'val2014'),
            'cap': os.path.join(ann_root, 'captions_val2014.json')
        }
        roots['trainrestval'] = {
            'img': (roots['train']['img'], roots['valid']['img']),
            'cap': (roots['train']['cap'], roots['valid']['cap'])
        }

        if ann_root is not None:
            ids = {}
            ids['train'] = np.load(os.path.join(ann_root, 'coco_train_ids.npy'))
            ids['valid'] = np.load(os.path.join(ann_root, 'coco_dev_ids.npy'))
            if cut_validation:
                ids['valid'] = ids['valid'][:5000]
            ids['test'] = np.load(os.path.join(ann_root, 'coco_test_ids.npy'))
            ids['trainrestval'] = (ids['train'], np.load(os.path.join(ann_root, 'coco_restval_ids.npy')))

            if use_restval:
                roots['train'] = roots['trainrestval']
                ids['train'] = ids['trainrestval']
        else:
            ids = None

        self.train_examples, self.valid_examples, self.test_examples = self.get_samples(roots, ids)

    def split_examples(self):
        return {
            'train': self.train_examples,
            'valid': self.valid_examples,
            'test': self.test_examples,
        }

    def splits(self):
        train_split = CPairedDataset(self.train_examples, self.image_field, overfit=self.overfit)
        valid_split = CPairedDataset(self.valid_examples, self.image_field, overfit=self.overfit)
        test_split = CPairedDataset(self.test_examples, self.image_field, overfit=self.overfit)
        return train_split, valid_split, test_split

    def get_samples(self, roots, ids_dataset=None):
        train_samples = []
        valid_samples = []
        test_samples = []
        heights = []
        widths = []

        splits = ['valid', 'test'] if self.overfit else ['train', 'valid', 'test']
        for split in splits:
            if isinstance(roots[split]['cap'], tuple):
                coco_dataset = (pyCOCO(roots[split]['cap'][0]), pyCOCO(roots[split]['cap'][1]))
                root = roots[split]['img']
            else:
                coco_dataset = (pyCOCO(roots[split]['cap']),)
                root = (roots[split]['img'],)

            if ids_dataset is None:
                ids = list(coco_dataset.anns.keys())
            else:
                ids = ids_dataset[split]

            if isinstance(ids, tuple):
                bp = len(ids[0])
                ids = list(ids[0]) + list(ids[1])
            else:
                bp = len(ids)
            for index in tqdm(range(len(ids))):
                if index < bp:
                    coco = coco_dataset[0]
                    img_root = root[0]
                else:
                    coco = coco_dataset[1]
                    img_root = root[1]

                ann_id = ids[index]
                caption = coco.anns[ann_id]['caption']
                img_id = coco.anns[ann_id]['image_id']
                filename = coco.loadImgs(img_id)[0]['file_name']
                height = coco.loadImgs(img_id)[0]['height']
                width = coco.loadImgs(img_id)[0]['width']
                heights.append(height)
                widths.append(width)

                example = Example.fromdict({
                    'image_id': img_id,
                    'image': os.path.join(img_root, filename),
                    'text': caption,
                })

                if split == 'train':
                    train_samples.append(example)
                elif split == 'valid':
                    valid_samples.append(example)
                elif split == 'test':
                    test_samples.append(example)

        if self.overfit:
            train_samples = valid_samples
        return train_samples, valid_samples, test_samples


def build_coco_dataloaders(image_field=None, text_field=None, config=None, mode='freezing', device='cpu'):

    examples = COCO(None, text_field, img_root=config.path2img_root, ann_root=config.path2ann_root).split_examples()

    datasets = {
        'valid': CPairedDataset(examples['valid'], image_field, overfit=False),
        'valid_dict': CDictionaryDataset(examples['valid'], image_field, overfit=False),
        'test_dict': CDictionaryDataset(examples['test'], image_field, overfit=False),
    }

    collators = {
        'valid': PairedCollator(image_field, device=device),
        'valid_dict': DictionaryCollator(image_field, device=device),
        'test_dict': DictionaryCollator(image_field, device=device),
    }

    sc_batch_size = config.batch_size if mode == 'freezing' else config.batch_size // 4

    dataloaders = {}
    # test and valid
    dataloaders['valid_dict'] = DataLoader(
        datasets['valid_dict'],
        batch_size=max(1, sc_batch_size * 2),
        num_workers=config.num_workers,
        collate_fn=collators['valid_dict'],
    )
    dataloaders['test_dict'] = DataLoader(
        datasets['test_dict'],
        batch_size=max(1, sc_batch_size * 2),
        num_workers=config.num_workers,
        collate_fn=collators['test_dict'],
    )

    return dataloaders
