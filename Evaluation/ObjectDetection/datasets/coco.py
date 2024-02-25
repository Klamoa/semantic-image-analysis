import os
from torch.utils.data import Dataset


class COCODataset(Dataset):
    def __init__(self, coco=None, images_path=""):
        self.__img_ids = coco.getImgIds()
        self.__paths = [os.path.join(images_path, img['file_name']) for img in coco.loadImgs(self.__img_ids)]

    def __len__(self):
        return len(self.__img_ids)

    def __getitem__(self, idx):
        return {'id': self.__img_ids[idx], 'path': self.__paths[idx]}
