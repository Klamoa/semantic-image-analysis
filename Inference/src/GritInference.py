import sys
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from overrides import override
from PIL import Image
from omegaconf import OmegaConf

from .InferenceInterface import InferenceInterface

sys.path.append('../grit')
from datasets.caption.transforms import get_transform
from datasets.caption.field import TextField
from models.caption import Transformer
from models.caption.detector import build_detector
from engine.utils import nested_tensor_from_tensor_list


class GritInference(InferenceInterface):
    __config = None
    __device = None

    __hydra_config = None

    __model = None
    __text_field = None

    def __init__(self, config):
        self.__config = config

        # set device
        self.__device = torch.device(f"cuda:{0}")
        torch.cuda.set_device(0)

        # load hydra config
        self.__hydra_config = OmegaConf.load(self.__config["path2config"])

        # get text field
        self.__text_field = TextField(vocab_path=self.__config["path2vocab"])

        # extract reg features + initial grid features
        detector = build_detector(self.__hydra_config).to(self.__device)

        self.__model = Transformer(detector=detector, config=self.__hydra_config)
        self.__model.load_state_dict(torch.load(self.__config["path2model"])['state_dict'], strict=False)
        self.__model = self.__model.to(self.__device)

    @override
    def _preprocessing(self, image_path: str):
        transform = get_transform(self.__hydra_config.dataset.transform_cfg)['valid']

        img = Image.open(image_path).convert('RGB')
        img = transform(img)
        img = nested_tensor_from_tensor_list([img]).to(self.__device)

        return img

    @override
    def _predicting(self, image):
        with torch.no_grad():
            out, _ = self.__model(
                image,
                seq=None,
                use_beam_search=True,
                max_len=self.__hydra_config.model.beam_len,
                eos_idx=self.__hydra_config.model.eos_idx,
                beam_size=self.__hydra_config.model.beam_size,
                out_size=1,
                return_probs=False,
            )
        caption = self.__text_field.decode(out, join_words=True)[0]
        return caption

    @override
    def _postprocessing(self, result):
        return {
            "caption": result
        }
