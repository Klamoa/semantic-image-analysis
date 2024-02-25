import sys
import torch

from overrides import override
from PIL import Image

from .InferenceInterface import InferenceInterface
from .utils import *

sys.path.append('../MAGIC')
from image_captioning.clip.clip import CLIP
from image_captioning.language_model.simctg import SimCTG


class MagicInference(InferenceInterface):
    __config = None
    __device = None

    __generation_model = None
    __clip = None

    __input_ids = None
    __k = None
    __alpha = None
    __beta = None
    __decoding_len = None

    def __init__(self, config):
        self.__config = config

        disbalePrint() # disable printing
        # set device
        self.__device = torch.device(f"cuda:{0}")
        torch.cuda.set_device(0)

        # set parameter of magic
        self.__k = self.__config["k"]
        self.__alpha = self.__config["alpha"]
        self.__beta = self.__config["beta"]
        self.__decoding_len = self.__config["decoding_len"]

        # load language model
        sos_token, pad_token = r'<-start_of_text->', r'<-pad->'
        self.__generation_model = SimCTG(self.__config["path2language"], sos_token, pad_token).to(self.__device)
        self.__generation_model.eval()

        # load image model
        self.__clip = CLIP(self.__config["path2clip"]).to(self.__device)
        self.__clip.eval()

        # setup input ids
        start_token = self.__generation_model.tokenizer.tokenize(sos_token)
        start_token_id = self.__generation_model.tokenizer.convert_tokens_to_ids(start_token)
        self.__input_ids = torch.LongTensor(start_token_id).view(1, -1).to(self.__device)
        enablePrint() # enable printing

    @override
    def _preprocessing(self, image_path: str):
        return Image.open(image_path).convert('RGB')

    @override
    def _predicting(self, image):
        disbalePrint()
        out = self.__generation_model.magic_search(
            self.__input_ids,
            self.__k,
            self.__alpha,
            self.__decoding_len,
            self.__beta,
            image,
            self.__clip,
            60)
        enablePrint()
        return out

    @override
    def _postprocessing(self, result):
        return {
            "caption": result
        }
