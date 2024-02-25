import sys
import time

from overrides import override

import torch
# import torch.distributed as dist
from tqdm import tqdm
import itertools

from datasets import metrics

sys.path.append('../')
from EvaluatorInterface import EvaluatorInterface


class ImageCaptionEvaluator(EvaluatorInterface):

    _config = None
    _device = None
    _dataloaders = None

    def __init__(self, config=None, seed=42):
        self._config = config

        # set seed
        torch.manual_seed(seed)

    @override
    def setup_model(self):
        # get device
        self._device = torch.device(f"cuda:{0}")
        torch.cuda.set_device(0)

    @override
    def evaluate_metrics(self):
        gen, gts = {}, {}

        counter = 0
        times = []
        with tqdm(desc=f'Evaluation on {self._config.split}',
                  unit='it', total=len(self._dataloaders[f'{self._config.split}_dict'])) as pbar:

            for it, batch in enumerate(iter(self._dataloaders[f'{self._config.split}_dict'])):
                counter += 1
                start_it = time.time()
                bs = len(batch['samples'])

                # run the evaluation on a batch, needs to be changed according to the model
                result = self.run_batch(batch)

                end_it = time.time()
                times.append(end_it - start_it)

                if it > 0 and it % 100 == 0:
                    print(
                        f"\nNumber of iterations: {counter}, batch_size={bs}, Total time per 1 batch: {sum(times) / counter:0.5f}s"
                    )

                for i, (gts_i, gen_i) in enumerate(result):
                    gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                    gen[f'{it}_{i}'] = [gen_i]
                    gts[f'{it}_{i}'] = gts_i
                pbar.update()

        avg_time = sum(times) / counter
        print(f"Iters: {counter}\nTotal time per 1 batch: {avg_time:0.5f}s")
        gts = metrics.PTBTokenizer.tokenize(gts)
        gen = metrics.PTBTokenizer.tokenize(gen)
        scores, _ = metrics.compute_scores(gts, gen)
        print(f'{self._config.split} scores: ' + str(scores) + '\n')
