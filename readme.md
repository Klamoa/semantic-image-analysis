# State-of-the-art in semantic image analysis: from the visual scene to the comprehensive textual description

## Setup MS COCO dataset

```
root
├── ms-coco
│   ├── annotations
│   ├── val2014
│   └── val2017
```

1. Make folder structure as shown above
2. Download:
   * ``2014 Val images [41K/6GB]`` and ``2014 Train/Val annotations [241MB]`` for image captioning
   * ``2017 Val images [5K/1GB]`` and ``2017 Train/Val annotations [241MB]`` for object detection
3. Put the files (images and JSON) in to the correct directory under ``.\ms-coco``

## Setup Python 3

1. Python version used 3.11
2. Run ``python install -r requirements.txt`` to install all requirements including GRIT and MAGIC
3. Run ``python -m spacy download en`` (https://github.com/davidnvq/grit?tab=readme-ov-file#installation)
4. Install the required CUDA version from https://pytorch.org/ (tested with CUDA 11.8)

## Image Captioning Evaluation

### Running AzureAiVisionImgCapEvaluator

To run ``AzureAiVisionImgCapEvaluator.py`` one need to do following steps:

1. Create an account for Azure
2. Create a resource
3. Run the command found in ``results.md`` from the ``.\Evaluation\ImageCaptioning`` directory with your ``endpoint`` and ``key``

### Running GritImgCapEvaluator

To run ``GritImgCapEvaluator.py`` one need to do following steps:

1. Clone the repository from https://github.com/davidnvq/grit.git
2. To install Deformable Attention:
   * Change to ``cd grit/models/ops``
   * Run ``python setup.py build develop`` to install the Deformable Attention
   * Check success with ``python test.py``
3. Download a checkpoint from the model zoo found in https://github.com/davidnvq/grit and place the checkpoint into the folder ``checkpoint``
4. Change ``ann_root``, ``img_root``, ``hdf_path`` and ``vocab_path`` to the location of the dataset. The file can be found in ``.\configs\caption\coco_config.yaml``
5. Run the command found in ``results.md`` from the ``.\Evaluation\ImageCaptioning`` directory

### Running MagicImgCapEvaluator

To run ``MagicImgCapEvaluator.py`` one need to do following steps:

1. Clone the repository from https://github.com/yxuansu/MAGIC.git
2. Make two changes in ``.\image_captioning\language_model\simctg.py``
   * Change line 13 to ``from .loss_func import contrastive_loss``
   * Change line 140 to ``from .utlis import PlugAndPlayContrastiveDecodingOneStepFast``
3. Run the command found in ``results.md`` from the ``.\Evaluation\ImageCaptioning`` directory

## Object Detection Evaluation

### Running AzureAiVisionObjDetEvaluator

To run ``AzureAiVisionObjDetEvaluator.py`` one need to do following steps:

1. Create an account for Azure
2. Create a resource
3. Run the command found in ``results.md`` from the ``.\Evaluation\ObjectDetection`` directory with your ``endpoint`` and ``key``

### Running GoogleCloudVisionObjDetEvaluator

To run ``GoogleCloudVisionObjDetEvaluator.py`` one need to do following steps:

1. Create an account for Google Cloud
2. Follow the steps from Google to create a project and activate it
3. Run the command found in ``results.md`` from the ``.\Evaluation\ObjectDetection`` directory

### Running YoloObjDetEvaluator

To run ``YoloObjDetEvaluator.py`` one need to do following steps:

1. Run the command found in ``results.md`` from the ``.\Evaluation\ObjectDetection`` directory
   

## Inference

1. Run ``py .\ImageAnalyser.py --image ".." --analyser ".." [--print-result] [--show-image] [--save-image]`` from the ``.\Inference`` directory
2. Available analysers are ``yolo``, ``azure``, ``google``, ``grit``, ``magic``, ``cocoVal2014`` and ``cocoVal2017`` 

## Acknowledgement

Uses code from https://github.com/yxuansu/MAGIC.git and https://github.com/davidnvq/grit.git.