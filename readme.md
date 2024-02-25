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

## Image Captioning Evaluation

### Running AzureAiVisionEvaluator

To run ``AzureAiVisionEvaluator.py`` one need to do following steps:

1. Create an account for Azure
2. Create a resource
3. Run the command found in ``results.md`` from the ``.\Evaluation\ImageCaptioning`` directory with your ``endpoint`` and ``key``

### Running GritEvaluator

To run ``GritEvaluator.py`` one need to do following steps:

1. Clone the repository from https://github.com/davidnvq/grit.git
2. Install the requirements as mentioned in the repository
3. Run ``cd .\models\ops`` to change directory
4. Run ``py setup.py build`` and ``py setup.py install`` to install ``MultiScaleDeformableAttention``
5. Download a checkpoint from the model zoo found in https://github.com/davidnvq/grit and place the checkpoint into the folder ``checkpoint``
6. Change ``ann_root``, ``img_root``, ``hdf_path`` and ``vocab_path`` to the location of the dataset. The file can be found in ``.\configs\caption\coco_config.yaml``
7. Run the command found in ``results.md`` from the ``.\Evaluation\ImageCaptioning`` directory

### Running MagicEvaluator

To run ``MagicEvaluator.py`` one need to do following steps:

1. Clone the repository from https://github.com/yxuansu/MAGIC.git
2. Install the requirements as mentioned in the repository
3. Make two changes in ``.\image_captioning\language_model\simctg.py``
   * Change line 13 to ``from .loss_func import contrastive_loss``
   * Change line 140 to ``from .utlis import PlugAndPlayContrastiveDecodingOneStepFast``
4. Run the command found in ``results.md`` from the ``.\Evaluation\ImageCaptioning`` directory

## Object Detection Evaluation

### Running AzureAiVisionEvaluator

To run ``AzureAiVisionEvaluator.py`` one need to do following steps:

1. Create an account for Azure
2. Create a resource
3. Run the command found in ``results.md`` from the ``.\Evaluation\ObjectDetection`` directory with your ``endpoint`` and ``key``

### Running GoogleCloudVisionEvaluator

To run ``GoogleCloudVisionEvaluator.py`` one need to do following steps:

1. Create an account for Google Cloud
2. Follow the steps from Google to create a project and activate it
3. Run the command found in ``results.md`` from the ``.\Evaluation\ObjectDetection`` directory

### Running YoloEvaluator

To run ``YoloEvaluator.py`` one need to do following steps:

1. Run ``pip install ultralytics``
2. Run the command found in ``results.md`` from the ``.\Evaluation\ObjectDetection`` directory
   

## Inference

1. Run ``py .\ImageAnalyser.py --image ".." --analyser ".." [--print-result] [--show-image] [--save-image]`` from the ``.\Inference`` directory

## Acknowledgement

Uses code from https://github.com/yxuansu/MAGIC.git and https://github.com/davidnvq/grit.git.