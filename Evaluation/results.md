# Object Detection (MS COCO 2017)

## YOLOv8

```sh
py .\YoloEvaluator.py --split val2017 --path2model "yolov8m.pt"
```

```sh
py .\YoloEvaluator.py --split val2017 --path2model "yolov8l.pt"
```

```sh
py .\YoloEvaluator.py --split val2017 --path2model "yolov8x.pt"
```

## Azure AI Vision

```sh
py .\AzureAiVisionEvaluator.py --split val2017 --vision_endpoint "..." --vision_key "..."
```

## Google Cloud Vision

```sh
py .\GoogleCloudVisionEvaluator.py --split val2017
```

## Table

### Valid

| Model               | AP@[IoU=0.50:0.95] | AP@[IoU=0.50] | AP@[IoU=0.75] |
|:--------------------|-------------------:|--------------:|--------------:|
| YOLOv8m             |               44.9 |          58.4 |          49.1 |
| YOLOv8l             |               47.1 |          60.6 |          51.4 |
| YOLOv8x             |               48.2 |          61.9 |          52.7 |
| Azure Ai Vision     |               16.0 |          25.8 |          17.0 |
| Google Cloud Vision |               17.0 |          24.0 |          18.9 |

# Image Captioning (COCO Karpahy split)

## Grit (valid):

```sh
py .\GritEvaluator.py --split valid
```

[//]: # (```py)

[//]: # ({'BLEU': [0.8207786842920414, 0.6709795848687291, 0.5252838463437185, 0.4024178194516863], 'METEOR': 0.2983502667973255, 'ROUGE': 0.5944741975399597, 'CIDEr': 1.3574313593264613})

[//]: # (```)

## Grit+ (valid):

```sh
py .\GritEvaluator.py --path2model "../../grit/checkpoint/grit_checkpoint_4ds.pth" --split valid
```

[//]: # (```py)

[//]: # ({'BLEU': [0.8292800536709961, 0.6776099348967689, 0.5326284154980532, 0.40995273497512985], 'METEOR': 0.2976108902062192, 'ROUGE': 0.5957500432884378, 'CIDEr': 1.3794508769003124})

[//]: # (```)

## Grit (test):

```sh
py .\GritEvaluator.py --split test
```

[//]: # (```py)

[//]: # ({'BLEU': [0.8169450876612468, 0.6659656976578642, 0.5221573710371649, 0.40095980016684163], 'METEOR': 0.2982741827366902, 'ROUGE': 0.5915861414935372, 'CIDEr': 1.367754717294518})

[//]: # (```)

## Grit+ (test):

```sh
py .\GritEvaluator.py --path2model "../../grit/checkpoint/grit_checkpoint_4ds.pth" --split test
```

[//]: # (```py)

[//]: # ({'BLEU': [0.8283305015417358, 0.6761726094890992, 0.5331336454457238, 0.41117609451544174], 'METEOR': 0.29898785078984785, 'ROUGE': 0.5965110456829231, 'CIDEr': 1.3950086880355723})

[//]: # (```)

## Magic (valid)

```sh
py .\MagicEvaluator.py --split valid
```

[//]: # (```py)

[//]: # ({'BLEU': [0.5577394941213332, 0.34628724583842835, 0.20507096726390306, 0.12147960168088585], 'METEOR': 0.170627120014616, 'ROUGE': 0.39468176155215434, 'CIDEr': 0.46821273101117733})

[//]: # (```)

## Magic (test)

```sh
py .\MagicEvaluator.py --split test
```

[//]: # (```py)

[//]: # ({'BLEU': [0.5647497270426728, 0.35277803381007805, 0.2093075353646834, 0.12448374645951399], 'METEOR': 0.17309087763290584, 'ROUGE': 0.3957271072395506, 'CIDEr': 0.48350808407410195})

[//]: # (```)

## Azure AI Vision (valid)

```sh
py .\AzureAiVisionEvaluator.py --vision_endpoint "..." --vision_key "..." --split valid
```

[//]: # (```py)

[//]: # ({'BLEU': [0.6931128786372772, 0.5700384964261264, 0.4445366883545693, 0.339177781132265], 'METEOR': 0.26383743695540124, 'ROUGE': 0.5732704646566703, 'CIDEr': 1.1355119410689063})

[//]: # (```)

## Azure AI Vision (test)

```sh
py .\AzureAiVisionEvaluator.py --vision_endpoint "..." --vision_key "..." --split test
```

[//]: # (```py)

[//]: # ({'BLEU': [0.6906055678257669, 0.5646801185545512, 0.43767921230405077, 0.32962501203544553], 'METEOR': 0.26174652217210986, 'ROUGE': 0.5705673328266736, 'CIDEr': 1.133344394462253})

[//]: # (```)

## Table

### Valid
| Model           | BLEU@1 | BLEU@2 | BLEU@3 | BLEU@4 | METEOR | ROUGE | CIDEr |
|:----------------|-------:|-------:|-------:|-------:|-------:|------:|------:|
| Grit            |   82.1 |   67.1 |   52.5 |   40.2 |   29.9 |  59.4 | 135.7 |
| Grit+           |   82.9 |   67.8 |   53.3 |   41.0 |   29.8 |  59.6 | 137.9 |
| Magic           |   55.8 |   34.6 |   20.5 |   12.1 |   17.1 |  39.5 |  46.8 |
| Azure Ai Vision |   69.3 |   57.0 |   44.4 |   33.9 |   26.4 |  57.3 | 113.6 |

### Test
| Model           | BLEU@1 | BLEU@2 | BLEU@3 | BLEU@4 | METEOR | ROUGE | CIDEr |
|:----------------|-------:|-------:|-------:|-------:|-------:|------:|------:|
| Grit            |   81.7 |   66.6 |   52.2 |   40.1 |   29.8 |  59.2 | 136.8 |
| Grit+           |   82.8 |   67.6 |   53.3 |   41.1 |   29.9 |  59.7 | 139.5 |
| Magic           |   56.5 |   35.2 |   20.9 |   12.4 |   17.3 |  39.6 |  48.4 |
| Azure Ai Vision |   69.1 |   56.5 |   43.8 |   33.0 |   26.2 |  57.1 | 113.3 |
