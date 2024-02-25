import argparse

import yaml
from src import (YoloInference, AzureAiVisionInference, GoogleCloudVisionInference,
                 MagicInference, GritInference,
                 CocoVal2017, CocoVal2014)

from PIL import Image, ImageDraw, ImageFont


def draw_image(image_path: str, result: dict) -> Image:
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    if result.get("objects", None) is not None:
        # Draw rectangles and labels
        for item in result["objects"]:
            cls_name = item["cls_name"]
            score = item["score"]
            xyxy = item["xyxy"]

            # Draw rectangle
            draw.rectangle(xyxy, outline="red", width=2)

            # Draw label and score
            label = f"{cls_name}: {score:.2f}"

            x = xyxy[0]
            y = xyxy[1] - 15
            if y < 0:
                y = xyxy[1] + 5
                x = xyxy[0] + 5

            draw.text((x, y), label, fill="red", font=font)

        return image


def print_result(result: dict):
    objects = result.get("objects", None)
    if objects is not None:
        print(f"\nObjects:")
        print(f"{len(objects)} Object(s) found.")
        for i, obj in enumerate(objects):
            print(f"#{i:02d} | Name: {obj['cls_name']}, Score: {obj['score']:.2f}, Bbox: ({obj['xyxy'][0]:.2f}, {obj['xyxy'][1]:.2f}, {obj['xyxy'][2]:.2f}, {obj['xyxy'][3]:.2f})")

    caption = result.get("caption", None)
    if caption is not None:
        print(f"\nCaption:")
        print(f"{caption}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=str, default="", help="image to predict")
    parser.add_argument("--analyser", required=True, type=str, default="yolo", help="analyser to run the given image on. Options are yolo, azure, google, grit, magic, cocoVal2014, cocoVal2017")
    parser.add_argument("--print-result", dest="print_result", action="store_true", help="whether to print result or not")
    parser.add_argument("--show-image", dest="show_image", action="store_true", help="whether to show image or not")
    parser.add_argument("--save-image", dest="save_image", action="store_true", help="whether to save image or not")
    options = parser.parse_args()

    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    match options.analyser:
        case "yolo":
            model = YoloInference(config["yolo"])
        case "azure":
            model = AzureAiVisionInference(config["azure"])
        case "google":
            model = GoogleCloudVisionInference(config["google"])
        case "grit":
            model = GritInference(config["grit"])
        case "magic":
            model = MagicInference(config["magic"])
        case "cocoVal2014":
            model = CocoVal2014(config["cocoVal2014"])
        case "cocoVal2017":
            model = CocoVal2017(config["cocoVal2017"])
        case _:
            raise Exception("No valid analyser found")

    result = model.run_inference(options.image)

    if options.print_result:
        print_result(result)

    if options.show_image or options.save_image:
        image = draw_image(options.image, result)

        if options.show_image:
            image.show()

        if options.save_image:
            image.save("image.png")
