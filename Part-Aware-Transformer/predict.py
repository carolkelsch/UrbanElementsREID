import os
import argparse
import pandas as pd
import numpy as np
import torch
from config import cfg
from model import make_model
from utils.logger import setup_logger
from PIL import Image
import torchvision.transforms as transforms


def load_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0)


def generate_predictions_from_csv(cfg, model, test_csv_path):
    """
    Generate predictions for images listed in test.csv.
    """
    # Load test.csv
    test_data = pd.read_csv(test_csv_path)
    image_paths = test_data['imageName'].tolist()

    # Extract features
    with torch.no_grad():
        features = []
        for img_path in image_paths:
            img = load_image(img_path)  # Implement this function to load and preprocess the image
            img = img.cuda()
            output = model(img)
            features.append(output.cpu().numpy())

        features = np.vstack(features)

    # Generate predictions (dummy example: use indices as predictions)
    predictions = {}
    for i, img_path in enumerate(image_paths):
        predictions[os.path.basename(img_path)] = " ".join(map(str, range(len(features))))

    # Save to CSV
    submission_path = os.path.join(cfg.LOG_ROOT, "submission.csv")
    with open(submission_path, "w") as f:
        f.write("imageName,Corresponding Indexes\n")
        for img_name, indices in predictions.items():
            f.write(f"{img_name},{indices}\n")

    print(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Prediction")
    parser.add_argument(
        "--config_file", default="./config/PAT.yml", help="path to config file", type=str
    )
    parser.add_argument(
        "--test_csv", default="./data/test.csv", help="path to test.csv file", type=str
    )
    parser.add_argument(
        "opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER
    )
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("PAT", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0)
    model.load_param(cfg.TEST.WEIGHT)

    # Generate predictions for test.csv
    generate_predictions_from_csv(cfg, model, args.test_csv)