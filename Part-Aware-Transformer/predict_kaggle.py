import os
import argparse
import numpy as np
import torch
from config import cfg
from data.build_DG_dataloader import build_reid_test_loader
from model import make_model
from utils.logger import setup_logger
from utils.re_ranking import re_ranking


def extract_features(model, dataloaders, num_query, use_flip=False):
    """
    Extract features with optional flipping.

    Args:
        model: The model used for feature extraction.
        dataloaders: DataLoader providing batches of images.
        num_query: Number of query images.
        use_flip: Whether to use flipping during feature extraction.

    Returns:
        qf: Query features.
        gf: Gallery features.
    """
    features = []

    for data in dataloaders:
        img, _, _, _, _ = data.values()
        input_img = img.cuda()

        # Extract features for original images
        outputs_orig = model(input_img)

        if use_flip:
            # Extract features for flipped images
            input_img_flip = torch.flip(input_img, dims=[3])  # Horizontal flip
            outputs_flip = model(input_img_flip)
            ff = (outputs_orig.float() + outputs_flip.float()) / 2
        else:
            ff = outputs_orig.float()

        # Normalize features
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features.append(ff)

    # Concatenate all features
    features = torch.cat(features, 0)

    # Separate query and gallery features
    qf = features[:num_query]
    gf = features[num_query:]
    return qf, gf


def save_submission(indices, image_names, output_path):
    """
    Save the predictions in the required CSV format.

    Args:
        indices: Sorted indices of predictions.
        image_names: List of query image names.
        output_path: Path to save the CSV file.
    """
    with open(output_path, 'w') as f:
        f.write("imageName,Corresponding Indexes\n")
        for i, img_name in enumerate(image_names):
            ranked_list = ' '.join(map(str, indices[i] + 1))  # Convert to 1-based index
            f.write(f"{img_name},{ranked_list}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Kaggle Prediction")
    parser.add_argument(
        "--config_file", default="./config/PAT.yml", help="path to config file", type=str
    )
    parser.add_argument(
        "--output_file", default="submission.csv", help="path to save the output CSV", type=str
    )
    parser.add_argument(
        "--use_flip", action="store_true", help="Enable flipping during feature extraction"
    )
    args = parser.parse_args()

    # Load configuration
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.freeze()

    # Setup logger
    output_dir = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = setup_logger("PAT", output_dir, if_train=False)
    logger.info("Running with config:\n{}".format(cfg))

    # Set device
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # Load model
    model = make_model(cfg, cfg.MODEL.NAME, 0, 0, 0)
    model.load_param(cfg.TEST.WEIGHT)

    # Load data
    for testname in cfg.DATASETS.TEST:
        val_loader, num_query = build_reid_test_loader(cfg, testname)

    # Extract features
    with torch.no_grad():
        qf, gf = extract_features(model, val_loader, num_query, use_flip=args.use_flip)

    # Compute distances
    q_g_dist = np.dot(qf.cpu().numpy(), np.transpose(gf.cpu().numpy()))
    q_q_dist = np.dot(qf.cpu().numpy(), np.transpose(qf.cpu().numpy()))
    g_g_dist = np.dot(gf.cpu().numpy(), np.transpose(gf.cpu().numpy()))

    # Re-ranking
    re_rank_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)

    # Sort indices
    indices = np.argsort(re_rank_dist, axis=1)[:, :100]

    # Get query image names
    image_names = [os.path.basename(data['img_path'][0]) for data in val_loader.dataset.query]

    # Save predictions
    save_submission(indices, image_names, args.output_file)

    print(f"Predictions saved to {args.output_file}")