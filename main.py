import argparse
import os

import h5py
from tqdm import tqdm

from config import config
from feature_extractor import FeatureExtractor2D
import nets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video_dpath', type=str, help="The directory path of videos.")
    parser.add_argument('-m', '--model', type=str, help="The name of model from which you extract features.")
    parser.add_argument('-b', '--batch_size', type=int, default=25, help="The batch size.")
    parser.add_argument('-s', '--stride', type=int, default=5, help="Extract feature from every <s> frames.")
    parser.add_argument('-o', '--out', type=str, help="The file path of extracted feature.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    config = config[args.model]

    model = getattr(nets, args.model)(pretrained=True)
    model.cuda()
    model.eval()

    extractor = FeatureExtractor2D(
        stride=args.stride,
        mean=config.mean,
        std=config.std,
        resize_to=config.resize_to,
        crop_to=config.crop_to,
        model=model,
        batch_size=args.batch_size)

    videos = os.listdir(args.video_dpath)
    h5 = h5py.File(args.out, 'w') if not os.path.exists(args.out) else h5py.File(args.out, 'r+')
    for video in tqdm(videos):
        video_id = os.path.splitext(video)[0]
        if video_id in h5.keys():
            continue

        video_fpath = os.path.join(args.video_dpath, video)
        feats = extractor(video_fpath)
        if feats is not None:
            h5[video_id] = feats

    h5.close()

