"""
Uses weights and models implementation from
https://github.com/IrvingMeng/MagFace
"""

import argparse
import shutil
import sys
from os import makedirs, path

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from data_loader_test import TestDataLoader


class Extractor:
    def __init__(self, args):
        # adds MagFace directory to build model
        sys.path.insert(0, args.magface_dir)
        sys.path.insert(0, path.join(args.magface_dir, "inference"))
        from network_inf import builder_inf

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = builder_inf(args)
        self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(self.device)
        self.model.eval()

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((112, 112)),
                transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
            ]
        )
        self.loader = TestDataLoader(
            batch_size=args.batch_size,
            workers=args.workers,
            source=args.source,
            image_list=args.image_list,
            transform=transform,
            rgb=False,
        )

        if path.exists(args.dest):
            shutil.rmtree(args.dest)
        makedirs(args.dest)

        self.destination = args.dest
        self.batch_size = args.batch_size
        self.image_paths = np.asarray(self.loader.dataset.samples)

    def run(self):
        idx = 0
        with torch.no_grad():
            for imgs in tqdm(self.loader):
                imgs = imgs.to(self.device)

                embeddings = self.model(imgs)
                embeddings = embeddings.cpu().numpy()

                image_paths = self.image_paths[idx : idx + self.batch_size]
                self.save_features(image_paths, embeddings)
                idx += self.batch_size

    def save_features(self, image_paths, embeddings):
        for i in range(0, len(embeddings)):
            image_name = path.split(image_paths[i])[1]
            sub_folder = path.basename(path.normpath(path.split(image_paths[i])[0]))
            dest_path = path.join(self.destination, sub_folder)

            if not path.exists(dest_path):
                makedirs(dest_path)

            features_name = path.join(dest_path, image_name[:-3] + "npy")
            np.save(features_name, embeddings[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features with MagFace")
    parser.add_argument("--image_list", "-i", help="Path to image list.")
    parser.add_argument("--source", "-s", help="Aditional path to append to images.")
    parser.add_argument("--dest", "-d", help="Folder to save the extractions.")
    parser.add_argument("--batch_size", "-b", help="Batch size.", default=250, type=int)
    parser.add_argument("--workers", "-w", help="Workers number.", default=4, type=int)
    parser.add_argument(
        "--magface_dir", "-m", help="Path to MagFace folder.", default="../../MagFace/"
    )

    # MagFace params
    parser.add_argument(
        "--arch", default="iresnet100", type=str, help="backbone architechture"
    )
    parser.add_argument(
        "--embedding_size", default=512, type=int, help="The embedding feature size"
    )
    parser.add_argument(
        "--resume",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint",
        default="../../MagFace/pretrained/magface_epoch_00025.pth",
    )
    parser.add_argument("--cpu-mode", action="store_true", help="Use the CPU.")
    parser.add_argument(
        "--dist", default=1, help="use this if model is trained with dist"
    )

    args = parser.parse_args()

    extractor = Extractor(args)
    extractor.run()
