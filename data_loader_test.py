from os import path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


class ImageList(ImageFolder):
    def __init__(self, source, image_list, transform, rgb):
        image_names = pd.read_csv(image_list, delimiter=" ", header=None)
        image_names = np.array(image_names)

        if source is not None:
            self.samples = [
                path.join(source, image_name) for image_name in image_names[:, 0]
            ]
        else:
            self.samples = image_names[:, 0]

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((112, 112)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        else:
            self.transform = transform

        self.rgb = rgb

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # checks if input is RGB or BGR
        if self.rgb:
            img = Image.open(self.samples[index]).convert("RGB")
        else:
            img = cv2.imread(self.samples[index])

        return self.transform(img)


class TestDataLoader(DataLoader):
    def __init__(
        self, batch_size, workers, source, image_list, transform=None, rgb=True
    ):
        self._dataset = ImageList(source, image_list, transform, rgb)

        super(TestDataLoader, self).__init__(
            self._dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=workers,
            drop_last=False,
        )
