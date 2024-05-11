import json
import os
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset
import numpy as np
import polars as pl

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import motorbike_project as mp


class MotorBikeDataset(Dataset):
    def __init__(self, config_path: str, session: str = 'train', labels_csv_path: str = '', folder_path: str = ''):
        """
            In this class, there are two data mode to choose from:
            - `csv`: You need to provide a folder containing the images, and a csv file containing the labels

            - `ssl`: In this mode, you just need to import list of folder paths, which are divided into classes already

        Args:
            `config_path` (str): The path to the config file
            `session` (str, optional): The session of the dataset, must be in [`train`, `val`, `test`]
            `labels_csv_path` (str, optional): The path to the csv file containing the labels
            `folder_path` (str, optional): The path to the folder containing the images

        """

        self.session = session
        self.config_path = config_path
        self.transform = mp.Transform(session=session)
        self.labels_csv_path = labels_csv_path
        self.folder_path = folder_path

        if not os.path.exists(config_path):
            raise ValueError(f'Config path {config_path} does not exist')

        with open(os.path.join(self.config_path, 'class.json'), 'r') as f:
            self.config_class: dict = json.load(f)

        self.load_dataset()

    def _get_label(self, img, labels):
        # 2 is the max label, others will be downsampled to 2
        try:
            label = min(labels[labels['file name'] == img]['answer'].values[0], 3)
        except:
            label = 1
        return label

    def load_dataset(self):
        self.labels = {}

        if self.session == 'train':
            img_path = os.path.join(self.folder_path, 'train', 'images')
        elif self.session == 'val':
            img_path = os.path.join(self.folder_path, 'valid', 'images')
        else:
            img_path = os.path.join(self.folder_path, 'test', 'images')

        # Read the csv file
        labels = pl.read_csv(self.labels_csv_path).to_pandas()
        dirs = tuple(os.listdir(img_path))
        futures = {}

        with ThreadPoolExecutor(max_workers=100) as executor:
            print('Start processing images')
            for idx, img in enumerate(dirs):
                print(f'{idx:>6}|{len(dirs):<6} - Submitting {img}', end='\r')
                futures[executor.submit(self._get_label, img, labels)] = img

            print()
            print('Start getting results')
            print()
            for idx, future in enumerate((as_completed(futures))):
                label = future.result()
                img = futures[future]
                print(f'{idx:>6}|{len(dirs):<6} - Processing {img} - {label}', end='\r')
                self.labels[os.path.join(img_path, img)] = label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_path = list(self.labels.keys())[index]
        label = self.labels[img_path]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img_np = np.array(img)
            img = self.transform(img_np)

        return img, label


if __name__ == '__main__':
    train_dataset = MotorBikeDataset(
        config_path=r'C:\Users\QUANPC\Documents\GitHub\Motocycle-Detection-BKAI\src\motorbike_project\config',
        session='train',
        labels_csv_path=r'D:\Data Deep Learning\FINAL-DATASET\FINAL-DATASET\result.csv',
        folder_path=r'D:\Data Deep Learning\FINAL-DATASET\FINAL-DATASET\train'
    )

    print(train_dataset[0])
