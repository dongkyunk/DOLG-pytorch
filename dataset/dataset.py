import os
import jpeg4py as jpeg
import pandas as pd
from sklearn import preprocessing
from torch.utils.data import Dataset
from dataset.transform import image_transform
from config import Config


def img_path_from_id(id):
    img_path = os.path.join(Config.DATA_DIR, 'train',
                            id[0], id[1], id[2], f'{id}.jpg')
    return img_path


class LmkRetrDataset(Dataset):
    def __init__(self):
        self.df = pd.read_csv(Config.CSV_PATH)
        self.landmark_id_encoder = preprocessing.LabelEncoder()
        self.df['landmark_id'] = self.landmark_id_encoder.fit_transform(
            self.df['landmark_id'])
        self.df['path'] = self.df['id'].apply(img_path_from_id)
        self.paths = self.df['path'].values
        self.ids = self.df['id'].values
        self.landmark_ids = self.df['landmark_id'].values
        self.transform = image_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path, id, landmark_id = self.paths[idx], self.ids[idx], self.landmark_ids[idx]
        img = jpeg.JPEG(path).decode()
        if self.transform:
            img = self.transform(image=img)['image']
        return img, landmark_id, id
