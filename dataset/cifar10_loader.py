from sklearn.model_selection import train_test_split
import keras
import numpy as np
from .image_dataset import ImageDataset
from .dataset_split import DatasetSplit
from typing import NamedTuple


class DatasetSplitTuple(NamedTuple):
    train: tuple[np.ndarray, np.ndarray]
    eval: tuple[np.ndarray, np.ndarray]
    test: tuple[np.ndarray, np.ndarray]


class CIFAR10Loader:

    MAP_LABEL_ID = {
        'airplane': 0,
        'automobile': 1,
        'bird': 2,
        'cat': 3,
        'deer': 4,
        'dog': 5,
        'frog': 6,
        'horse': 7,
        'ship': 8,
        'truck': 9
    }

    def __init__(self, seed, eval_size=0.2):
        train, eval, test = self._load_data(eval_size, seed)
        self.train = ImageDataset(*train, "Original Trainset", 1.0)
        self.eval = ImageDataset(*eval, "Original Evalset", 1.0)
        self.test = ImageDataset(*test, "Original Testset", 1.0)
        print("CFAR10データセットはTrain用が50000枚、Test用が10000枚の画像データからなるセットです。ダウンロードできました。")
        print("Train用の2割をEvalation用にします。つまり (Train, Evaluation, Test) は (40000, 10000, 10000) の枚数からなります。")
        

    @property
    def _dataset_split_map(self):
        return {
            DatasetSplit.TRAIN: self.train,
            DatasetSplit.EVAL: self.eval,
            DatasetSplit.TEST: self.test
        }
    

    def get_datasubsets(self, class_names: list, verbose: bool=False) -> tuple[ImageDataset, ImageDataset, ImageDataset]:
        return (
            self.extract_class_subset_by_split(DatasetSplit.TRAIN, class_names, verbose)
            , self.extract_class_subset_by_split(DatasetSplit.EVAL, class_names, verbose)
            , self.extract_class_subset_by_split(DatasetSplit.TEST, class_names, verbose)
        )
    

    def extract_class_subset_by_split(self, split: DatasetSplit, class_names: list, verbose: bool=False) -> ImageDataset:
        if split not in self._dataset_split_map:
            raise ValueError(f"Invalid DatasetType: {split}")
        full_dataset = self._dataset_split_map[split]
        return self._extract_class_subset(full_dataset, split.value, class_names, verbose)


    def _extract_class_subset(self, full_dataset: ImageDataset, subset_name: str, class_names: list, verbose) -> ImageDataset:
        assert len(class_names) > 0
        if verbose:
            print(f"{full_dataset.name} -> ラベル{class_names}が付与された画像データを抽出します。")
        x, y = [], []
        for name in class_names:
            img, label = self._extract_class_data(full_dataset, name)
            x.append(img)
            y.append(label)
        return ImageDataset(np.vstack(x), np.hstack(y), f"{subset_name}{class_names}", True)


    def _extract_class_data(self, dataset: ImageDataset, class_name: str):
        """ class_name で指定されたクラスに該当するデータをデータセットから取り出して返す。
            CIFAR10データセットは、各クラス5000枚の画像があるので、5000枚の画像データが
            この関数によって返される。
        """
        if class_name not in self.MAP_LABEL_ID:
            raise ValueError(f"Unknown class name: {class_name}")
        label_id = self.MAP_LABEL_ID[class_name]
        idx = np.where(dataset.labels==label_id)[0]
        label_id = self.MAP_LABEL_ID[class_name]
        imgs = dataset.images[idx]
        labels = np.full((len(imgs), ), label_id, dtype=int)
        return imgs, labels


    def _load_data(self, eval_size, seed) -> DatasetSplitTuple:
        train, test = keras.datasets.cifar10.load_data()
        train_X, eval_X, train_y, eval_y = train_test_split(
            train[0], train[1], test_size=eval_size, stratify=train[1], random_state=seed)
        return DatasetSplitTuple((train_X, train_y), (eval_X, eval_y), test)