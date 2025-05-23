from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from typing import Iterator
from dataset import ImageDataset, CIFAR10Loader
from .pca_image_compressor import PCAImageCompressor


class Experiment(ABC):
    """ 課題の条件を満たすデータセットを提供する抽象クラス
        すべての課題は、入力となるデータセットの状態を変化させてモデルの精度変化を確認する
        という共通のフレームで捉えることができる。このクラスを継承する以下の3クラスは
        データセットの状態を変化させる役割を担う。
    """
    def __init__(self, name: str, conditions: list):
        self.name = name
        self.conditions = conditions

    @abstractmethod
    def __iter__(self) -> Iterator[ImageDataset]:
        pass


class TrainSizeExperiment(Experiment):
    """ 課題1: 学習データ数を変更したときのモデルの精度変化を確認する """

    def __init__(self, dataset: ImageDataset, train_ratios: list[float], seed: int):
        super().__init__("Change in training data size", train_ratios)
        self.dataset = dataset
        self.seed = seed

    def __iter__(self) -> Iterator[ImageDataset]:
        for ratio in self.conditions:
            if not (0. < ratio < 1.):
                raise RuntimeError(f"Please set a 'size' parameter value between 0.0 and 1.0.")
            
            images_sub, _, labels_sub, _ = train_test_split(
                self.dataset.images,
                self.dataset.labels,
                train_size=ratio,
                stratify=self.dataset.labels,
                random_state=self.seed
            )
            print(f"訓練データ {len(images_sub)}個({int(ratio * 100)}%) で学習")
            yield ImageDataset(
                images=images_sub,
                labels=labels_sub, 
                name=self.dataset.name,
                condition=ratio,
                reducer=self.dataset.reducer,
                do_shuffle=False
            )


class NumClassExperiment(Experiment):
    """ 課題2: クラス数を変更したときのモデルの精度変化を確認する """

    def __init__(self, loader: CIFAR10Loader, class_groups: list[list[str]], seed: int, dims=200, ratio=0.5):
        super().__init__("Change in number of classes", class_groups)
        self.loader = loader
        self.ratio = ratio
        self.dims = dims
        self.seed = seed

    def __iter__(self) -> Iterator[ImageDataset]:
        for class_names in self.conditions:
            print(f"クラス数 {len(class_names)} 個で学習")
            dataset, _, _ = self.loader.get_datasubsets(class_names)
            cmp = PCAImageCompressor(n_components=self.dims)
            dataset.set_reducer(cmp.model)
            dataset.images = cmp.fit_transform(dataset)
            dataset.condition = len(class_names)
            yield dataset


class FeatureCompressExperiment(Experiment):
    """ 課題3: 特徴量数を変更したときのモデルの精度変化を確認する """

    def __init__(self, dataset: ImageDataset, dims: list):
        super().__init__("Change in number of features", dims)
        self.dataset = dataset

    def __iter__(self) -> Iterator[ImageDataset]:
        for dim in self.conditions:
            cmp = PCAImageCompressor(n_components=dim)
            reduced = cmp.fit_transform(self.dataset)
            yield ImageDataset(reduced, self.dataset.labels, "PCA", dim, cmp.model)

