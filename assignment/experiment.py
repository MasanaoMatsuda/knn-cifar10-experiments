from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from typing import Iterator, Tuple
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
    def __iter__(self) -> Iterator[Tuple[ImageDataset, ImageDataset]]:
        pass


class TrainSizeExperiment(Experiment):
    """ 課題1: 学習データ数を変更したときのモデルの精度変化を確認する """

    def __init__(self, train_set: ImageDataset, eval_set: ImageDataset, 
                 train_ratios: list[float], seed: int):
        super().__init__("Change in training data size", train_ratios)
        self.train_set = train_set
        self.eval_set = eval_set
        self.seed = seed

    def __iter__(self) -> Iterator[Tuple[ImageDataset, ImageDataset]]:
        for ratio in self.conditions:
            if not (0. < ratio < 1.):
                raise RuntimeError(f"Please set a 'size' parameter value between 0.0 and 1.0.")
            
            images_sub, _, labels_sub, _ = train_test_split(
                self.train_set.images,
                self.train_set.labels,
                train_size=ratio,
                stratify=self.train_set.labels,
                random_state=self.seed
            )
            train_subset = ImageDataset(
                images=images_sub,
                labels=labels_sub, 
                name=self.train_set.name,
                condition=ratio,
                do_shuffle=False
            )
            print(f"訓練データ {len(images_sub)}個({int(ratio * 100)}%) で学習")
            yield train_subset, self.eval_set


class NumClassExperiment(Experiment):
    """ 課題2: クラス数を変更したときのモデルの精度変化を確認する """

    def __init__(self, loader: CIFAR10Loader, class_groups: list[list[str]], 
                 seed: int, dims: int=200, ratio: float=0.5):
        super().__init__("Change in number of classes", class_groups)
        self.loader = loader
        self.ratio = ratio
        self.dims = dims
        self.seed = seed

    def __iter__(self) -> Iterator[Tuple[ImageDataset, ImageDataset]]:
        for class_names in self.conditions:
            print(f"クラス数 {len(class_names)} 個で学習")
            train_set, eval_set, _ = self.loader.get_datasubsets(class_names)
            cmp = PCAImageCompressor(n_components=self.dims)
            train_set.images = cmp.fit_transform(train_set)
            eval_set.images = cmp.transform(eval_set)
            train_set.condition = len(class_names)
            yield train_set, eval_set


class FeatureCompressExperiment(Experiment):
    """ 課題3: 特徴量数を変更したときのモデルの精度変化を確認する """

    def __init__(self, train_set: ImageDataset, eval_set: ImageDataset, dims: list):
        super().__init__("Change in number of features", dims)
        self.train_set = train_set
        self.eval_set = eval_set

    def __iter__(self) -> Iterator[Tuple[ImageDataset, ImageDataset]]:
        for dim in self.conditions:
            print(f"特徴量数 {dim} 次元で学習")
            cmp = PCAImageCompressor(n_components=dim)
            train_set = ImageDataset(cmp.fit_transform(self.train_set), 
                                     self.train_set.labels, "PCA", dim)
            eval_set = ImageDataset(cmp.transform(self.eval_set),
                                    self.eval_set.labels, "PCA-eval", dim)
            yield train_set, eval_set

