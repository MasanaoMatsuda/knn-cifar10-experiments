# from sklearn.model_selection import train_test_split
# from dataset import CIFAR10Loader, ImageDataset
# from .pca_image_compressor import PCAImageCompressor
# from typing import Iterator
# from .experiment import TrainSizeExperiment, FeatureCompressExperiment, NumClassExperiment


# class AssignmentDataGenerator:

#     def __init__(self, dset: ImageDataset, dloader: CIFAR10Loader, seed: int):
#         self.dset = dset
#         self.loader = dloader
#         self.seed = seed
    
#     def yield_train_subsets(self, exp: TrainSizeExperiment) -> Iterator[ImageDataset]:
#         for ratio in exp.conditions:
#             print(f"訓練データ {int(ratio * 100)}% で学習")
#             yield  self.__get_diet_data(ratio)

#     def yield_reduced_features(self, feature_dims: list[int]) -> Iterator[ImageDataset]:
#         for dim in feature_dims:
#             reducer = PCAImageCompressor(n_components=dim)
#             reduced_train = reducer.fit_transform(self.dset)
#             yield ImageDataset(reduced_train, self.dset.labels, "component", dim)

#     def yield_class_groups(self, class_groups: list[list[str]], ratio=0.5):
#         for class_names in class_groups:
#             train_set, _, _ = self.loader.get_datasubsets(class_names) # test_set を捨てているが処理が軽いのでこのまま使う
#             self.__get_diet_data(ratio)
#             yield train_set

#     def __get_diet_data(self, train_size: float):
#         if not (0. < train_size < 1.):
#             raise RuntimeError(f"Please set a 'size' parameter value between 0.0 and 1.0.")
#         images_sub, _, labels_sub, _ = train_test_split(
#             self.dset.images,
#             self.dset.labels,
#             train_size=train_size,
#             stratify=self.dset.labels,
#             random_state=self.seed
#         )
#         return ImageDataset(
#             images=images_sub,
#             labels=labels_sub,
#             name=self.dset.name,
#             condition=train_size,
#             do_shuffle=False
#         )