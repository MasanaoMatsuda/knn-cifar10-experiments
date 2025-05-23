import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from dataset import ImageDataset


class PCAImageCompressor:
    " 画像データにPCAを適用して特徴量次元を圧縮・可視化するユーティリティクラス "

    def __init__(self, n_components):
        self.model = PCA(n_components=n_components)
        self.__fitted = False

    def fit_transform(self, dset: ImageDataset):
        p_components = self.model.fit_transform(dset.flat_images)
        self.__fitted = True
        return p_components
    
    def transform(self, dset: ImageDataset):
        if self.__fitted:
            return self.model.transform(dset.flat_images)
        raise RuntimeError("モデルのフィッティングをしていません。fit_transform() を先に呼び出してください。")
    
    def evaluate(self, dset: ImageDataset):
        return self.model.transform(dset.flat_images)

    def plot_cummulative_contribution_ratio(self):
        if not self.__fitted:
            raise RuntimeError("モデルのフィッティングをしていません。fit_transform() を先に呼び出してください。")
        ratios = self.model.explained_variance_ratio_
        cum_ratios = np.cumsum(ratios)
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(cum_ratios)+1), cum_ratios, color="black", marker='o')
        plt.xlabel("N-principal components")
        plt.ylabel("Cummulative Contribution Ratio")
        plt.axhline(0.9, color='red', linestyle='--', label="90%")
        plt.axhline(0.8, color='blue', linestyle='--', label="80%")
        plt.grid(True)
        plt.legend()
        plt.show()