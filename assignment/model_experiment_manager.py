from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from dataset import ImageDataset
from .experiment import Experiment
from .model_score_tracker import ModelScoreTracker


class ModelExperimentManager:

    def __init__(self, eval_dset: ImageDataset):
        self.eval_dset = eval_dset

    def run_experiment(self, experiment: Experiment, param: dict) -> ModelScoreTracker:
        tracker = ModelScoreTracker()
        for dset in experiment:
            cv_res = self._grid_search_cv(dset.flat_images, dset.labels, param)
            eval_x = self._evaluation_data(dset)
            y_hat = cv_res.best_estimator_.predict(eval_x)
            acc = accuracy_score(self.eval_dset.labels, y_hat)
            tracker.aggregate(dset.condition, cv_res, acc)
        tracker.presentation(experiment.name)
        return tracker
    
    def _grid_search_cv(self, X, y, param: dict):
        cv = GridSearchCV(
            estimator=KNeighborsClassifier(),
            param_grid=param,
            cv=3,  # クロスバリデーション分割数
            scoring='accuracy',
            n_jobs=-1,       # 全CPUコア使用
            verbose=1        # 実行中に進捗表示
        )
        cv.fit(X, y)
        cv.best_estimator_.predict
        return cv
    
    def _evaluation_data(self, dataset: ImageDataset):
        """ PCAを適用する場合としない場合のどちらにも対応するためのメソッド """
        if dataset.reducer is not None:
            return dataset.reducer.transform(self.eval_dset.flat_images)
        else:
            return self.eval_dset.flat_images