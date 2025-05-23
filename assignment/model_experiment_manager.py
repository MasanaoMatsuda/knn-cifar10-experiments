from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from .experiment import Experiment
from .model_score_tracker import ModelScoreTracker


class ModelExperimentManager:

    def __init__(self):
        pass

    def run_experiment(self, experiment: Experiment, param: dict) -> ModelScoreTracker:
        tracker = ModelScoreTracker()
        for train_set, eval_set in experiment:
            cv_res = self._grid_search_cv(train_set.flat_images, train_set.labels, param)
            y_hat = cv_res.best_estimator_.predict(eval_set.flat_images)
            acc = accuracy_score(eval_set.labels, y_hat)
            tracker.aggregate(train_set.condition, cv_res, acc)
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
    