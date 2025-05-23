import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


class ModelScoreTracker:

    def __init__(self):
        self.train_best_scores = []
        self.eval_accrs = []
        self.conditions = []
        self.best_params = []
        self.best_score = -1
        self.best_model = None

    def aggregate(self, condition: float, gs_cv: GridSearchCV, eval_score) -> None:
        if gs_cv.best_score_ > self.best_score:
            self.best_score = gs_cv.best_score_
            self.best_model = gs_cv.best_estimator_
        self.best_params.append(gs_cv.best_params_)
        self.train_best_scores.append(gs_cv.best_score_)
        self.eval_accrs.append(eval_score)
        self.conditions.append(condition)
    
    def presentation(self, exp_name: str):
        plt.figure(figsize=(8, 5))
        plt.plot(self.conditions, self.train_best_scores, marker='o', label="train best")
        plt.plot(self.conditions, self.eval_accrs, marker='x', label="evaluation")
        plt.xlabel(exp_name)
        plt.ylabel("accuracy")
        plt.title(f"k-NN Accuracy vs {exp_name}")
        plt.legend()
        plt.show()

    def log_results(self):
        for i in range(len(self.train_best_scores)):
            ts = self.train_best_scores[i]
            es = self.eval_accrs[i]
            print(f"条件{self.conditions[i]} --> BestSetting({self.best_params[i]}) & Score(学習:{ts}, 検証:{es})")