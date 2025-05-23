
# K-Nearest Neighbors Experiment Framework

画像分類における k-NN モデルの性能を3条件下で評価しています。このプロジェクトのモジュール群は、その評価フレームワークを提供します。

---

## 📁ディレクトリ構成と設計思想

```
k_nearest_neighbors/
├── notebook.ipynb                # main.pyの代用（実験の可視化など対話的に実行）
├── assignment/                   # 実験設計と評価ロジックを提供
│   ├── experiment.py             # 各種実験の定義（クラス数、特徴数、データサイズ）
│   ├── model_experiment_manager.py  # 実験実行管理クラス
│   ├── model_score_tracker.py       # 評価結果の集約・可視化
│   └── pca_image_compressor.py      # PCA による特徴量圧縮
└── dataset/                     # CIFAR-10 データの読み込み・分割・管理
    ├── cifar10_loader.py        # データのダウンロード、分割、クラス抽出
    ├── dataset_split.py         # 分割種別の Enum（TRAIN/EVAL/TEST）
    └── image_dataset.py         # 実験で使用する画像データ構造
```

- `/dataset` はデータの読み込みや前処理といった **土台（基盤）となる処理** を提供します。
- `/assignment` は各課題に応じた **実験ロジック（アルゴリズムや評価）** を定義しています。

---

## 🧠assignment モジュールの主なクラス
assignment モジュールでは、各課題に応じた 実験ロジック（アルゴリズムや評価） を定義しています。
### ▶️Experiment（抽象クラス）
- 各実験の共通インタフェース。
- `__iter__()` で条件を変化させたデータセットを逐次返却。

    #### ▶️TrainSizeExperiment
    - Experimentの具象クラスで、課題 1 に対応するクラス。
    - 訓練データのサイズ変化による性能評価。

    #### ▶️NumClassExperiment
    - Experimentの具象クラスで、課題 2 に対応するクラス。
    - 使用クラス数の変化による評価。内部で PCA を使用。

    #### ▶️FeatureCompressExperiment
    - Experimentの具象クラスで、課題 3 に対応するクラス。
    - 特徴量数（PCA次元数）の変更による性能評価。

### ▶️ModelExperimentManager
- 実験（Experiment）を実行し、`GridSearchCV` によるモデル最適化を行う。

### ▶️ModelScoreTracker
- 実験結果（精度、最適パラメータなど）を記録・集計し、条件ごとに比較しやすい形式で保持します。
- `ModelExperimentManager` が `run_experiment()` 内部で使用しており、実験結果を収集し、`
  presentation()` メソッドにより表形式で出力する役割を担います。

### ▶️PCAImageCompressor
- PCA による次元圧縮と寄与率の可視化を行う。

---

## 📦dataset モジュールの主なクラス
dataset モジュールでは、データの読み込みや前処理といった 土台（基盤）となる処理 を提供します。
### ▶️CIFAR10Loader
- CIFAR-10 データセットのダウンロード、学習・評価・テストへの分割。
- 特定クラスに基づくサブセット抽出が可能。

### ▶️DatasetSplit（Enum）
- `TRAIN`, `EVAL`, `TEST` を定義し、分割処理で使用。

### ▶️ImageDataset
- 実験や学習時に使用する標準的な画像データ構造。
- `flat_images`, `labels`, `reducer`, `shuffle` などの属性を管理。

---

## 実行例（Notebook）

### 1. データの読み込み
#### 1-1. CIFAR-10 の読み込み

```python
from dataset.cifar10_loader import CIFAR10Loader

loader = CIFAR10Loader(seed=42)
dataset = loader.train
eval_dataset = loader.eval
```

- CIFAR-10 をダウンロードし、トレーニング・評価・テストの 3 分割に分けます。
- `dataset` は訓練用の `ImageDataset`、`eval_dataset` は評価用に使用されます。

---

#### 1-2. CIFAR10の特定のクラスのみを抽出する場合

```python
selected_classes = ["airplane", "automobile", "ship"]
train, eval, test = loader.get_datasubsets(selected_classes)
```

- 指定したクラス名に対応する画像データのみを抽出し、訓練・評価・テスト用に取得します。
- クラス数を減らして性能への影響を調べるなどの実験に適しています。

---

### 2. 実験の実行（例：TrainSizeExperiment）

```python
from assignment.experiment import TrainSizeExperiment
from assignment.model_experiment_manager import ModelExperimentManager

settings = [0.1, 0.3, 0.5]
experiment = TrainSizeExperiment(dataset, settings, seed=42)
manager = ModelExperimentManager(eval_dataset)
tracker = manager.run_experiment(experiment, param={"n_neighbors": [1, 3, 5]})
```

- トレーニングデータの割合を 10%、30%、50% に変化させた実験を実行。
- `run_experiment()` により、グリッドサーチで最適なハイパーパラメータ（ここでは `n_neighbors`）を探索し、精度を評価します。

---
## ⚠️注意点

以下は、このプロジェクトを利用する際に留意すべき事項です。

1. **`ImageDataset` を使わないと動かない**
   - 実験やモデル評価では、データは必ず `ImageDataset` 型で渡す必要があります。
   - numpy 配列や他の構造体を直接使用すると `.flat_images` や `.reducer` へのアクセスでエラーになります。
   - ✅ 対処法：データは `CIFAR10Loader` を使って読み込むか、自前の配列から `ImageDataset` を生成してください。

2. **PCA を適用していないと次元圧縮実験が失敗する**
   - `FeatureCompressExperiment` や `NumClassExperiment` では PCA 適用が前提です。
   - 評価用データ（`eval_dataset`）にも学習データに適用したPCAモデルを適用していなければ、データ次元の不一致により評価に失敗します。
   - ✅ 対処法：評価データを保持する`ImageDataset`の`set_reducer()`関数を用いて、学習データに適用したのと同一のPCAモデルをセットしてください。

3. **クラス名のスペルミスによる抽出失敗**
   - `"airplane"` や `"automobile"` など、CIFAR-10 のクラス名は厳密にハードコードされています。
   - ✅ 対処法：指定可能なクラス名は `CIFAR10Loader.MAP_LABEL_ID.keys()` で確認可能です。

4. **条件リストの値域が不適切**
   - 例えば `TrainSizeExperiment` において 0.0〜1.0 の範囲外の比率を指定すると `RuntimeError` になります。
   - ✅ 対処法：必ず仕様に準拠した範囲で条件リストを設定してください。

---