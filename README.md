# Regression_FeatureEffect

## 概要

本プロジェクトは、回帰モデルを用いて最適な説明変数Xの値を逆解析し、さらに特定の説明変数を変化させた際の目的変数yの変化を可視化するPythonスクリプト群です。

- データ・設定ファイルは `data.csv`・`config.json` を利用します。
- 回帰モデルは以下から選択可能です：
  - LinearRegression
  - ElasticNet
  - RandomForest
  - GBR (GradientBoostingRegressor)
  - SVR

-## ディレクトリ構成

- `src/` : メインロジックを格納したパッケージ
  - `data_loader.py` : データ・設定ファイルの読み込み
  - `regression_model.py` : モデル選択・学習・予測
  - `inverse_analysis.py` : 逆解析（最適化）
  - `visualization.py` : 可視化
  - `main.py` : 全体の実行制御
- `config.json` : 目的変数数・目的・説明変数範囲などの設定
- `data.csv` : 入力データ

## 使い方

1. 必要なパッケージをインストール

```bash
pip install -r requirements.txt
```

2. `main.py` を実行

```bash
python src/main.py
```

3. モデル選択や可視化対象変数などは、`main.py`内または`config.json`で指定してください。

## 設定ファイル例

`config.json` の例：
```json
{
  "num_targets": 2,
  "objectives": ["maximize", "minimize"],
  "x_bounds": [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
}
```

## ライセンス

MIT License
