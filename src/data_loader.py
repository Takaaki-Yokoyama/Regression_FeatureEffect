import pandas as pd
import json
from pathlib import Path


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def load_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    return df


def get_feature_target_names(df: pd.DataFrame, num_targets: int):
    feature_names = df.columns[:-num_targets].tolist()
    target_names = df.columns[-num_targets:].tolist()
    return feature_names, target_names


if __name__ == "__main__":
    # 動作確認用
    root = Path(__file__).resolve().parent.parent
    config = load_config(root / "config.json")
    df = load_data(root / "data.csv")
    feature_names, target_names = get_feature_target_names(df, config["num_targets"])
    print("特徴量:", feature_names)
    print("目的変数:", target_names)
