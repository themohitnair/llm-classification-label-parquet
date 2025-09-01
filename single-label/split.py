import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser(description="Stratified train/test split for multiple label columns.")
    parser.add_argument("--input_parquet", required=True, help="Input Parquet file path")
    parser.add_argument("--output_dir", default="splits", help="Output directory for split files")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test set size (float for proportion, int for count)")
    parser.add_argument("--label_columns", type=str,
                        default="purpose,polarity,emotion,delivery,domain",
                        help="Comma-separated label columns to split")
    args = parser.parse_args()

    df = pd.read_parquet(args.input_parquet)
    label_cols = [x.strip() for x in args.label_columns.split(",")]
    text_col = "description"
    os.makedirs(args.output_dir, exist_ok=True)

    for label_col in label_cols:
        if label_col not in df.columns:
            print(f"[warn] Column '{label_col}' not found in input file, skipping.")
            continue

        valid_df = df[df[label_col].notnull()]
        if valid_df.empty or valid_df[label_col].nunique() < 2:
            print(f"[warn] Not enough data/classes for column '{label_col}', skipping.")
            continue

        stratify_col = valid_df[label_col]
        # Handle test_size flexibly: int means count, float means proportion
        ts = args.test_size
        if isinstance(ts, float) and (ts > 0 and ts < 1):
            actual_test_size = ts
        elif isinstance(ts, int) and ts > 0 and ts < len(valid_df):
            actual_test_size = ts
        else:
            actual_test_size = 0.2  # fallback

        train, test = train_test_split(
            valid_df,
            test_size=actual_test_size,
            random_state=42,
            stratify=stratify_col
        )

        train_path = os.path.join(args.output_dir, f"train_{label_col}.parquet")
        test_path  = os.path.join(args.output_dir, f"test_{label_col}.parquet")
        train.to_parquet(train_path, index=False)
        test.to_parquet(test_path, index=False)

        print(f"[{label_col}] -> train: {len(train)}, test: {len(test)}, classes: {train[label_col].nunique()} saved.")

if __name__ == "__main__":
    main()
