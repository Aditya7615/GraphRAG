import os, sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.features.data_utils import engineer_features

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'ipl_cleaned.csv')
FEATURES_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'ipl_features.csv')


def main():
    print("=" * 60)
    print("BUILD FEATURES")
    print("=" * 60)

    df_clean = pd.read_csv(CLEANED_PATH)
    print(f"Cleaned data: {df_clean.shape[0]:,} rows")

    df = engineer_features(df_clean)
    print(f"Features: {df.shape[0]:,} rows, {df.shape[1]} cols")

    target = df['overall_performance_score']
    print(f"Target: min={target.min():.2f}, mean={target.mean():.2f}, "
          f"median={target.median():.2f}, max={target.max():.2f}")
    print(f"Categories:\n{df['performance_category'].value_counts()}")

    df.to_csv(FEATURES_PATH, index=False)
    print(f"Saved to {FEATURES_PATH}")


if __name__ == '__main__':
    main()
