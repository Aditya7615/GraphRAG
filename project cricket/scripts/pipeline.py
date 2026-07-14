import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.build_features import main as build_features
from src.models.train_model import main as train_models


def main():
    print("\n" + "=" * 60)
    print("FULL PIPELINE")
    print("=" * 60)

    build_features()
    train_models()

    print("\nPipeline complete!")


if __name__ == '__main__':
    main()
