from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = [
    "prompt",
     "context",
    "reference_answer",
    "model_answer"
]


class DatasetLoader:
    """
    Loads and validates evaluation datasets.
    """

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    def load(self) -> pd.DataFrame:

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.file_path}"
            )

        suffix = self.file_path.suffix.lower()

        if suffix == ".csv":
            df = pd.read_csv(self.file_path)

        elif suffix == ".json":
            df = pd.read_json(self.file_path)

        else:
            raise ValueError(
                "Only CSV and JSON files are supported."
            )

        self.validate(df)

        return df

    def validate(self, df: pd.DataFrame):

        missing = [
            col for col in REQUIRED_COLUMNS
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns: {missing}"
            )

    def summary(self, df: pd.DataFrame):

        print("\n========== Dataset Summary ==========")

        print(f"Rows : {len(df)}")

        print(f"Columns : {len(df.columns)}")

        print("\nColumn Names")

        for column in df.columns:
            print(f" - {column}")

        print("=====================================\n")