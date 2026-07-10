from services.dataset_loader import DatasetLoader


loader = DatasetLoader("datasets/sample_dataset.csv")

df = loader.load()

loader.summary(df)

print(df.head())