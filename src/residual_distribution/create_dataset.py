import pandas as pd
from src.feature_engineering.merge import load_mmp


df = pd.read_parquet("src/documentation/paper_baseline/data/data.parquet")

df = df.query("doe_id == 'AL6_15_AL6_15'")

df.to_parquet("src/documentation/paper_residuals/data/data.parquet")
