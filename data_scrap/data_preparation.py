from concurrent.futures import ThreadPoolExecutor
import numpy as np
from preprocessing import Preprocessing
from tqdm import tqdm
import pandas as pd

tqdm.pandas()


def parallel_preprocess(texts, pipeline_func, n_workers=4):
    """Parallel text preprocessing"""
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        return list(tqdm(executor.map(pipeline_func, texts), total=len(texts)))


preprocessor = Preprocessing()

df = pd.read_csv("data/data.csv")

chunks = np.array_split(df["comment"], 4)

processed_chunks = []
for chunk in tqdm(chunks, desc="Processing chunks"):
    processed_chunks.append(
        parallel_preprocess(chunk, preprocessor.preprocessing_pipeline_roberta)
    )

df["comment"] = pd.concat([pd.Series(c) for c in processed_chunks], ignore_index=True)

df.to_csv("data/processed_data_2.csv", index=False)
