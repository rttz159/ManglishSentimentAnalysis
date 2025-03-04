import pandas as pd
from preprocessing import Preprocessing
import asyncio


async def process_comments(comments):
    return await asyncio.gather(
        *(instance.preprocessing_pipeline(comment) for comment in comments)
    )


instance = Preprocessing()

df = pd.read_csv("data/data.csv")
df["comment"] = asyncio.run(process_comments(df["comment"].tolist()))
df.to_csv("data/processed_data.csv", index=False)
