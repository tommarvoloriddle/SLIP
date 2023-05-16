import pandas as pd
df = pd.read_csv("<path to captions.txt>")
df['id'] = [id_ for id_ in range(df.shape[0])]
df.to_csv("<path to folder you want your csv to be stored>/captions.csv", index=False)
df = pd.read_csv("path to captions.csv")
print(df.head())
