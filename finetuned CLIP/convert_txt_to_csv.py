import pandas as pd
df = pd.read_csv("/scratch/sa6981/Deep-Learning-Pokemon/captions.txt")
df['id'] = [id_ for id_ in range(df.shape[0])]
df.to_csv("/scratch/sa6981/Deep-Learning-Pokemon/captions.csv", index=False)
df = pd.read_csv("/scratch/sa6981/Deep-Learning-Pokemon/captions.csv")
print(df.head())
