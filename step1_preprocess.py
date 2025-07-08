import pandas as pd

# Load the training data
train_df = pd.read_csv("train.tsv", sep="\t", header=None)

# Assign column names based on dataset structure
train_df.columns = [
    "id", "label", "statement", "subject", "speaker",
    "speaker_job", "state_info", "party", "barely_true",
    "false", "half_true", "mostly_true", "pants_on_fire", "context"
]

# Select only statement and label
df = train_df[["statement", "label"]].dropna()

# Map the textual labels to binary: 0 = Fake, 1 = Real
def map_label(label):
    return 0 if label in ['false', 'pants-fire'] else 1

df["label"] = df["label"].apply(map_label)

# Preview the cleaned data
print(df.head())
