import pandas as pd


train_csv = pd.read_csv(filepath_or_buffer="train.csv")
print("Training set shape", train_csv.shape)

test_csv = pd.read_csv(filepath_or_buffer="test.csv")
print("Test set shape", test_csv.shape)

tumor_keywords = pd.read_csv(filepath_or_buffer="keyword2tumor_type.csv")

print("Tumor keywords set shape", tumor_keywords.shape)
tumor_keywords.head()
test_csv.head()
train_csv.head()


