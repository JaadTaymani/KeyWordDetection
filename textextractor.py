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

train_csv.groupby(by="label").size()


def read_html(doc_id: int) -> str:
    with open(file=f"htmls/{doc_id}.html",
              mode="r",
              encoding="latin1") as f:
        html = f.read()
    return html


train_csv["html"] = train_csv["doc_id"].apply(read_html)

train_csv.sample(n=5, random_state=42)





import warnings

from bs4 import BeautifulSoup

warnings.filterwarnings(action="ignore")


def extract_html_text(html):
    bs = BeautifulSoup(markup=html, features="lxml")
    for script in bs(name=["script", "style"]):
        script.decompose()
    return bs.get_text(separator=" ")


train_csv["html_text"] = train_csv["html"].apply(extract_html_text)

train_csv.sample(n=5, random_state=42)

from gensim.parsing import preprocessing


def preprocess_html_text(html_text: str) -> str:
    preprocessed_text = preprocessing.strip_non_alphanum(s=html_text)
    preprocessed_text = preprocessing.strip_multiple_whitespaces(s=preprocessed_text)
    preprocessed_text = preprocessing.strip_punctuation(s=preprocessed_text)
    preprocessed_text = preprocessing.strip_numeric(s=preprocessed_text)

    preprocessed_text = preprocessing.stem_text(text=preprocessed_text)
    preprocessed_text = preprocessing.remove_stopwords(s=preprocessed_text)
    return preprocessed_text


train_csv["preprocessed_html_text"] = train_csv["html_text"].apply(preprocess_html_text)

train_csv.sample(n=5, random_state=42)


