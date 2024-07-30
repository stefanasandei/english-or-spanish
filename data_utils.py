import pandas as pd
import numpy as np
import re

# 1. read the data
df = pd.read_csv("data.csv")

# 2. clean the dataset
used_languages = [
    "English",
    "Portugeese",
    "French",
    "Spanish",
    "Italian",
    "Sweedish",
    "German",
]  # languages with mostly ascii words (to reduce the total vocab size)
df = df[df["Language"].isin(used_languages)]


def clean_text(row: str) -> str:
    regex_patterns = [
        r"[^a-zA-Z0-9\+(\-){1,}]{1,}",
        r"[,;-]",
        r"[\d-]",
        r"[()]",
        r"[{}]",  # regex is hard...
        r"[+]",
    ]
    for pat in regex_patterns:
        row["Text"] = re.sub(pat, " ", row["Text"])
    row["Text"] = row["Text"].lower().strip()
    return row


df = df.apply(clean_text, axis=1)

# 3. encode the data
chars = sorted(set(list("".join(df["Text"]))))
char_to_index = {char: idx + 1 for idx, char in enumerate(sorted(chars))}
char_to_index["<PAD>"] = 0
vocab_size = len(char_to_index)
lang_to_index = {lang: idx for idx, lang in enumerate(used_languages)}

# parameters
max_chars_in_word = 10
max_words_in_sentence = 15


def encode_char(char: str) -> np.array:
    # create a one-hot encoding
    vec = np.zeros((vocab_size))
    vec[char_to_index.get(char, 0)] = 1
    return vec


def encode_word(word: str) -> np.array:
    # return a matrix of one-hot vectors (one for each char)
    encoded = []
    for i, char in enumerate(word):
        if i == max_chars_in_word:
            break
        encoded.append(encode_char(char))
    encoded.extend([encode_char(" ") for _ in range(max_chars_in_word - len(encoded))])
    return np.array(encoded)


def encode_sentence(sentence: str) -> list[np.array]:
    # return a list with words (2d arrays) made out of one-hot vectors
    words = sentence.split()[:max_words_in_sentence]
    encoded_words = [encode_word(word) for word in words]
    return encoded_words


def encode_label(label: str) -> np.array:
    vec = np.zeros(len(used_languages))
    vec[lang_to_index[label]] = 1
    return vec.shape


# 4. create inputs and labels
def transform_dataframe(
    df: pd.DataFrame, encode_text: callable, encode_lang: callable
) -> pd.DataFrame:
    df_transformed = pd.DataFrame(
        {
            "input": df["Text"].apply(encode_text),
            "label": df["Language"].apply(encode_lang),
        }
    )
    return df_transformed


x, y = [], []
for index, row in df.iterrows():
    a = encode_sentence(row["Text"])
    for b in a:
        x.append(b)
        y.extend(encode_label(row["Language"]))

# 1 batch is max_chars_in_word by vocab_size
x, y = np.array(x), np.array(y)


# functions to be used in the training scripts
def get_data_params() -> dict:
    return {
        "labels": df["Language"].unique(),
        "vocab_size": vocab_size,
        "max_chars_in_word": max_chars_in_word,
        "max_words_in_sentence": max_words_in_sentence,
        "data_size": len(x),
    }


def get_dataset(seed=42) -> tuple[np.array, np.array]:
    np.random.seed(seed)

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    return x[indices], y[indices]
