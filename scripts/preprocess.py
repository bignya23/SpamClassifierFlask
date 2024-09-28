import torch
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle


X_train, X_test, y_train, y_test, device = None, None, None, None, None
def preprocess():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reading the Csv file
    read_csv = pd.read_csv("../data/data.csv")

    read_csv.dropna(subset=["email"], inplace=True)

    # Converting it into bow-data using vectorizer
    text = read_csv["email"]
    vectorizer = CountVectorizer()
    bow_data = vectorizer.fit_transform(text).toarray()

    X = torch.tensor(bow_data, dtype=torch.float)
    y = torch.tensor(read_csv["label"], dtype=torch.float)

    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

    return X_train, X_test, y_train, y_test, device


if __name__ == "__main__":
    preprocess()