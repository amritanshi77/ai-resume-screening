import spacy
import re

nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def preprocess_text(text):
    text = clean_text(text)
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and len(token.text) > 2
    ]
    return " ".join(tokens)