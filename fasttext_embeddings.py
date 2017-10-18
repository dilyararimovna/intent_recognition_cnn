import fasttext
import numpy as np
import re

def split_string(s):
    return re.findall(r"[\w']+|[‑–—“”€№…’\"#$%&\'()+,-./:;<>?]", s)

def text2embeddings(texts, fasttext_model, text_size, embedding_size):
    all_embeddings = []
    for text in texts:
        embeddings = []  # embedded text
        tokens = split_string(text)
        if len(tokens) > text_size:
            tokens = tokens[-text_size:]
        for token in tokens:
            embeddings.append(fasttext_model[token])
        if len(tokens) < text_size:
            pads = [np.zeros(embedding_size) for _ in range(text_size - len(tokens))]
            embeddings = pads + embeddings
        embeddings = np.asarray(embeddings)
        all_embeddings.append(embeddings)

    all_embeddings = np.asarray(all_embeddings)
    return all_embeddings