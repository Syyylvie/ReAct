from gensim.models import KeyedVectors
import numpy as np

# Load a word2vec model stored in the C *text* format.
# wv_from_text = KeyedVectors.load_word2vec_format('glove.6B/glove.6B.300d.txt', binary=False)
# wv_from_text.fill_norms() 
wv_from_text = KeyedVectors.load("word_vecs_glove")
### compute n_grams
def compute_n_grams(s: str, n: int):
    if n > len(s):
        print(f"length exced ! Please reconsider {n}")
        return []
    return [s[i : i + n] for i in range(len(s)-n+1)]

### average word2vec
def word2vec(s: str, d: int):
    n = len(s)
    ### for loop: n --
    for i in range(n, -1, -1):
        ### compute n-grams
        grams = compute_n_grams(s, i)
        print(grams)
        vec = np.zeros(d, dtype=np.float32)
        vec = wv_from_text.get_mean_vector(grams, ignore_missing=True)
        if vec.any():
            return vec
    print(f"No vectors available for all n-grams in this word: {s}")
    return None

### recommend 5 candidates from a given entity
def recommend(entity: str):
    vec = word2vec(entity.lower(), 300)
    words = wv_from_text.most_similar([vec], topn=5)
    print(words)