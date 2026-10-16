"""BM25 (Okapi) keyword search over the chunks, with a French analyzer.

Analyzer: lowercase -> split on non-letters/digits -> drop French stopwords -> Snowball
French stemming (optional) -> strip accents, so "écologique", "ecologiques" and
"Écologie" meet. Scores are computed with a precomputed sparse weight matrix:
score(q, d) = sum over query terms of idf(t) * tf(t,d) * (k1+1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl)).
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import numpy as np
import snowballstemmer
from sklearn.feature_extraction.text import CountVectorizer

TOKEN_RE = re.compile(r"[^\W_]+")

STOPWORDS = set("""
a à au aux avec ce ces cet cette dans de des du elle en et eux il ils je la le les leur leurs lui ma mais me
même mes moi mon ne nos notre nous on ou où par pas pour qu que qui sa se ses son sur ta te tes toi ton tu un
une vos votre vous y d l j m n s t c été être est sont ont a avoir fait faire peut peuvent dont ainsi
afin cela celle celui ceux comme donc entre jusqu lors plus sans selon si sous tout tous toute toutes très
quel quelle quels quelles quoi
""".split())


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


_STEMMER = snowballstemmer.stemmer("french")


@lru_cache(maxsize=None)
def normalize(token: str, stem: bool) -> str:
    return strip_accents(_STEMMER.stemWord(token) if stem else token)


def analyze(text: str, stem: bool = True) -> list[str]:
    return [normalize(t, stem) for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75, stem: bool = True):
        self.stem = stem
        self.vectorizer = CountVectorizer(analyzer=lambda t: analyze(t, stem))
        tf = self.vectorizer.fit_transform(texts).tocsr().astype(np.float64)
        n_docs = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        norm = k1 * (1 - b + b * doc_len / doc_len.mean())
        # BM25 weight of every (doc, term) pair, so a query is a column sum
        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        tf.data = self.idf[tf.indices] * tf.data * (k1 + 1) / (tf.data + norm[rows])
        self.weights = tf.tocsc()

    def scores(self, query: str) -> np.ndarray:
        vocab = self.vectorizer.vocabulary_
        cols = [vocab[t] for t in analyze(query, self.stem) if t in vocab]
        if not cols:
            return np.zeros(self.weights.shape[0])
        return np.asarray(self.weights[:, cols].sum(axis=1)).ravel()

    def search(self, query: str, k: int = 50) -> list[tuple[int, float]]:
        s = self.scores(query)
        top = np.argsort(-s, kind="stable")[:k]
        return [(int(i), float(s[i])) for i in top if s[i] > 0]
