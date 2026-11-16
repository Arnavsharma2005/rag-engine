import re
import math
from collections import Counter

# Common English words that carry no meaning for keyword search
STOPWORDS = set("""a an and are as at be by for from has have how i in is it its of on or
that the this to was were what when where which who why will with does do did""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25Index:
    """Term statistics for one document, built once and reused by every query."""

    def __init__(self, chunks: list[dict], k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(c["text"])) for c in chunks]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.n = len(self.tfs)
        self.avg_len = (sum(self.lens) / self.n) if self.n else 0
        df = Counter(t for tf in self.tfs for t in tf)
        self.idf = {t: math.log(1 + (self.n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, k: int) -> list[dict]:
        terms = set(tokenize(query)) & self.idf.keys()
        scores = []
        for i, tf in enumerate(self.tfs):
            s = 0.0
            for t in terms:
                f = tf.get(t)
                if f:
                    norm = self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg_len)
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            if s > 0:
                scores.append({"id": i, "score": s})
        scores.sort(key=lambda h: -h["score"])
        return scores[:k]


_index_cache: dict = {}


def bm25_index(chunks: list[dict]) -> BM25Index:
    """The BM25 index for these chunks, cached by content so it is built once per upload."""
    key = hash(tuple(c["text"] for c in chunks))
    idx = _index_cache.get(key)
    if idx is None:
        _index_cache.clear()  # one document at a time
        idx = _index_cache[key] = BM25Index(chunks)
    return idx


def bm25_search(chunks: list[dict], query: str, k: int) -> list[dict]:
    """Rank chunks by BM25 keyword score. Catches exact terms (names, codes,
    numbers) that embeddings tend to blur."""
    if not chunks:
        return []
    return bm25_index(chunks).search(query, k)


def rrf_fuse(rankings: list[list[int]], k: int, c=60) -> list[dict]:
    """Reciprocal rank fusion: merge ranked id lists without needing their
    scores to be comparable."""
    score = Counter()
    for ranking in rankings:
        for rank, i in enumerate(ranking):
            score[i] += 1 / (c + rank + 1)
    return [{"id": i, "score": s} for i, s in score.most_common(k)]
