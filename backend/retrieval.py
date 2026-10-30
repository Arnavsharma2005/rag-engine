import re
import math
from collections import Counter

# Common English words that carry no meaning for keyword search
STOPWORDS = set("""a an and are as at be by for from has have how i in is it its of on or
that the this to was were what when where which who why will with does do did""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def bm25_search(chunks: list[dict], query: str, k: int, k1=1.5, b=0.75) -> list[dict]:
    """Rank chunks by BM25 keyword score. Catches exact terms (names, codes,
    numbers) that embeddings tend to blur."""
    docs = [tokenize(c["text"]) for c in chunks]
    if not docs:
        return []
    avg_len = sum(len(d) for d in docs) / len(docs)
    df = Counter(t for d in docs for t in set(d))
    n = len(docs)

    scores = []
    for i, d in enumerate(docs):
        tf = Counter(d)
        s = 0.0
        for t in set(tokenize(query)):
            if t not in tf:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avg_len))
        if s > 0:
            scores.append({"id": i, "score": s})
    scores.sort(key=lambda h: -h["score"])
    return scores[:k]


def rrf_fuse(rankings: list[list[int]], k: int, c=60) -> list[dict]:
    """Reciprocal rank fusion: merge ranked id lists without needing their
    scores to be comparable."""
    score = Counter()
    for ranking in rankings:
        for rank, i in enumerate(ranking):
            score[i] += 1 / (c + rank + 1)
    return [{"id": i, "score": s} for i, s in score.most_common(k)]
