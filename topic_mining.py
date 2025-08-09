import re
import string
from typing import List

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Import your own keyword extraction & cleaning utilities
from keyword_extraction import find_keywords
from utils.cleaning_utils import clean_text


def topic_modeling_with_keywords_lda(
    train_docs: List[str],
    n_topics: int = 3,
    top_n_docs: int = 5,
    max_keywords: int = 10
) -> List[str]:
    """
    Fit an LDA model and extract top keyword phrases for each topic
    using PageRank-based keyword extraction.

    Steps:
    1. Vectorize documents into term-frequency matrix.
    2. Fit LDA and compute per-document topic distributions.
    3. For each topic:
       a. Select top-N documents with highest topic probability.
       b. Aggregate their text and clean it.
       c. Extract top keywords using `find_keywords`.

    Args:
        train_docs (List[str]): List of document strings.
        n_topics (int): Number of topics to model with LDA.
        top_n_docs (int): Number of top documents to aggregate per topic.
        max_keywords (int): Number of keywords/phrases to extract per topic.

    Returns:
        List[str]: Comma-separated keyword strings, one per topic.
    """

    # 🛑 Early exit if there are no documents
    if not train_docs:
        return []

    # 📊 Dynamic term-frequency thresholds for vectorizer
    # If only 1 doc, use max_df=1.0; else limit very common terms to <=95% of docs
    min_df = 1
    max_df = 1.0 if len(train_docs) == 1 else 0.95

    # 1️⃣ Convert documents to term-frequency representation
    vectorizer = CountVectorizer(
        stop_words='english',
        min_df=min_df,
        max_df=max_df
    )
    doc_term_matrix = vectorizer.fit_transform(train_docs)

    # 2️⃣ Fit LDA model on the term-frequency matrix
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42
    )
    lda.fit(doc_term_matrix)

    # 🔢 Topic distribution for each document
    topic_probs = lda.transform(doc_term_matrix)

    # Final list of keyword strings (one per topic)
    topic_keywords_list: List[str] = []

    # 3️⃣ For each topic...
    for topic_idx in range(n_topics):
        # a) Select top documents for this topic by probability
        sorted_docs = np.argsort(topic_probs[:, topic_idx])[::-1]  # highest first
        selected = sorted_docs[:top_n_docs]

        # b) Aggregate their text
        agg_text = " ".join(train_docs[i] for i in selected)
        clean_agg = clean_text(agg_text)  # custom cleaning function

        if not clean_agg:
            topic_keywords_list.append('')
            continue

        # c) Extract keywords/phrases using PageRank-based method
        keywords_with_scores = find_keywords(
            clean_agg,
            num_keywords=max_keywords
        )

        # Only keep the keyword text, discard scores/counts
        keywords = [keyword for keyword, _, _ in keywords_with_scores]

        # Join keywords into a single comma-separated string
        topic_keywords_list.append(", ".join(keywords))

    return topic_keywords_list
