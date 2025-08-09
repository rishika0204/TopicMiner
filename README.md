# 🧭 TopicMiner — LDA Topics + PhraseRank Keywords

**TopicMiner** discovers latent topics with **LDA** and then extracts the most representative **keyword phrases per topic** using your TextRank-style extractor (`find_keywords`).  
It’s a simple, fast bridge between **probabilistic topics** and **human-friendly phrases**.

---

## ✨ What it does

- 🧮 Fits **Latent Dirichlet Allocation (LDA)** on a set of documents
- 📊 Finds the **top-N documents** most associated with each topic
- 🧹 Cleans & aggregates those documents per topic
- 🏷 Uses **PhraseRank** (`find_keywords`) to extract **keywords/phrases** per topic
- 📦 Returns a **list of strings**, one comma-separated phrase list for each topic

---

## 📦 Install

```bash
pip install numpy scikit-learn
````

Also make sure your PhraseRank extractor is importable:

```
keyword_extraction.py
utils/
  └─ cleaning_utils.py
```

> `topic_mining.py` imports:
>
> * `find_keywords` from `keyword_extraction`
> * `clean_text` from `utils.cleaning_utils`

---

## ⚡ Quick Start

```python
from topic_mining import topic_modeling_with_keywords_lda

docs = [
    "Transformers have improved state of the art in NLP tasks like QA and summarization.",
    "Convolutional networks remain strong for images; ViTs are now competitive.",
    "RNNs and LSTMs are used less for long-range text, attention dominates.",
    "Radiology reports need domain-specific models with careful evaluation.",
    "GANs, diffusion models and autoencoders power modern generative imaging."
]

topic_keywords = topic_modeling_with_keywords_lda(
    train_docs=docs,
    n_topics=3,       # number of LDA topics
    top_n_docs=5,     # top docs per topic to aggregate
    max_keywords=8    # phrases per topic
)

for i, kw in enumerate(topic_keywords, 1):
    print(f"Topic {i}: {kw}")
```

**Example Output**

```
Topic 1: transformer, nlp task, summarization, question answering, attention
Topic 2: image model, convolutional network, vision transformer, competitive
Topic 3: radiology report, generative imaging, diffusion model, autoencoder
```

---

## 🔍 How it works (under the hood)

1. **Vectorize** documents with `CountVectorizer` (English stopwords).
   Dynamic document-frequency thresholds:

   * `min_df = 1`
   * `max_df = 1.0` if only one doc, else `0.95`

2. **Fit LDA** (`sklearn.decomposition.LatentDirichletAllocation`) with `n_components = n_topics`.

3. **Score docs** for each topic via `lda.transform(...)` and pick **top\_n\_docs**.

4. **Aggregate** the selected docs’ text and clean it with `clean_text`.

5. **Extract keywords/phrases** from the aggregated text using `find_keywords(…, num_keywords=max_keywords)`.

6. **Return** a list of comma-joined keyword strings — one per topic.

---

## ⚙️ Function Signature

```python
def topic_modeling_with_keywords_lda(
    train_docs: List[str],
    n_topics: int = 3,
    top_n_docs: int = 5,
    max_keywords: int = 10
) -> List[str]:
    """Fit LDA and extract top phrases per topic with find_keywords()."""
```

### Parameters

| Name           | Type        | Default | Description                                |
| -------------- | ----------- | ------- | ------------------------------------------ |
| `train_docs`   | `List[str]` | —       | Raw document strings                       |
| `n_topics`     | `int`       | `3`     | Number of LDA topics                       |
| `top_n_docs`   | `int`       | `5`     | Docs per topic to aggregate for keywording |
| `max_keywords` | `int`       | `10`    | Phrases to return per topic                |

### Returns

* `List[str]` — Each entry is a **comma-separated** phrase list for a topic.

---

## 🧪 Tuning tips

* **`n_topics`**: Start with 5–10 for varied corpora; fewer for small datasets.
* **`max_df` / `min_df`**: Adjust if topics look too generic or too sparse.
* **`top_n_docs`**: Raise this to smooth noisy docs; lower to sharpen topic character.
* **`max_keywords`**: 5–12 usually reads well for dashboards.
* **Seed**: The code uses `random_state=42` for reproducible LDA runs.

---

## 📂 Project layout

```
.
├── topic_mining.py                 # this file
├── keyword_extraction.py           # PhraseRank-style extractor (dependency)
├── utils/
│   └── cleaning_utils.py           # clean_text() used here
└── README.md
```

---

## ✅ Why this approach?

* LDA provides **interpretable topic buckets**.
* PhraseRank turns those buckets into **human-friendly phrases**.
* The combo yields topic labels that are **both statistically grounded and readable**.

---

## 📜 License

MIT — use it freely in research and production.

---

💡 *Turn opaque topics into crisp, readable labels with TopicMiner.*
