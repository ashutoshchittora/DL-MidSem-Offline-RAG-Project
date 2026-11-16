# Offline Wikipedia Retrieval-Augmented Generation

## Deep Learning / Generative AI Project

**Institution:** Indian Institute of Technology Bombay  
**Programme:** ePGD in Artificial Intelligence and Data Science  
**Project:** Offline Wikipedia RAG System

---

# Abstract

This project presents an offline Retrieval-Augmented Generation
(RAG) system for question answering over a locally stored
Wikipedia corpus.

A subset of 1,000 Wikipedia articles was divided into 8,191
overlapping passages. The passages were indexed using
Term Frequency–Inverse Document Frequency (TF-IDF)
with a vocabulary of 30,000 terms.

For a given question, the system retrieves relevant passages
using cosine similarity and generates an evidence-grounded
answer using Flan-T5-small.

Two evidence-fusion strategies were compared:
concatenation and generate-from-evidence.

Retrieval sizes K=1, 3 and 5 were evaluated using a dataset
of 20 manually prepared questions.

A total of 120 evaluations involving 300 model-generation
calls were conducted.

Concatenation with K=1 achieved the best results:
75% Exact Match and 85.62% Token F1.

The experiments demonstrated that increasing the number
of retrieved passages did not necessarily improve answer
quality and could introduce irrelevant information.

The final application supports local artifact loading,
configuration-driven execution, source attribution and
offline inference.

---

# 1. Introduction

Large language models can generate natural-language answers,
but their outputs may contain factual inaccuracies or
unsupported information.

Retrieval-Augmented Generation addresses this limitation
by supplying external evidence to a language model.

In RAG, relevant information is retrieved from a knowledge
corpus and included in the prompt used for generation.

This project implements a lightweight RAG system using
classical TF-IDF retrieval and a pretrained encoder-decoder
Transformer model.

The objective is to study how retrieval size and evidence
fusion influence answer quality and computational efficiency.

---

# 2. Dataset and Preprocessing

## 2.1 Wikipedia Corpus

The dataset was obtained from Wikimedia Wikipedia,
configuration `20231101.en`.

A subset of 1,000 English Wikipedia articles was selected.

Each article includes an identifier, title, source URL
and textual content.

## 2.2 Passage Chunking

Articles were divided into overlapping word-based passages.

The chunking configuration was:

- Chunk size: 100 words
- Overlap: 20 words
- Total indexed passages: 8,191

Overlap was used to reduce information loss at chunk boundaries.

## 2.3 TF-IDF Index Construction

A TF-IDF vectorizer was fitted on the passage corpus.

The resulting sparse matrix contained:

- 8,191 rows representing passages
- 30,000 columns representing vocabulary terms

The matrix was stored in compressed sparse-row format.

The fitted vectorizer, TF-IDF matrix and index metadata
were saved locally.

---

# 3. Methodology

## 3.1 Retrieval

Each user question was transformed into a TF-IDF vector
using the fitted vocabulary.

Cosine similarity was calculated between the question
vector and indexed passage vectors.

The highest-scoring passages were returned as evidence.

## 3.2 Evidence Preparation

Retrieved passages were shortened to a maximum of
80 Flan-T5 tokenizer tokens per passage.

This was necessary to satisfy the generator's
512-token input limit.

## 3.3 Answer Generation

The generator was Google's Flan-T5-small,
a pretrained encoder-decoder Transformer.

The generator received a prompt containing:

- Task instructions
- Retrieved evidence
- User question

Generation used deterministic decoding.

## 3.4 Evidence Fusion

Two fusion methods were evaluated.

### Concatenation

The top-K evidence passages were combined into a single
context before generating an answer.

### Generate-from-Evidence

Each passage was processed independently to generate
a candidate answer.

The candidate answers were subsequently combined
using an additional generation step.

---

# 4. Experimental Setup

An evaluation dataset containing 20 questions covering
14 source articles was prepared.

The experiments compared:

- Retrieval sizes K = 1, 3, 5
- Concatenation fusion
- Generate-from-evidence fusion

The total number of evaluations was:

20 questions × 3 K values × 2 methods = 120

The experiments required 300 generation calls.

## 4.1 Evaluation Metrics

The following metrics were used:

- Exact Match
- Token F1
- Retrieval Hit@K
- Evidence Support@K
- Abstention rate
- Generation time
- Model-call count
- Token consumption

---

# 5. Experimental Results

## 5.1 Accuracy Comparison

| Method | K | Exact Match (%) | Token F1 (%) |
|---|---:|---:|---:|
| Concatenation | 1 | 75 | 85.62 |
| Concatenation | 3 | 70 | 78.88 |
| Concatenation | 5 | 65 | 73.88 |
| Generate-from-Evidence | 1 | 50 | 57.54 |
| Generate-from-Evidence | 3 | 35 | 43.66 |
| Generate-from-Evidence | 5 | 20 | 24.38 |

## 5.2 Overall Fusion Comparison

| Metric | Concatenation | Generate-from-Evidence |
|---|---:|---:|
| Mean Exact Match | 70% | 35% |
| Mean Token F1 | 79.46% | 41.86% |
| Mean generation time | 0.0618 s | 0.1995 s |
| Total model calls | 60 | 240 |
| Total input tokens | 16,891 | 28,210 |

## 5.3 Retrieval Performance

Article Retrieval Hit@K was 100%.

Original evidence support was 90%.

Prepared evidence support was 85%.

This demonstrated that correct article retrieval did not
necessarily guarantee answer-bearing passages.

---

# 6. Discussion

The experiments showed that concatenation achieved better
accuracy than generate-from-evidence across all three
retrieval sizes.

Increasing K reduced Exact Match for both methods.

One explanation is that additional passages introduced
irrelevant or conflicting information.

Generate-from-evidence also introduced an additional
synthesis step, which sometimes failed to preserve
correct candidate answers.

The results suggest that retrieving a smaller amount
of highly relevant evidence can be more effective
than supplying larger quantities of potentially
irrelevant context.

However, these findings are specific to the selected
corpus, evaluation questions and generator configuration.

---

# 7. Limitations

The main limitations are:

1. The dataset contains only 1,000 Wikipedia articles.
2. The evaluation contains only 20 questions.
3. The questions were constructed from the indexed corpus.
4. Some questions informed development decisions.
5. TF-IDF primarily captures lexical similarity.
6. Prefix shortening can remove relevant evidence.
7. Flan-T5-small may produce incomplete or incorrect answers.
8. Exact Match can penalize factually correct paraphrases.
9. Evidence support was evaluated using lexical text matching.

---

# 8. Conclusion

This project successfully implemented an offline Wikipedia
Retrieval-Augmented Generation system.

The system combines TF-IDF retrieval with
Flan-T5-small answer generation.

A systematic evaluation of retrieval size and fusion
strategy identified concatenation with K=1 as the
best-performing configuration.

This configuration achieved:

- Exact Match: 75%
- Token F1: 85.62%

The experiments demonstrated that increasing retrieval
size did not necessarily improve generation quality.

The final application supports configuration-driven
execution, local artifact loading and offline inference.

Future work may investigate dense retrieval, hybrid
search, reranking and improved evidence selection.

---

# References

1. Wikimedia Wikipedia dataset — `wikimedia/wikipedia`.
2. Google Flan-T5-small — `google/flan-t5-small`.
3. scikit-learn — TF-IDF vectorization and cosine similarity.
4. Hugging Face Transformers — Model loading and generation.
5. PyTorch — Deep learning inference.