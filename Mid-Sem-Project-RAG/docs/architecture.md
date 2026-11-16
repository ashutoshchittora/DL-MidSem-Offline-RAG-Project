# RAG System Architecture and Methodology

## 1. Introduction

This project implements an offline Retrieval-Augmented Generation
(RAG) system for question answering over a locally stored
Wikipedia corpus.

The system combines classical information retrieval with
a pretrained sequence-to-sequence language model.

The architecture separates document indexing from
question answering.

## 2. Data Preparation

### Dataset

Source: Wikimedia Wikipedia

Configuration: 20231101.en

A corpus containing 1,000 Wikipedia articles was selected
and saved locally.

### Passage Chunking

Wikipedia articles were divided into overlapping passages.

Parameters:

- Chunk size: 100 words
- Overlap: 20 words
- Total passages: 8,191

Overlapping chunks help preserve information near passage
boundaries.

Each passage retains:

- Unique chunk ID
- Source article ID
- Article title
- Wikipedia URL
- Chunk number
- Text
- Word count

## 3. TF-IDF Indexing

The passages are represented using TF-IDF vectors.

TF-IDF stands for:

Term Frequency–Inverse Document Frequency.

It assigns weights to terms based on their frequency
within a passage and their distribution across the corpus.

The resulting TF-IDF matrix has dimensions:

8,191 × 30,000

Rows represent passages.

Columns represent vocabulary terms.

The matrix is stored in sparse CSR format.

### Saved Artifacts

- vectorizer.joblib
- tfidf_matrix.npz
- index_metadata.json

These artifacts allow the index to be loaded without
recomputing TF-IDF.

## 4. Passage Retrieval

A user question is transformed using the fitted
TF-IDF vectorizer.

The question vector and passage vectors share the
same vocabulary dimensions.

Cosine similarity measures their directional similarity.

The retriever ranks passages according to similarity.

The final configuration selects the highest-ranked passage:

K = 1

## 5. Evidence Preparation

Retrieved passages are shortened before generation.

The selected experimental baseline uses a maximum of
80 Flan-T5 tokenizer tokens per passage.

This reduces input length and helps respect the
generator's 512-token input limit.

However, prefix shortening can remove important information
that appears later in a passage.

The John Hubley question demonstrated this limitation.

## 6. Evidence Fusion

Two strategies were evaluated.

### Concatenation

Retrieved passage texts are combined before generation.

This requires one model-generation call.

### Generate-from-Evidence

The generator processes each retrieved passage independently.

Candidate answers are then supplied to another generation
call for final synthesis.

This requires K + 1 generation calls.

Experimental results favored concatenation for the
selected dataset and model.

## 7. Answer Generation

Generator model:

google/flan-t5-small

Architecture:

Encoder-decoder Transformer

The generator receives a prompt containing:

- Instructions
- Retrieved evidence
- User question

The model produces an answer conditioned on the prompt.

Generation uses deterministic decoding with:

- do_sample=False
- num_beams=1

The model input is checked against the 512-token limit
before generation.

## 8. Final Application Pipeline

The final application uses:

1. TF-IDF retrieval
2. Cosine similarity
3. Top-1 passage selection
4. 80-token prefix shortening
5. Concatenation fusion
6. Flan-T5-small answer generation

Application configuration is stored in:

config/rag_config.json

The standalone entry point is:

src/run_rag.py

## 9. Evaluation Methodology

The evaluation dataset contains:

- 20 questions
- 14 source articles
- Reference answers
- Expected source articles

The experiment compared:

- K = 1, 3, 5
- Concatenation
- Generate-from-evidence

A total of 120 evaluations were conducted.

### Evaluation Metrics

- Exact Match
- Token F1
- Retrieval Hit@K
- Evidence Support@K
- Abstention rate
- Generation time
- Token consumption
- Model calls

## 10. Selected Configuration

The best-performing configuration was:

Concatenation Fusion with K=1

Results:

- Exact Match: 75%
- Token F1: 85.62%

The selected approach also required fewer model calls
than generate-from-evidence fusion.

## 11. Limitations

The current architecture has several limitations:

1. TF-IDF depends primarily on lexical similarity.
2. The indexed corpus contains only a Wikipedia subset.
3. Passage shortening can discard useful information.
4. Irrelevant evidence can contaminate generated answers.
5. Flan-T5-small may generate incorrect or incomplete answers.
6. Results are based on a relatively small evaluation dataset.
7. Article-level retrieval success does not guarantee
   answer-bearing passages.

## 12. Conclusion

The implemented architecture demonstrates how retrieval
can augment a pretrained language model using an external
knowledge corpus.

The experiments show that retrieval size, evidence selection
and fusion strategy influence answer quality.

For the evaluated corpus and model, a simple concatenation
strategy using one highly ranked passage provided the
strongest accuracy-efficiency tradeoff.