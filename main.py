import json
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# TEXT PREPROCESSING
# =====================================================

def preprocess(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    return text


# =====================================================
# LOAD DOCUMENTS
# =====================================================

with open("docs.json", "r") as f:
    docs = json.load(f)

with open("queries.json", "r") as f:
    queries = json.load(f)


# =====================================================
# RETRIEVAL FUNCTION
# =====================================================

def retrieve(query, docs, k=2):

    # Original document texts
    original_texts = [doc["text"] for doc in docs]

    # Preprocessed texts
    processed_texts = [
        preprocess(doc["text"]) for doc in docs
    ]

    processed_query = preprocess(query)

    # TF-IDF Vectorizer with bigrams
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2)
    )

    # Convert docs into vectors
    doc_vectors = vectorizer.fit_transform(
        processed_texts
    )

    # Convert query into vector
    query_vector = vectorizer.transform(
        [processed_query]
    )

    # Similarity scores
    similarities = cosine_similarity(
        query_vector,
        doc_vectors
    )[0]

    # Top-k document indices
    top_indices = similarities.argsort()[-k:][::-1]

    retrieved_docs = []

    for idx in top_indices:

        retrieved_docs.append({
            "text": original_texts[idx],
            "score": round(float(similarities[idx]), 3)
        })

    return retrieved_docs


# =====================================================
# PROMPT BUILDER
# =====================================================

def build_prompt(query, context):

    context_block = "\n".join([
        f"{i+1}. {doc['text']}"
        for i, doc in enumerate(context)
    ])

    prompt = f"""
You are a factual QA assistant.

Rules:
- Use ONLY the provided context
- Do NOT hallucinate
- If answer is missing, say:
  "Insufficient information available."

Context:
{context_block}

Question:
{query}

Answer:
"""

    return prompt


# =====================================================
# ANSWER GENERATOR
# =====================================================

def generate_answer(query, context):

    context_texts = [
        doc["text"] for doc in context
    ]

    processed_contexts = [
        preprocess(text)
        for text in context_texts
    ]

    processed_query = preprocess(query)

    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2)
    )

    vectors = vectorizer.fit_transform(
        processed_contexts + [processed_query]
    )

    query_vector = vectors[-1]

    context_vectors = vectors[:-1]

    similarities = cosine_similarity(
        query_vector,
        context_vectors
    )[0]

    best_score = similarities.max()

    # Confidence threshold
    if best_score < 0.1:
        return "Insufficient information available."

    best_index = similarities.argmax()

    return context_texts[best_index]


# =====================================================
# EVALUATION FUNCTION
# =====================================================

def evaluate(answer, query, context):

    context_text = " ".join([
        doc["text"] for doc in context
    ])

    vectorizer = TfidfVectorizer(
        stop_words='english'
    )

    vectors = vectorizer.fit_transform([
        preprocess(answer),
        preprocess(query),
        preprocess(context_text)
    ])

    answer_vector = vectors[0]
    query_vector = vectors[1]
    context_vector = vectors[2]

    # Relevance score
    relevance = cosine_similarity(
        answer_vector,
        query_vector
    )[0][0]

    # Groundedness score
    groundedness = cosine_similarity(
        answer_vector,
        context_vector
    )[0][0]

    # Final score
    final_score = (relevance + groundedness) / 2

    return round(float(final_score), 2)


# =====================================================
# MAIN PIPELINE
# =====================================================

print("\n==============================")
print(" THRIFTY AI - SIMPLE RAG ")
print("==============================\n")

for item in queries:

    query = item["query"]

    print("=" * 60)

    print(f"\nQUERY:\n{query}")

    # ---------------------------------------------
    # RETRIEVAL
    # ---------------------------------------------

    retrieved_context = retrieve(
        query,
        docs,
        k=2
    )

    print("\nRETRIEVED CONTEXT:")

    for doc in retrieved_context:

        print(
            f"- {doc['text']} "
            f"(Similarity: {doc['score']})"
        )

    # ---------------------------------------------
    # PROMPT
    # ---------------------------------------------

    prompt = build_prompt(
        query,
        retrieved_context
    )

    print("\nPROMPT:")

    print(prompt)

    # ---------------------------------------------
    # ANSWER GENERATION
    # ---------------------------------------------

    answer = generate_answer(
        query,
        retrieved_context
    )

    print("GENERATED ANSWER:")

    print(answer)

    # ---------------------------------------------
    # EVALUATION
    # ---------------------------------------------

    score = evaluate(
        answer,
        query,
        retrieved_context
    )

    print("\nEVALUATION SCORE:")

    print(score)

    print("\n")
