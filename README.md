# RAG-implementation-without-any-database-or-API
This is the repository containing my implementation of RAG architecture using simple tf-idf and obtaining the related outpcomes using NLP.

DETAILED DESIGN CHOICES AND ASSUMPTIONS



DESIGN CHOICES – DETAILED EXPLANATION



1\. CHOICE OF TF-IDF FOR RETRIEVAL



WHAT WAS CHOSEN?



The retrieval system uses:



\- TF-IDF vectorization

\- Cosine similarity ranking



instead of:

\- deep learning embeddings

\- transformer models

\- vector databases





WHY WAS THIS CHOICE MADE?



The assignment explicitly required:

\- lightweight implementation

\- in-memory retrieval

\- no external APIs

\- no external vector databases



TF-IDF satisfies all these constraints while still providing meaningful document retrieval.





WHY TF-IDF WORKS WELL HERE



The dataset in this assignment is:

\- very small

\- factual

\- keyword-oriented

\- structured



Example:



Question:

Who launched Chandrayaan-3?



Relevant document:

India launched Chandrayaan-3 in 2023.



Important keywords overlap directly:

\- launched

\- Chandrayaan-3



TF-IDF performs very well in such situations.





WHY NOT USE EMBEDDINGS?



Embedding models like:

\- Sentence Transformers

\- BERT

\- OpenAI embeddings



would increase:

\- complexity

\- computation

\- dependencies



without significantly improving performance for such a small dataset.



The goal of the assignment is to demonstrate understanding of:

\- retrieval pipelines

\- information flow

\- system design



not necessarily large-scale semantic search systems.





ENGINEERING TRADEOFF



This design prioritizes:



\- simplicity

\- interpretability

\- lightweight execution

\- constraint compliance



over:

\- semantic depth

\- neural understanding





2\. CHOICE OF COSINE SIMILARITY



WHAT WAS CHOSEN?



Cosine similarity was used to compare:

\- query vectors

\- document vectors





WHY COSINE SIMILARITY?



TF-IDF converts text into numerical vectors.



Cosine similarity measures:



“How close two vectors are in direction”



instead of magnitude.



This is useful because:

\- long documents should not automatically dominate

\- keyword alignment matters more than sentence length





EXAMPLE



Query:

Who launched Chandrayaan-3?



Document:

India launched Chandrayaan-3 in 2023.



Both vectors contain:

\- launched

\- Chandrayaan-3



So cosine similarity becomes high.





WHY THIS IS GOOD FOR RAG



RAG systems fundamentally rely on:

\- retrieval ranking

\- relevance scoring



Cosine similarity is one of the most standard methods for:

\- document ranking

\- vector similarity search



Thus it reflects real-world retrieval system behavior.





3\. CHOICE OF TEXT PREPROCESSING



WHAT WAS ADDED?



The preprocessing pipeline includes:

\- lowercasing

\- punctuation removal





WHY LOWERCASE?



Without lowercasing:



ISRO



and



isro



would be treated as different tokens.



Lowercasing improves:

\- consistency

\- matching accuracy

\- retrieval quality





WHY REMOVE PUNCTUATION?



Punctuation creates unnecessary token differences.



Example:



Chandrayaan-3?



and



Chandrayaan-3



should represent the same concept.



Removing punctuation:

\- reduces noise

\- improves token matching

\- stabilizes similarity calculations





WHY KEEP PREPROCESSING SIMPLE?



More advanced NLP preprocessing:

\- stemming

\- lemmatization

\- POS tagging



was intentionally avoided to:

\- keep implementation lightweight

\- maintain readability

\- avoid unnecessary complexity



for a small factual dataset.



4\. CHOICE OF BIGRAM TF-IDF



WHAT WAS CHOSEN?



The vectorizer uses:



ngram\_range=(1,2)



This includes:

\- unigrams (single words)

\- bigrams (two-word phrases)





WHY ADD BIGRAMS?



Single-word TF-IDF may lose phrase-level meaning.



Example:



space research



is more meaningful together than:

\- space

\- research



independently.





BENEFITS



Bigram support improves:

\- phrase matching

\- contextual retrieval

\- ranking quality



especially for:

\- named entities

\- organizations

\- technical terms





WHY NOT LARGER N-GRAMS?



Using:

\- trigrams

\- 4-grams



would:

\- increase sparsity

\- reduce generalization

\- increase computational overhead



without much benefit for small documents.



Thus bigrams provide the best balance.







5\. CHOICE OF STRUCTURED PROMPT DESIGN



WHAT WAS CHOSEN?



The prompt explicitly separates:

\- instructions

\- context

\- question





WHY STRUCTURED PROMPTING?



Even though a real LLM is not used, structured prompts demonstrate understanding of:

\- prompt engineering

\- information organization

\- grounding strategies





EXAMPLE STRUCTURE



Context:

...



Question:

...



Answer:





WHY EXPLICIT INSTRUCTIONS?



Instructions like:



“Use ONLY the provided context”



simulate real-world RAG prompting techniques that aim to:

\- reduce hallucinations

\- enforce grounding

\- constrain answer generation





WHY THIS MATTERS



Modern RAG systems depend heavily on:

\- prompt quality

\- context formatting

\- grounding instructions



This design demonstrates awareness of those principles.





6\. CHOICE OF RULE-BASED ANSWER GENERATION



WHAT WAS CHOSEN?



The answer generator:

\- compares query with retrieved sentences

\- selects the highest similarity sentence



instead of generating free-form text.





WHY THIS CHOICE?



The assignment prohibited:

\- external APIs

\- pretrained LLMs



Thus a grounded extraction-based approach was used.





WHY EXTRACTION-BASED GENERATION IS GOOD



Advantages:

\- fully explainable

\- deterministic

\- hallucination-resistant

\- grounded in retrieved context





WHY NOT GENERATE NEW SENTENCES?



Generating free-form responses without an LLM:

\- becomes unreliable

\- increases hallucination risk

\- complicates evaluation



Thus sentence extraction is safer and more robust.





7\. CHOICE OF CONFIDENCE THRESHOLD



WHAT WAS ADDED?



A similarity threshold:



if best\_score < 0.1:

&#x20;   return "Insufficient information available."





WHY WAS THIS NECESSARY?



Without a threshold, the system may return:

\- weak matches

\- irrelevant answers

\- misleading outputs





WHY THRESHOLDING HELPS



This simulates real production systems where:

\- low-confidence outputs are filtered

\- uncertainty is acknowledged



It improves:

\- reliability

\- robustness

\- trustworthiness







8\. CHOICE OF EVALUATION METHOD



WHAT WAS CHOSEN?



Evaluation measures:

1\. Relevance

2\. Groundedness



using TF-IDF cosine similarity.





WHY EVALUATE BOTH?



A good answer should:

\- address the question

\- remain supported by context



Checking only one metric is insufficient.





WHY WEIGHTED AVERAGE?



Final score:



0.3 × relevance

\+ 0.7 × groundedness



Groundedness is weighted more heavily because:

\- factual consistency is critical in RAG

\- grounded answers are safer than creative answers





WHY NOT EXACT MATCH?



Exact-match evaluation:

\- is too strict

\- fails on paraphrases

\- ignores semantic similarity



Cosine similarity provides more flexibility.





1\. ASSUMPTION: DOCUMENTS ARE SHORT



WHY THIS ASSUMPTION WAS MADE



The provided dataset contains:

\- single-sentence factual documents



Thus:

\- chunking

\- indexing pipelines

\- hierarchical retrieval



were unnecessary.





IMPACT



This allows:

\- direct TF-IDF vectorization

\- fast in-memory retrieval

\- simplified implementation





2\. ASSUMPTION: ANSWERS EXIST IN RETRIEVED CONTEXT



WHY THIS ASSUMPTION WAS NECESSARY



The system uses extraction-based generation.



This means:

\- it cannot invent new information

\- it depends entirely on retrieved documents



Thus relevant answers must already exist in context.





WHY THIS IS REASONABLE



The assignment is focused on:

\- retrieval quality

\- grounding



not open-ended generation.



3\. ASSUMPTION: QUERIES ARE FACTUAL



WHY?



The provided examples are:

\- factual

\- direct

\- information-seeking



Example:



Which organization is ISRO?



This aligns well with:

\- TF-IDF retrieval

\- extraction-based answering





4\. ASSUMPTION: SMALL DATASET SIZE



WHY?



The dataset contains only a few documents.



Thus:

\- in-memory retrieval is sufficient

\- vector databases are unnecessary





ENGINEERING IMPACT



This keeps:

\- memory usage low

\- implementation lightweight

\- runtime fast



5\. ASSUMPTION: NO MULTI-HOP REASONING REQUIRED



WHAT IS MULTI-HOP REASONING?



Combining information from multiple documents.



Example:



Who launched the mission that landed on the moon's south pole?



would require:

\- Chandrayaan-3 launched by India

\- mission landed on moon south pole



combined reasoning.





WHY ASSUMED UNNECESSARY?



Provided queries are:

\- single-hop factual questions



Thus sentence extraction is sufficient.







6\. ASSUMPTION: TF-IDF IS SUFFICIENT



WHY?



The dataset:

\- has strong keyword overlap

\- uses consistent terminology



Thus semantic embeddings are not strictly necessary.





TRADEOFF



This assumption prioritizes:

\- simplicity

\- interpretability

\- assignment compliance



over deep semantic understanding.

