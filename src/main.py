import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# ----- Load Corpus -----
def load_corpus(path="Corpus"):
    files = sorted(os.listdir(path))
    docs = []
    filenames = []
    for fname in files:
        with open(os.path.join(path, fname), 'r', encoding='utf-8') as f:
            docs.append(f.read())
            filenames.append(fname)
    return docs, filenames

# ----- Rocchio Feedback Formula -----
def rocchio_update(
    query_vector: csr_matrix,
    doc_vectors: csr_matrix,
    relevant_indices: list,
    irrelevant_indices: list,
    alpha: float = 1.0,
    beta: float = 0.75,
    gamma: float = 0.25,
) -> csr_matrix:
    if len(relevant_indices) == 0 and len(irrelevant_indices) == 0:
        return query_vector

    new_query = alpha * query_vector

    if relevant_indices:
        rel_vecs = doc_vectors[relevant_indices]
        rel_sum = rel_vecs.mean(axis=0)
        new_query += beta * rel_sum

    if irrelevant_indices:
        irrel_vecs = doc_vectors[irrelevant_indices]
        irrel_sum = irrel_vecs.mean(axis=0)
        new_query -= gamma * irrel_sum

    return new_query

# ----- Main Interactive Loop -----
def main():
    documents, filenames = load_corpus()
    vectorizer = TfidfVectorizer(stop_words='english')
    doc_vectors = vectorizer.fit_transform(documents)

    query = input("Enter your search query: ")
    query_vector = vectorizer.transform([query])

    for round_num in range(3):  # Loop for 3 feedback rounds
        print(f"\n🔄 ROUND {round_num + 1} 🔄")

        similarities = cosine_similarity(query_vector, doc_vectors).flatten()
        ranked_indices = np.argsort(similarities)[::-1]

        # Display top 5
        print("\nTop 5 matching documents:")
        for rank, idx in enumerate(ranked_indices[:5], start=1):
            print(f"{rank}. {filenames[idx]} (Score: {similarities[idx]:.4f})")

        # Ask for feedback
        relevant = input("👉 Enter relevant doc numbers (e.g. 1 3): ").strip().split()
        irrelevant = input("🚫 Enter irrelevant doc numbers (e.g. 2 4): ").strip().split()

        rel_indices = [ranked_indices[int(i)-1] for i in relevant if i.isdigit()]
        irrel_indices = [ranked_indices[int(i)-1] for i in irrelevant if i.isdigit()]

        # Update query
        query_vector = rocchio_update(query_vector, doc_vectors, rel_indices, irrel_indices)

    print("\n✅ Final results after 3 rounds of feedback:")
    final_similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    final_ranked = np.argsort(final_similarities)[::-1]
    for rank, idx in enumerate(final_ranked[:5], start=1):
        print(f"{rank}. {filenames[idx]} (Score: {final_similarities[idx]:.4f})")

if __name__ == "__main__":
    main()
