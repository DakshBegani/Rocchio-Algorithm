import numpy as np
from scipy.sparse import csr_matrix

def rocchio_update(
    query_vector: csr_matrix,
    doc_vectors: csr_matrix,
    relevant_indices: list,
    irrelevant_indices: list,
    alpha: float = 1.0,
    beta: float = 0.75,
    gamma: float = 0.25,
) -> csr_matrix:
    """
    Apply the Rocchio formula:
    q_new = alpha * q + beta * (sum of relevant docs / len(relevant)) - gamma * (sum of irrelevant docs / len(irrelevant))
    """
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
