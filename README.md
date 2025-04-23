# 🎯 Rocchio Algorithm Demo: Relevance Feedback with Movie Plots

This is an interactive Python demo of the **Rocchio Algorithm** for relevance feedback in information retrieval. It uses a collection of **movie plot summaries** as the document corpus and shows how search results improve over multiple rounds of feedback.

---

## 🔍 What is the Rocchio Algorithm?

The Rocchio Algorithm is a classic method used in information retrieval to refine a search query based on user feedback about which results are relevant or irrelevant.

\[
\vec{q}_{\text{new}} = \alpha \vec{q} + \frac{\beta}{|D_r|} \sum_{\vec{d}_r \in D_r} \vec{d}_r - \frac{\gamma}{|D_{nr}|} \sum_{\vec{d}_{nr} \in D_{nr}} \vec{d}_{nr}
\]

Where:
- \( \vec{q} \): Original query vector
- \( D_r \): Relevant documents
- \( D_{nr} \): Irrelevant documents
- \( \alpha, \beta, \gamma \): Weights (default: 1.0, 0.75, 0.25)



