# Habr tutorial outline

### Working title
AtomicNG updates an LLM's n-gram memory with forward passes and bit-exact rollback

### Venue
Machine Learning hub. Plan for a 6–10 minute tutorial.

1. Introduce the problem of adding facts to an existing model. Explain where fine-tuning and retrieval place the new information.
2. Explain Hao Wu's Delta N-gram method (arXiv:2610.31125). Cover T = T⁰ + Δ, the writer, D1 block averaging, and what clearing Δ restores.
3. Introduce AtomicNG's NumPy core, write guards, exact undo, conformance tests, and evaluation protocols.
4. Show the quickstart and a screenshot of the CLI or HTML report.
5. Use the calibration table and describe the 860-step writer run, including the observed +0.0422 acquisition result on one document.
6. Explain the remaining vLLM engine hook, the PyTorch/XLA version constraint, and the scope of the acquisition results.
7. Link to GitHub, the package, the docs, and the Colab notebook.

Use the architecture diagram, an overlay manifest example, and the test protocol to support the walkthrough.
