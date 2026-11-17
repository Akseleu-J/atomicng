# AtomicNG 1.0.0 release checklist

This checklist records the release work planned at 0.9.0rc1.

## Recorded as complete

- [x] 83 of 83 Windows tests and 72 of 72 Linux CUDA tests on Tesla T4 with Python 3.13.
- [x] Paper conformance tests and the calibration matrix.
- [x] Bit-exact rollback on CPU and in deterministic CUDA mode.
- [x] Package build, Twine checks, and the vLLM general plugin entry point.
- [x] Public README, conformance and calibration docs, vLLM guide, and Colab link.
- [x] v0.5.0a2 GitHub release with wheel and trained writer.
- [x] MkDocs Material and the Pages workflow.
- [x] Acquisition Δ_empty = +0.0422 in the recorded 860-step writer run.

## Publication and external validation

1. Upload the intended distribution to PyPI and verify installation from the index.
2. Run the [TPU notebook](https://colab.research.google.com/github/Akseleu-J/atomicng/main/notebooks/tpu_calibration.ipynb) in a Colab TPU runtime and retain the reports from `/content/results`.
3. Validate the merged-table path with a model that has a built-in n-gram table on Linux and record the result in `CALIBRATION.md`.
4. Configure GitHub Pages to publish from `gh-pages` after the docs workflow creates it.
5. Optionally file the upstream vLLM issue drafted in `docs/vllm.md`.

## Commands recorded for the release

```sh
# Upload the release-candidate artifacts.
python -m twine upload dist/atomicng-0.9.0rc1*
# After the release checks pass, publish the final tag and release notes.
git tag v1.0.0 && git push origin v1.0.0
gh release create v1.0.0 --title "AtomicNG 1.0.0" --generate-notes
```
