# AtomicNG 1.0.0 announcement plan

Lead with what the project does, cite the paper, and link to reproducible results and the Colab notebook. Describe the limits of the TPU and vLLM validation. Reserve time to answer questions during the first 24 hours.

| # | Venue | Format | Timing |
|---|---|---|---|
| 1 | r/LocalLLaMA | Project overview, repository link, and calibration results | Tuesday to Thursday, 13:00–15:00 UTC |
| 2 | r/LLMDevs | Developer post on the API, plugins, and package | 3–5 days after the first post |
| 3 | Hacker News | Short Show HN post and repository link | Morning Eastern Time |
| 4 | r/MachineLearning | Research post with the paper and evaluation scope | After initial feedback |
| 5 | Habr | Full tutorial in the Machine Learning hub | Daytime |
| 6 | X and Bluesky | Short thread with a demo and a Colab link | Alongside the first post |
| 7 | LocalLLaMA, llama.cpp, and Unsloth Discord communities | Brief showcase in the relevant channel | After the first post |
| 8 | GitHub Discussions | Release announcement | At launch |

Tailor each post to its venue. Use numbers from [CALIBRATION.md](../CALIBRATION.md), attribute the method to Hao Wu, and distinguish the TPU D1 harness from full BPTT training. The current vLLM path exports a merged table and exposes a plugin provider. Engine-graph integration remains upstream work.

Before posting, prepare a screenshot or GIF of the smoke check or notebook, check the README rendering, and verify the package page.
