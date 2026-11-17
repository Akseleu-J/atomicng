# Show HN preparation

## Submission fields

- **Form** — [Hacker News submit](https://news.ycombinator.com/submit)
- **URL** — `https://github.com/Akseleu-J/atomicng#try-the-cpu-demo`
- **Title** — Required prefix `Show HN:`, then AtomicNG and one concrete capability in the author's words. Possible focus: writing a removable n-gram memory patch and verifying exact rollback.
- **Text** — Leave empty for a link submission. Add the context in an author-written first comment. The [Hacker News guidelines](https://news.ycombinator.com/newsguidelines.html) ask contributors to write their own posts and comments.

## Facts for the author's first comment

1. Why you built it. Pick a real reason from your work on AtomicNG and describe the experiment or constraint that led to it.
2. What readers can reproduce. Install `atomicng==1.0.0` and run the README example. In the verified CPU run, the write touched 42 rows, changed the matching probe, and clearing the patch restored the original output bytes. Eight neutral probes matched before and after clearing. The example uses the deterministic NumPy `TinyModel` backend and requires no model download.
3. What readers can try next. The [pretrained Writer example](https://github.com/Akseleu-J/atomicng/blob/main/examples/pretrained_writer.py) downloads a public 234,961-parameter checkpoint for SmolLM2-135M and verifies the file's SHA-256. The CPU run wrote 19 rows and verified bit-exact rollback with a separately initialized projection. Its NLL change was negligible. The original 860-step run's +0.0422 acquisition result came from one document. See [the calibration record](https://github.com/Akseleu-J/atomicng/blob/main/CALIBRATION.md).
4. Where feedback would help. Pick one specific question, such as how others would design a retention or interference benchmark for successive writes.

## Optional video

The 30-second English MP4 is in the workspace's `atomicng-show-hn` folder. It uses the captured output from the published 1.0.0 package. Use the runnable repository as the submission URL. If the video is uploaded separately, it can supplement the first comment.

The first frame identifies the NumPy test backend.

| Time | Screen |
|---|---|
| 0–7 s | Show the example and the baseline probe captured before the write. |
| 7–15 s | Run it and highlight the touched row count and changed matching probe. |
| 15–23 s | Highlight the byte-for-byte restoration after clearing the patch. |
| 23–30 s | Show the neutral control result and the repository URL. |
