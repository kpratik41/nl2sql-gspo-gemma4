# synthmark-detect

Scores text for a SynthID-Text watermark, and serves detection over HTTP.

Detection reads token ids and hashes them: no forward pass, no model weights, no
GPU, ~1,250 documents/s per key on ordinary CPU. It belongs on its own hardware,
away from the serving fleet.

```python
from synthmark_detect import Detector

result = Detector(key, tokenizer).detect(text)
print(result.score, result.p_value, result.num_tokens_scored)
```

The HTTP service (`synthmark_detect.serve`) needs `pip install
"synthmark-detect[serve]"`. It exposes `POST /detect` ("is this marked by this
key?") and `POST /attribute` ("which of our models wrote this?", corrected for
the number of keys tested).

A watermark key is symmetric: whoever can detect with it can also forge with it.
Run this as a service that holds the keys; do not distribute keys to callers.

Full documentation: <https://github.com/kpratik41/nl2sql-gspo-gemma4/tree/watermarking>
