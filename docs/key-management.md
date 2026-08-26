# Key management

The watermark key is **signing-grade secret material**. Anyone who holds it can do two
things: detect the watermark, and forge it. The second is the one that gets overlooked.
A leaked key does not merely stop working — it lets a third party produce text that your
own detector will confirm as yours.

Design the deployment around that fact.

## Derive, do not distribute

Escrow one master secret. Derive every operational key from it with a non-secret label:

```python
key = derive_key(master_secret, "markets-research/v1")
```

Derivation is HKDF-SHA256, so labels give cryptographically independent keys and one
derived key reveals nothing about the master or about its siblings. The practical wins:

- one secret to escrow, rotate, back up and audit, instead of a growing pile of files;
- new keys for new teams with no new secret material;
- any authorised host can reconstruct any key, so keys never travel between machines;
- rotation is a new label (`markets-research/v2`), and old text stays detectable under the
  old label for as long as you keep deriving it.

## Do not hand out the detector

The obvious way to let a downstream team check text is to give them the key. Doing so
gives them forging capability at the same time.

Run detection as a service instead (`synthmark serve`). Callers submit text and receive a
score; the key never leaves the service. This also gives you one place to enforce
authentication, rate limits, and audit logging.

## Separation between business units

Give each unit its own label. Because keys are independent, one unit's detector scores
another unit's watermarked output at chance — the wrong-key AUC in the evaluation is
0.5. That property is what makes it safe to run several units on shared infrastructure:
a detection result is scoped to the key that produced it and says nothing about anyone
else's traffic.

## Operational rules

| Rule | Why |
|---|---|
| Store the master secret in a secrets manager or HSM, never in a repo, image, or shell profile | It is the root of every derived key |
| Key files are mode 0600, created that way (`WatermarkKey.save` does this) | Avoids a window where the secret is world-readable |
| Log the `fingerprint`, never the key | `public_summary()` is safe to log; it is a SHA-256 digest |
| Do not log request bodies in the detection service | Submitted text is by definition text somebody is suspicious about |
| Treat a saved Bayesian detector as key material | Its config embeds the key |
| Version labels from the start (`/v1`) | Rotation without a naming scheme means re-keying everything at once |

## What rotation does and does not do

Rotating produces a new key for new generation. It does **not** invalidate the old
watermark: text generated under `v1` remains detectable under `v1` forever, and remains
forgeable by anyone who took a copy of `v1`. Rotation limits the blast radius of a future
leak; it does not repair a past one. If a key is known to have leaked, the honest position
is that detections under that key no longer distinguish your output from an impostor's,
and results under it should be treated as unreliable from the leak date onward.
