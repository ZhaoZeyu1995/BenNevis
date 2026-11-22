# Reference data

What Kaldi produces, recorded, so that replacing Kaldi can be checked against
it without Kaldi.

This directory is **empty until the first successful nightly run**. It is not
committed by hand: the data has to come from a real recipe run inside the
pinned Kaldi image, and fabricating it would be worse than having none —
every comparison against it would pass while meaning nothing.

## How it gets here

1. The nightly workflow runs `egs/yesno/asr` end to end in the Kaldi image.
2. It then runs `tests/reference/capture.py`, which writes `yesno/` and a
   `manifest.json`, and uploads the result as a build artefact.
3. That artefact is downloaded and committed here, in its own PR, with the
   image digest recorded in the manifest.

`tests/reference/test_against_reference.py` then checks the committed data is
internally consistent on every run, and skips while this directory is empty.

## Why record rather than compare live

Kaldi is bit-rotting. Recent commits cannot fetch the OpenFST version their
own source requires, and the OpenBLAS they pin no longer compiles on current
hardware. The image that still works is pinned by digest, and its output is
worth capturing while it can still be produced at all.

Recording also moves the comparison into tiers 1 and 2, which is where the
migration work happens. A PR that replaces Kaldi's feature extraction can be
gated without Kaldi being installed anywhere.

## What is in a capture

| Kind | Compared | Why |
| --- | --- | --- |
| Symbol tables, lexicon and graph FSTs as text, transcriptions | **as bytes** | no reason for a correct reimplementation to differ by a character |
| A few feature matrices in full | `rtol=1e-4` | bit-exactness is not achievable: FFT and accumulation order differ |
| The remaining feature matrices, as statistics | `rtol=1e-4` | every matrix in full would be 12MB of float32 for yesno and buy nothing over a handful |
| CMVN statistics in full | `rtol=1e-4` | two rows per speaker, so there is no reason not to |

Statistics rather than checksums for the arrays not stored whole: a checksum
of floats cannot be compared at a tolerance, so it would answer the wrong
question. The statistics include the first row's sum, which catches a time
reversal or a frame-offset error that every whole-matrix moment is blind to.

## Changing it

Don't, by hand. Re-capture, commit the new manifest, and say in the PR what
moved and why. A reference edited to make a test pass is no longer a
reference.
