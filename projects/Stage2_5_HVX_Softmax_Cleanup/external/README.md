# External source area

`scripts/fetch_upstream.sh` creates a clean checkout at `external/llama.cpp`
and verifies commit `0a50d9909a3478e82679f505bf8595d1eee4b0a8` before applying any
project-owned benchmark patch. The checkout and all of its build products are
ignored; provenance is captured in each external-baseline run directory.

