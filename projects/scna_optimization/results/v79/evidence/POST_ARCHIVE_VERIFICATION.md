# Post-archive verification

- All nine historical SCNA project directories are present under `projects/Archived/` and absent from the project root.
- `scripts/build.sh` completed after archival without reading an archived path.
- Every rebuilt reference and SCNA artifact was byte-identical to the artifact used for the recorded benchmark.
- SCNA environment smoke passed after the rebuild; SCNA q4 measured request returned `ret=0`.
- The rebuilt original reference bundle produced `ret=0` for both baseline and LUT-exp q4 smoke requests.
- Correctness parser: 180/180, determinism 10/10 with one checksum.
- Benchmark parser: 1200/1200 valid Host samples, all `ret=0`.
- Static gate: 112 instructions, 36 packets, 0 spill, 0 B frame.
- Reference source-tree provenance gate: exact match.

Rebuilt DSP artifact SHA256:

```text
reference d31df325eb0d93a86450fa7dc25a2bfc5be2eedc4d5001c37d69e1adfdc82624
scna      fe5c8475c00dc390664c64f60d61e069ffdff2429b5e484c91bf679b80c7b6ef
```
