# V81 SCNA HVX Static Assembly Check

Counts come from the final packetized v81 ELF. `vlut16` is used only for branchless register-indexed tree traversal; it is not the LUT exponential backend.

| Precision | Width | Direct bytes | Tree bytes | Size reduction | Direct inst. | Tree inst. | Inst. reduction | Direct mul | Tree mul | Tree lookup | Tree inst./packet |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP16 | d8 | 612 | 364 | 1.68x | 153 | 91 | 1.68x | 24 | 4 | 11 | 2.53 |
| FP16 | d16 | 1196 | 456 | 2.62x | 299 | 114 | 2.62x | 48 | 4 | 19 | 2.33 |
| FP16 | d32 | 2344 | 668 | 3.51x | 586 | 167 | 3.51x | 96 | 4 | 35 | 2.39 |
| INT8 | d8 | 656 | 544 | 1.21x | 164 | 136 | 1.21x | 10 | 4 | 11 | 2.39 |
| INT8 | d16 | 1072 | 632 | 1.70x | 268 | 158 | 1.70x | 18 | 4 | 19 | 2.29 |
| INT8 | d32 | 1904 | 808 | 2.36x | 476 | 202 | 2.36x | 34 | 4 | 35 | 2.27 |

The table is a static resource-pressure check, not a cycle model. Runtime qtimer measurements remain authoritative for latency because cache state, call overhead, and instruction issue constraints are device-dependent.
