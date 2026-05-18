# Runtime Stagewise Context

| Stage | Full KAFN | Budget-Prune | Gate-L1 | Notes |
| --- | --- | --- | --- | --- |
| train full dictionary (sec) | 11.93 +/- 1.55 (n=3) | 11.83 +/- 1.55 (n=3) | 13.32 +/- 2.50 (n=3) | dataset=susy_binary_200000; budget=200; manifests_used=3 |
| H-build/export (sec) | 10.41 +/- 0.03 (n=3) | 4.06 +/- 0.10 (n=3) | 3.96 +/- 0.50 (n=3) | dataset=susy_binary_200000; budget=200; manifests_used=3 |
| selection (sec) | 0.00 +/- 0.00 (n=3) | 0.03 +/- 0.02 (n=3) | 0.03 +/- 0.02 (n=3) | dataset=susy_binary_200000; budget=200; manifests_used=3 |
| refit head (sec) | 0.00 +/- 0.00 (n=3) | 2.31 +/- 0.03 (n=3) | 2.70 +/- 0.02 (n=3) | dataset=susy_binary_200000; budget=200; manifests_used=3 |
| inference (ms/sample) | 0.0001 +/- 0.0000 (n=3) | 0.0001 +/- 0.0000 (n=3) | 0.0001 +/- 0.0000 (n=3) | dataset=susy_binary_200000; budget=200; manifests_used=3 |
| memory peak (MB) | 2717.77 +/- 12.90 (n=3) | 2438.06 +/- 16.46 (n=3) | 2443.55 +/- 19.16 (n=3) | dataset=susy_binary_200000; budget=200; manifests_used=3 |
