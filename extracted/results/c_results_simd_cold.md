N=100000, 8192-bit vectors, query noise ~18%, cache=cold (64MB flushed before every query)
| store | mode | ms/query | recall(planted) | nodes fired | ops fraction | agrees w/ dense |
|---|---|---|---|---|---|---|
| uniform | dense | 7.935 | 100% | 100% | 1.000 | 100% |
| uniform | cascade m=0 | 1.215 | 100% | 0.0% | 0.125 | 100% |
| uniform | cascade m=30 | 1.121 | 100% | 0.0% | 0.125 | 100% |
| uniform | cascade m=60 | 1.368 | 100% | 0.0% | 0.125 | 100% |
| uniform | cascade m=120 | 1.249 | 100% | 0.0% | 0.125 | 100% |
| uniform | cascade m=240 | 1.386 | 100% | 0.0% | 0.125 | 100% |
| clusters of 500, 3% intra-noise (near-duplicates) | dense | 8.657 | 100% | 100% | 1.000 | 100% |
| clusters of 500, 3% intra-noise (near-duplicates) | cascade m=0 | 1.174 | 100% | 0.0% | 0.125 | 100% |
| clusters of 500, 3% intra-noise (near-duplicates) | cascade m=30 | 1.216 | 100% | 0.1% | 0.126 | 100% |
| clusters of 500, 3% intra-noise (near-duplicates) | cascade m=60 | 1.374 | 100% | 0.5% | 0.129 | 100% |
| clusters of 500, 3% intra-noise (near-duplicates) | cascade m=120 | 1.261 | 100% | 0.5% | 0.129 | 100% |
| clusters of 500, 3% intra-noise (near-duplicates) | cascade m=240 | 1.607 | 100% | 0.5% | 0.129 | 100% |
