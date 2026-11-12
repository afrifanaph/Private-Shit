N=100000, 8192-bit vectors, query noise ~18%, cache=cold (64MB flushed before every query)
| store | mode | ms/query | recall(planted) | nodes fired | ops fraction | agrees w/ dense |
|---|---|---|---|---|---|---|
| uniform | dense | 19.123 | 100% | 100% | 1.000 | 100% |
| uniform | cascade m=0 | 1.580 | 100% | 0.0% | 0.125 | 100% |
| uniform | cascade m=30 | 2.073 | 100% | 0.0% | 0.125 | 100% |
| uniform | cascade m=60 | 2.081 | 100% | 0.0% | 0.125 | 100% |
| uniform | cascade m=120 | 2.027 | 100% | 0.0% | 0.125 | 100% |
| uniform | cascade m=240 | 2.175 | 100% | 0.0% | 0.125 | 100% |
| clusters of 500, 3% intra-noise (near-duplicates) | dense | 18.272 | 100% | 100% | 1.000 | 100% |
| clusters of 500, 3% intra-noise (near-duplicates) | cascade m=0 | 1.829 | 100% | 0.0% | 0.125 | 100% |
| clusters of 500, 3% intra-noise (near-duplicates) | cascade m=30 | 1.881 | 100% | 0.1% | 0.126 | 100% |
| clusters of 500, 3% intra-noise (near-duplicates) | cascade m=60 | 1.877 | 100% | 0.5% | 0.129 | 100% |
| clusters of 500, 3% intra-noise (near-duplicates) | cascade m=120 | 1.853 | 100% | 0.5% | 0.129 | 100% |
| clusters of 500, 3% intra-noise (near-duplicates) | cascade m=240 | 1.811 | 100% | 0.5% | 0.129 | 100% |
