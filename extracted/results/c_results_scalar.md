N=100000, 8192-bit vectors, query noise ~18%

| store | mode | ms/query | recall(planted) | nodes fired | ops fraction | agrees w/ dense |
|---|---|---|---|---|---|---|
| uniform | dense | 19.964 | 100% | 100% | 1.000 | 100% |
| uniform | cascade m=0 | 1.130 | 100% | 0.0% | 0.125 | 100% |
| uniform | cascade m=30 | 1.471 | 100% | 0.0% | 0.125 | 100% |
| uniform | cascade m=60 | 1.381 | 100% | 0.0% | 0.125 | 100% |
| uniform | cascade m=120 | 1.065 | 100% | 0.0% | 0.125 | 100% |
| uniform | cascade m=240 | 1.143 | 100% | 0.0% | 0.125 | 100% |
| clusters of 50, 6% intra-noise | dense | 19.258 | 100% | 100% | 1.000 | 100% |
| clusters of 50, 6% intra-noise | cascade m=0 | 1.236 | 100% | 0.0% | 0.125 | 100% |
| clusters of 50, 6% intra-noise | cascade m=30 | 1.151 | 100% | 0.0% | 0.125 | 100% |
| clusters of 50, 6% intra-noise | cascade m=60 | 1.117 | 100% | 0.0% | 0.125 | 100% |
| clusters of 50, 6% intra-noise | cascade m=120 | 1.110 | 100% | 0.1% | 0.125 | 100% |
| clusters of 50, 6% intra-noise | cascade m=240 | 1.090 | 100% | 0.1% | 0.125 | 100% |
| clusters of 50, 12% intra-noise | dense | 18.445 | 100% | 100% | 1.000 | 100% |
| clusters of 50, 12% intra-noise | cascade m=0 | 1.106 | 100% | 0.0% | 0.125 | 100% |
| clusters of 50, 12% intra-noise | cascade m=30 | 1.091 | 100% | 0.0% | 0.125 | 100% |
| clusters of 50, 12% intra-noise | cascade m=60 | 1.135 | 100% | 0.0% | 0.125 | 100% |
| clusters of 50, 12% intra-noise | cascade m=120 | 1.153 | 100% | 0.0% | 0.125 | 100% |
| clusters of 50, 12% intra-noise | cascade m=240 | 1.162 | 100% | 0.1% | 0.125 | 100% |
