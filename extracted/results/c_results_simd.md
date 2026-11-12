N=100000, 8192-bit vectors, query noise ~18%

| store | mode | ms/query | recall(planted) | nodes fired | ops fraction | agrees w/ dense |
|---|---|---|---|---|---|---|
| uniform | dense | 8.327 | 100% | 100% | 1.000 | 100% |
| uniform | cascade m=0 | 0.755 | 100% | 0.0% | 0.125 | 100% |
| uniform | cascade m=30 | 0.778 | 100% | 0.0% | 0.125 | 100% |
| uniform | cascade m=60 | 0.787 | 100% | 0.0% | 0.125 | 100% |
| uniform | cascade m=120 | 0.777 | 100% | 0.0% | 0.125 | 100% |
| uniform | cascade m=240 | 0.774 | 100% | 0.0% | 0.125 | 100% |
| clusters of 50, 6% intra-noise | dense | 7.883 | 100% | 100% | 1.000 | 100% |
| clusters of 50, 6% intra-noise | cascade m=0 | 0.788 | 100% | 0.0% | 0.125 | 100% |
| clusters of 50, 6% intra-noise | cascade m=30 | 0.775 | 100% | 0.0% | 0.125 | 100% |
| clusters of 50, 6% intra-noise | cascade m=60 | 0.771 | 100% | 0.0% | 0.125 | 100% |
| clusters of 50, 6% intra-noise | cascade m=120 | 0.778 | 100% | 0.1% | 0.125 | 100% |
| clusters of 50, 6% intra-noise | cascade m=240 | 0.800 | 100% | 0.1% | 0.125 | 100% |
| clusters of 50, 12% intra-noise | dense | 7.602 | 100% | 100% | 1.000 | 100% |
| clusters of 50, 12% intra-noise | cascade m=0 | 0.782 | 100% | 0.0% | 0.125 | 100% |
| clusters of 50, 12% intra-noise | cascade m=30 | 0.857 | 100% | 0.0% | 0.125 | 100% |
| clusters of 50, 12% intra-noise | cascade m=60 | 0.808 | 100% | 0.0% | 0.125 | 100% |
| clusters of 50, 12% intra-noise | cascade m=120 | 0.796 | 100% | 0.0% | 0.125 | 100% |
| clusters of 50, 12% intra-noise | cascade m=240 | 0.794 | 100% | 0.1% | 0.125 | 100% |
