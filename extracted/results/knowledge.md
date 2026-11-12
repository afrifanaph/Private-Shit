# Phase D: sharded holographic knowledge memory

Synthetic graph: 5000 entities, 16 relations, 20,000 facts (4 per subject, functional relations). Queries: 1,500 per cell; multi-hop: 1,000 paths. Dense cleanup unless stated.

An explicit table needs at least 17 bits/fact (relation id + object id, ignoring indexing). Holographic cost is D / (facts per shard).

## K1. 1-hop accuracy vs shard load and dimension

| D (bits) | facts/shard (target) | shards | bits/fact | persistent bytes | 1-hop accuracy |
|---|---|---|---|---|---|
| 4096 | 16 | 1250 | 256 | 678,890 | 100.0% |
| 4096 | 32 | 625 | 128 | 358,890 | 99.9% |
| 4096 | 64 | 312 | 64 | 198,634 | 98.1% |
| 4096 | 128 | 156 | 32 | 118,762 | 75.3% |
| 4096 | 256 | 78 | 16 | 78,826 | 31.7% |
| 8192 | 16 | 1250 | 512 | 1,318,890 | 100.0% |
| 8192 | 32 | 625 | 256 | 678,890 | 100.0% |
| 8192 | 64 | 312 | 128 | 358,378 | 99.9% |
| 8192 | 128 | 156 | 64 | 198,634 | 98.5% |
| 8192 | 256 | 78 | 32 | 118,762 | 77.5% |
| 16384 | 16 | 1250 | 1024 | 2,598,890 | 100.0% |
| 16384 | 32 | 625 | 512 | 1,318,890 | 100.0% |
| 16384 | 64 | 312 | 256 | 677,866 | 99.9% |
| 16384 | 128 | 156 | 128 | 358,378 | 99.9% |
| 16384 | 256 | 78 | 64 | 198,634 | 99.5% |

## K2. Multi-hop accuracy (error compounds per hop)

| D | facts/shard | 1 hop | 2 hops | 3 hops | 4 hops |
|---|---|---|---|---|---|
| 8192 | 32 | 100.0% | 100.0% | 99.9% | 99.8% |
| 8192 | 64 | 99.9% | 100.0% | 99.7% | 99.7% |
| 16384 | 64 | 99.9% | 100.0% | 99.9% | 99.7% |

## K3. Corruption tolerance (fraction of stored shard bits flipped), D=8192, 32 facts/shard

| bits flipped | 1-hop | 2-hop |
|---|---|---|
| 0% | 100.0% | 100.0% |
| 5% | 100.0% | 100.0% |
| 10% | 100.0% | 100.0% |
| 20% | 99.6% | 98.8% |
| 30% | 80.5% | 70.4% |
| 40% | 14.6% | 2.5% |

## K4. Event-driven cleanup (cascade over the 5,000-entity item memory), D=8192, 32 facts/shard

Dense cleanup compares the query against all entities; the cascade first compares a prefix and only fully compares entities that 'fire'.

| mode | prefix bits | 1-hop accuracy | entity-compare ops vs dense | ms/query |
|---|---|---|---|---|
| dense | - | 100.0% | 1.00 | 1.23 |
| cascade z=2.0 | 2048 | 99.9% | 0.25 | 0.48 |
| cascade z=3.0 | 2048 | 100.0% | 0.27 | 0.50 |
| cascade z=2.0 | 4096 | 100.0% | 0.50 | 0.74 |
| cascade z=3.0 | 4096 | 100.0% | 0.50 | 0.73 |