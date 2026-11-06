
### Initial Experiments and Notes

We first tested the algorithms using synthetic data with 1,000 vectors, 32 dimensions, and 20 queries (k=10). All four algorithms ran successfully, although LSH and IVF showed relatively low recall compared to HNSW.

We then tested IVF using 10K vectors from SIFT1M, with 100 queries, k=10, and 64 clusters. We changed `nprobe` to see how the number of searched clusters affects recall and query latency.

| nprobe | Recall@10 | Query Latency (ms) | QPS |
|---|---|---|---|
| 4 | 0.904 | 0.075 | 13,330 |
| 16 | 1.000 | 0.291 | 3,435 |
| 32 | 1.000 | 0.571 | 1,751 |
| 64 | 1.000 | 1.714 | 584 |

From these initial results, increasing `nprobe` improved recall but also increased query latency. Interestingly, `nprobe=16` already achieved 100% recall for the selected queries, so searching all 64 clusters was not necessary in this case.

**Things to try next:**
- Increase the number of queries to 1,000 and test `nprobe=4, 8, 12, 16`.
- Test larger datasets (100K and 1M vectors).
- Try different numbers of clusters for IVF.
- Tune LSH parameters to improve its relatively low recall.
- Compare all four algorithms using the same datasets and evaluation metrics.
- Repeat the experiments using GloVe-100 with cosine distance.

These are only initial tests. More experiments are needed before drawing conclusions about the performance of each algorithm.
