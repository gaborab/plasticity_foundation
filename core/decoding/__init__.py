"""
Helpers of the time-resolved location decoding (HANDOFF_location_decoding.md, Tasks A and B).
data: band-passed trial arrays (trials x channels x bands x times). models: the sliding LDA. metrics: per-subject metric
tables, latency, jackknife and the side contrasts. The entry point is decoding/run_sliding_lda.py, the analysis
notebooks/LDA.ipynb.
"""
