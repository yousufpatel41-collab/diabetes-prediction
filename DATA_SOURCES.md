# Data source and explicit replacement

[OpenML diabetes ID 37, version 1](https://www.openml.org/d/37), a registry copy of the historical UCI Pima Indians Diabetes dataset attributed to NIDDK. The registry labels the dataset licence “Public”; this repository does not relabel the dataset as MIT. Source retrieval and raw/CSV SHA-256 hashes are recorded in `data/source_manifest.json`.

The source contains 768 records: 500 tested negative and 268 tested positive. The ARFF file is retained alongside its deterministic CSV conversion. On 12 October 2026 this explicitly replaced the previous CSV, whose 188 positive outcomes and provenance could not be verified. All earlier results are superseded.

The historical sample covers adult women from a specific population. It does not establish performance on other populations and is not a clinical screening tool. Missing-value treatment and model selection are described in the README.
