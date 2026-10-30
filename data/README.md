# Data preparation

Run `python -m tools.prepare_data` from the project root. It downloads the eight Cora files from the pinned tkipf/gcn commit, checks every SHA-256 in configs/cora_raw_sha256.json before deserializing, aligns the test indices and writes verified processed arrays. Existing processed output is never overwritten; select a new --output-dir to regenerate. Raw files and generated arrays are ignored by Git.

The original Windows checkout stored the single text file `ind.cora.test.index` with CRLF line endings. The downloader explicitly normalizes only this file to CRLF before comparing the recorded original SHA-256. Index values/order are unchanged; the other seven binary files are checked without transformation.

Original feature/adjacency verification used independent float64 references, maximum errors approximately 9.93e-9 / 2.98e-8. Three masks are disjoint and have 140/500/1000 nodes. The public preparation repeats shape, alignment-index, per-class label count, graph, normalization and finite-number assertions.
