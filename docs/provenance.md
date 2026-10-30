# Public-export provenance

Original scientific runs are preserved locally and are not overwritten by this release. `export_manifest.json` maps all 20 original checkpoints to their public exports, including original and public SHA-256 values. The four model/data source files are copied byte for byte; their hashes are recorded there.

Public adaptations: relative/configurable paths; an independent run identity instead of original-machine environment equality; a pinned data downloader and verifier; independent result summarization; portable plotting; localhost monitor plus a GitHub Pages static archive. These are new publication tools, not the frozen training entry point with an unchanged hash.

For exported checkpoints only `config.data_directory` is made relative. Model weights, Adam states, Torch RNG states and epoch remain unchanged. Whole checkpoint hashes consequently differ. CSV public checkpoint hashes refer to public bytes; `original_checkpoint_sha256` retains the original identity. Predictions and metric CSVs are copied without numerical changes. Logs/metadata have local absolute paths removed. The public report retains scientific content and uses public reproduction commands in its final section.

Model training algorithm, dropout placement, Adam updates, loss, early-stop indexing and selected last checkpoint are retained from the original loop. New runs never append to historical `results/test_runs.csv`. Portability validation results are recorded separately in `docs/publication_validation.json` before release.

Historical report v1.1 PDF SHA-256: `1ef3cecb8c11ab6bb50c9c86ef31784fe0cd06be3826006f4ddca0036482d2d4`. Original paper/whole reference repository/private learning notes/local backup ZIPs are not redistributed.
