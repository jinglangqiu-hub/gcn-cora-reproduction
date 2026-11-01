# Release artifacts

Download `cora_evidence_v0.1.0.zip`, `GCN_Cora_Report_v1.1_public.pdf`, and `SHA256SUMS.txt` from [https://github.com/jinglangqiu-hub/gcn-cora-reproduction/releases/tag/v0.1.0](https://github.com/jinglangqiu-hub/gcn-cora-reproduction/releases/tag/v0.1.0).

The English report is a separate [PDF download](https://github.com/jinglangqiu-hub/gcn-cora-reproduction/releases/download/v0.1.0/GCN_Cora_Report_v1.1_public_en.pdf), with its checksum in [SHA256SUMS_en.txt](https://github.com/jinglangqiu-hub/gcn-cora-reproduction/releases/download/v0.1.0/SHA256SUMS_en.txt). It contains 9 pages, all 9 tables, and the 5 English figures. The original Chinese PDF and evidence assets remain unchanged. The [PDF build manifest](../docs/report_en_pdf_manifest.json) records the source Markdown, builder, figure, and PDF hashes.

The evidence archive contains all 20 selected last checkpoints, all 20 prediction CSVs (1000 nodes each), epoch metrics, logs, configs, result records and an internal SHA-256 manifest. Verify archive bytes against SHA256SUMS.txt before extraction. Checkpoint exports retain tensor states but replace machine-specific data paths; see docs/export_manifest.json. Extract under a short path if Windows rejects deep directory names.
