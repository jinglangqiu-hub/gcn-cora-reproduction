# Reproducibility scope

The historical seed=0 full reruns used fresh Python processes and random initialization on the same Windows machine/environment. GCN 200 epochs / test 80.4%, MLP 22 epochs / test 21.6%. Every epoch metric except wall time, model/Adam/RNG tensor, full-graph logit and test probability matched exactly; saved prediction CSVs were byte-identical.

Tolerances declared before that check: tensor rtol=1e-5/atol=1e-7, metric absolute tolerance 1e-7. That check does not claim fresh environment installation, new operating system/GPU, or cross-framework bitwise equality. Public packaging validation is separate and is documented in publication_validation.json.
