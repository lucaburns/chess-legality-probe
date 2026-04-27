# 4/26 Update

## Summary

The project has moved from the initial residual-stream linear probe pipeline to
a broader set of interpretability experiments around Chess-GPT move legality.

## Updated SVG Diagrams

The finalized diagrams have been placed under `docs/figures/` and the README
now links to those files. The diagrams cover:

- candidate move legality
- move history to legality labeling
- generated-move labeling pipeline
- residual-stream and MLP activation probe locations
- neuron/activation clamping intervention

## Implemented MLP Probe

`chess_gpt_mlp_probe.py` trains a small nonlinear probe on the cached residual
stream activations. It uses the same dataset format and the same basic
cross-validation setup as the linear residual-stream probe, making it a direct
comparison between linear decodability and nonlinear decodability from the same
activation source.

Current results should be read as evidence about what is decodable from
activations, not as evidence that the model internally implements the same MLP
classifier.

## Implemented Neuron-Level Probe

The neuron extension now captures post-GELU MLP activations and trains probes
on those activations block by block. The resulting `top_neurons.csv` ranks MLP
activation dimensions by mean absolute probe-weight magnitude across folds.

This ranking identifies candidate activation dimensions for further analysis.
It should not be described as proving that individual neurons definitively
encode legality.

## Implemented Neuron/Activation Clamping

`neurons-extension/clamp_neurons_experiment.py` runs interventions that clamp
selected MLP activation dimensions during generation. The selected dimensions
come from the neuron-level probe ranking, and the hooks apply at the final
token position, matching the probe measurement point for the next generated
move.

The current sweep varies both the number of ranked dimensions selected per
block and the clamp coefficient, then measures changes in generated illegal
move rate.

## Current Interpretation

The residual-stream linear probe shows legality-related information is
somewhat linearly decodable above chance in later layers. The MLP probe gives a
nonlinear comparison on the same residual-stream activations and appears
competitive with or modestly stronger than the linear probe in later layers.

The neuron-level MLP activation probe gives a way to rank activation dimensions
by probe-weight magnitude. Those rankings are useful for inspection and
intervention, but they should be treated as probe-derived candidates rather
than as clean semantic units.

The clamping experiment is the strongest causal check currently implemented.
Preliminary sweep results suggest that stronger perturbations of ranked MLP
activation dimensions can change illegal-move rates, but the effect should be
interpreted cautiously because clamping can also be a broad distribution shift.

## Next Steps

- Repeat the probe comparisons across additional random seeds and datasets.
- Add grouped splits by game ID to reduce leakage from nearby positions in the
  same generated game.
- Compare PGN self-play positions against other position distributions.
- Run more targeted clamping or activation-patching interventions around
  specific layers, blocks, and move types.
- Report uncertainty for clamping effects, not only point estimates.
