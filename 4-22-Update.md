# 4/22 Update

## Current Model

Based on the previous project proposal, we have created a pipeline applying linear probes to interpret when a transformer chess model will make illegal moves based on internal state. The pipeline is split across three files:

- **`generate_games.py`** — uses the model in question to generate reasonable games, labels each move based on legality, and pulls per-layer residual stream activations.
- **`chess_gpt_probe.py`** — loads the dataset created by `generate_games.py` and trains linear probes. It reports train/test accuracy, AUROC, and test loss for each layer.
- **`chess_probe_common.py`** — shared module connecting the two scripts.

AUROC is the key metric here. Our model makes many more legal moves than illegal moves, meaning a classifier which always selects legal will perform extraordinarily well. AUROC controls for this: a value of 0.5 is chance, while values higher than this represent a linearly decodable signal.

At the moment, we have not been able to complete a sufficiently large run to obtain statistically meaningful results, but we are at a point where we will be able to complete a full model run in class on 4/23.

## What to Look For in the Full Run

### Sampling

1. **Sufficient illegal move count (> 200)** — we need to complete enough runs to establish a meaningful sample of illegal moves.
2. **Illegal rate is reasonable compared to pilot runs (~5%)**.
3. **Games show sufficient variety**, not repeating the same positions. Currently, the first few moves are random, which should accomplish this.
4. **Most games end due to a natural finish or illegal move**, as opposed to hitting our set move limit, to ensure sufficient depth.

### Probe Results

1. **AUROC should be low at embed (~0.5) and increase by the final layer.** When legality becomes linearly decodable is the motivating question. If embedding AUROC is high, it means some kind of leakage is occurring.
2. **Train/test accuracy should not show signs of overfitting** (train ≫ test, especially with the gap growing at later layers).
3. **We use multiple folds in our train/test split.** To ensure our per-layer AUROC estimates are not artifacts of a single split, we report mean and standard deviation across all folds; the std provides an uncertainty bar for comparisons across layers.

## Future Considerations

1. **Grouped k-fold splits** (by game ID rather than by position) to ensure near-duplicate positions from the same game do not appear in both train and test folds, which would inflate apparent probe accuracy.
2. **Alternative game generation methods**, such as random moves. As time allows, this would test whether the probe's signal depends on the distribution of positions Chess-GPT itself visits, or generalizes to arbitrary board states.
3. **Probability-of-illegal-move as a probe target.** We are currently using moderate-to-high temperature to generate the next move (multiple character-level tokens). This allows us to represent the distribution of results to a certain extent while saving large amounts of computation. However, using the model's probability mass on illegal moves directly (rather than sampling) would better represent the distribution as a whole. If we have time, we plan to look into implementing this.
