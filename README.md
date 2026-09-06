# Recurrence vs. Attention on WikiText-2: A Controlled Comparison

**Question**: Does architectural inductive bias (recurrence vs. attention) matter more than parameter count for language modeling, and how does that tradeoff shift with training budget?

**Status**: Results below are from the first real run under the fixed pipeline (seed=1337, seq_len=6, full scale: 25K sentences, 10 epochs). One seed, one context length -- the 3-seed and long-context (seq_len=30) conditions haven't been run yet, so treat directional findings as a first data point, not a final answer.

## Setup

- **Data**: WikiText-2 (raw), word-level tokenization, 25K/5K/5K sentence train/val/test subsample.
- **Task**: sliding-window next-word prediction, primary context length 6 (a second, longer condition at 30 is designed in but not yet run).
- **Models**: LSTM, GRU, Transformer (causal, 2-layer), parameter-matched, Adam, gradient clipping, 10 epochs.
- **Controls added**: fixed seed, per-architecture learning rate, CPU-time-based epoch timing (wall-clock was previously contaminated by background system load), per-epoch validation checkpointing, and a parameter-matching search for the transformer.

## Results (seed=1337, seq_len=6)

| Model       | Val perplexity | Params    |
| ----------- | -------------- | --------- |
| LSTM        | 3491           | 9,276,844 |
| GRU         | 3558           | 9,247,404 |
| Transformer | **2730**       | 8,376,412 |

- The transformer reaches ~22% lower validation perplexity than LSTM **despite having ~9.7% fewer parameters** (the param-matching search improved the original ~11% gap but didn't fully close it).
- **New**: a perplexity-vs-training-budget curve (previously impossible with the original one-epoch-count design) shows the transformer's advantage holds across the _entire_ training run, not just at epoch 10, and widens as training continues, since all three architectures overfit as training progresses (train loss ~4.1-5.1 vs. val/test loss ~7.9-8.2 for all three) but the transformer overfits more slowly.
- LSTM and GRU are within 2% of each other on perplexity and track almost identically on the budget curve -- the recurrence-vs-attention divide looks much bigger here than the LSTM-vs-GRU choice.
- The wall-clock timing bug is directly confirmed: this run, it was the _transformer's_ wall-clock time that swung wildly (200s-560s/epoch) while its CPU time stayed stable (368s-427s) -- last run it was GRU that swung. A different architecture affected each time is exactly what you'd expect from background load, not an architecture property.

## Bugs found and fixed (during the original review)

| Issue                                                          | Fix                                                                |
| -------------------------------------------------------------- | ------------------------------------------------------------------ |
| No random seed anywhere                                        | `set_seed()` before each model; run designed for 3-seed comparison |
| Shared LR across all architectures                             | Per-architecture `LR_BY_ARCH`                                      |
| Wall-clock timing confounded by background load                | Switched to `time.process_time()` (CPU time) -- confirmed above    |
| Training budget never varied                                   | Per-epoch val-perplexity checkpointing -> budget curve             |
| Param counts differed with no correction                       | Search over transformer `dim_feedforward`; improved 11%->9.7% gap  |
| Checkpoints overwritten across runs                            | Seed/seq_len-tagged filenames                                      |
| Undefined `colors` variable (`NameError` on fresh run)         | Defined explicitly                                                 |
| 4-5 cells only worked due to leftover interactive kernel state | Values passed as explicit function arguments; cells reordered      |

## Still open

- Multi-seed variance (3 seeds designed, 1 run so far) and the long-context (seq_len=30) condition are not yet run -- that's the next step before treating any of the above as a stable finding rather than a first result.
- Parameter match landed at 9.7% off target (1% tolerance configured but not reached in the search range used).

## Files

- `wikitext2_exploration_fixed.ipynb` -- full pipeline, GPU-ready, verified to execute cleanly end-to-end.
- `README.md` -- full-detail version with per-fix rationale and complete results/notes.

## References

Merity, S., Xiong, C., Bradbury, J., & Socher, R. (2016). _Pointer Sentinel Mixture Models._ (introduces WikiText-2)

Vaswani, A., et al. (2017). _Attention Is All You Need._

Hochreiter, S., & Schmidhuber, J. (1997). _Long Short-Term Memory._
