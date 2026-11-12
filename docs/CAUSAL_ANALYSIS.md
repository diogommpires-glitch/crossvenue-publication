# Advisory venue lead/lag research

Recovered from a misplaced NetScanner prototype on 8 November 2026. Run commands
from the installed project root. The source export is a separate tree.

## Usage

From the installed project, using its dedicated environment:

```powershell
.\.venv-paper\Scripts\python.exe -B -m src.runtime.paper_cli causal tests/fixtures/representative_recording
.\.venv-paper\Scripts\python.exe -B -m src.runtime.paper_cli causal <recording-directory> --output <new-report.json> --state-out <new-research-state.json>
```

Existing output files are never overwritten. No providers, account engines,
orders or financial databases are opened. Recording checksums, completeness and
arrival order use the existing validated reader. Provider generations and invalidation
clear only that provider's history; healthy status messages preserve it.
The cross-asset runner also publishes advisory research in **System → Lead/lag
research**. No strategy, routing, readiness or risk decision consumes these scores.

## Meaning and limits

- Stream identity is `(provider_id, instrument_id)`, falling back to `event.source`
  when a recording has no binding. Generation changes clear that provider only.
- Inputs are trades or complete, non-crossed book snapshots. Deltas alone,
  nonpositive prices and unsupported event kinds do not create returns. The default
  `first` policy locks each stream to its first accepted price kind, rather than
  mixing trade prices and book midpoints. `trade` and `midpoint` policies are explicit
  alternatives. Duplicate/reordered sequences are omitted when sequences exist;
  unsequenced events are not claimed to be deduplicated.
- Processing epoch seconds define one-second buckets by default. The last
  observed price in each **closed** bucket forms simple price-change basis points.
  No forward filling across gaps and no reordering by provider/source clocks.
- A default 300-bucket window, 64 streams and 10 paired samples bound analysis.
  Capacity omissions are explicit. Sparse histories have null research priorities.
  The newest closed return must be within five seconds to remain usable. Runtime limits
  are deliberately lower: eight streams, 120 buckets, cached summaries at most
  once per two seconds. Diagnostics expose capacity and input omissions.
- `input_age_seconds` measures actual last accepted input arrival. It is separate
  from `closed_return_age_seconds`, also exposed as `age_seconds` for compatibility.
  A fresh observation in an open bucket cannot refresh an old closed return.
  Each edge additionally reports `paired_age_seconds` and `paired_stale`, using
  its latest actual source-target aligned pair. Stale pairs cannot be discoveries
  or contribute to outbound lead strength, even if both individual streams updated.
- Shannon entropy is normalized over five fixed return bins. Bayesian estimates
  mean probability of a **positive observed return**, not profitable execution.
  Prior-relative forgetting preserves positive Beta parameters. Reported variance
  is uncertainty, not a performance confidence interval.
- FAST/SLOW/TRANSITION regimes use recent return standard deviation (over 10 bps,
  below 1 bp, or between). Decay half-lives are 30/300/165 seconds. These are
  explicit research defaults, not calibrated trading recommendations.
- Edges compare source returns at bucket t-1 with target returns at t. Missing
  pairs and constant series do not invent correlations. This is lagged Pearson
  correlation, **not a Granger test or causal proof**. Cross-asset comparisons do
  not imply fungibility, currency equivalence or an arbitrage opportunity.
- `edges` retains descriptive Pearson output. A separate `discoveries` list uses
  weighted Pearson, Kish effective sample size with a capped Bartlett correction
  over ten physical bucket lags, at least 32 effective pairs (30 degrees of freedom),
  and exact Benjamini-Yekutieli step-up at q=.01. The prespecified family includes
  every retained directed stream pair at one bucket lag; missing, constant, stale
  and insufficient pairs contribute no rejection. Missing buckets stay masked.
  Serial-adjusted Student p-values are explicitly approximate and uncalibrated:
  this does not guarantee FDR control for market data. `discovery_eligible` and
  `discovery` are distinct, and every edge remains `actionable: false`.
  No Hayashi-Yoshida/Group-Lasso model, raw OFI, liquidation data, trading signal,
  strategy adapter or profitable predictive capability is asserted or enabled.
- Priority is direction-neutral: decay times 40% entropy + 30% normalized Beta
  variance + 30% mean absolute outbound correlation. It is not expected P&L.
- Save/load retains bounded observation history, parameters and processing time
  in an exclusively created, checksummed research JSON file. Restoring this state
  never changes the paper ledger. `reset()` clears all research state. The Python
  API and CLI can restore and continue on a later recording. State version 3 adds
  exact input arrival times. Version 2 loads explicitly with unknown input times
  until each stream receives a new accepted input; bucket boundaries are never
  reported as exact arrival times. Price history, kinds, generations, sequences
  and processing clock are preserved. Timer/provider tails advance the saved
  continuation clock even without new market observations. Version 1 is rejected
  because its mixed-price history cannot safely establish a price policy.
  Atomic publication uses a flushed temporary file and exclusive hard link in
  the same directory. Filesystems without hard-link support fail explicitly.
- Entropy, information gain, cross entropy and KL helpers validate distributions.
  Graph centrality and semantic scores were placeholders and are not fabricated.
  Thompson-sampling actions and approximate confidence intervals from the original
  prototype are not enabled as trading decisions.

Tests use synthetic market events and temporary files. They do not establish
live acceptance, profitable predictability, or statistically significant causality.

## Continuation, configuration and calibration

```powershell
.\.venv-paper\Scripts\python.exe -m src.runtime.paper_cli causal <later-recording> --state-in <state-v2-or-v3.json> --state-out <new-state-v3.json>
.\.venv-paper\Scripts\python.exe -m src.runtime.paper_cli causal <recording> --evaluate-direction --output <new-report.json>
```

`--research-config` accepts a separate JSON object with exactly these fields:
`schema_version: 2`, `bucket_seconds: 1`, `max_buckets: 300`, `min_samples: 10`,
`max_streams: 64`, `forgetting: 0.95`, `price_policy: "first"`,
`freshness_seconds: 5`. It cannot be combined with restored state. History is
bounded to 10,000 buckets and 64 streams; runtime uses the smaller limits above.

Direction evaluation uses frozen defaults and chronological 60/20/20 intervals.
Each forecast is made after a closed bucket and scored only against the next
closed bucket, omitting gaps and split-crossing returns. Brier scores compare the
decayed positive-return probability to an unchanged 0.5 forecast. This evaluates
direction calibration only, not lag-edge significance or execution performance.
There is one configuration trial and no automatic promotion or paper adapter.
Execution evaluation subtracts all external flows between valid equity endpoints.
An invalid interior valuation makes return, drawdown and training score unavailable,
and that interval cannot select a candidate. Exact endpoint net PnL may still be
reported. Fees, fills, latency, inventory and markouts remain separate requirements
before proposing an adapter.

Runtime observation is confined to engine-accepted events on the account writer.
Research exceptions disable research only. Summaries expose measured compute time;
a summary exceeding 50 ms disables further research computation for the session.
This is a measured guard after a computation, not a pre-emptive scheduling deadline.
Local Windows/Python 3.11 checks of 130 complete synthetic buckets measured medians
of roughly 1.8/7.9/36.6 ms for 8/25/64 streams, including the full one-lag family.
The largest 64-stream sample was 54.6 ms, exceeding the runtime guard; runtime's
eight-stream cap remains necessary. Measurements vary with concurrent load.
These are workload-specific summary measurements, excluding ingest and concurrent
writer/UI load; they do not establish worst-case latency at the 10,000-bucket cap.
Persistence is CLI-only and separate from the ledger; runtime does not restore or
save research implicitly, including on shutdown. No application restart is needed
for tests; a currently running instance will not acquire this code until a user restart.
