Minimum Information Bipartition (MIB) via Arbitrary‑Conditional Normalizing Flows

Technical Specification — v1.0 (2025‑11‑08)

Owner: [TBD]
Status: Ready for implementation
Scope: Identify the Minimum Information Bipartition (MIB) of 64‑channel EEG using a single, consistent density model that supports arbitrary marginals and conditionals.
Non‑Goals: Inferring causal structure; time‑series dynamics beyond windowing; cloud/remote training.

⸻

1. Problem overview

1.1 Objective

Given multichannel EEG vectors x\in\mathbb{R}^{64} (one vector per analysis window), find a bipartition (A,B) of channels minimizing the mutual information (MI) between the two sets under a learned density model:
\[
(A^\,B^\) \in \arg\min_{A\cup B=\{1,\dots,64\},\,A\cap B=\varnothing}\ I(A;B).
\]

1.2 Motivation

Conventional MI estimators (kNN, contrastive/neural) suffer in 64D due to bias/variance and tuning instability. Normalizing flows provide exact, tractable log-densities. Using a single arbitrary‑conditional flow (ACFlow) avoids order‑dependence and enables efficient, fully batched MI evaluation for many candidate partitions.

⸻

2. Definitions & notation
	•	x \in \mathbb{R}^{D} with D=64; channels indexed by \{1,\dots,D\}.
	•	Partition (A,B): disjoint sets with union \{1,\dots,D\}.
	•	For any subset S, x_S denotes the subvector with indices in S.
	•	Binary mask m\in\{0,1\}^D: m_i=1 indicates “observed/conditioned” dimension. Observed set O=\{i: m_i=1\}; unobserved set U=O^c.
	•	All logs default to natural logs (nats). Bits = nats/\ln 2.

⸻

3. Requirements

3.1 Functional
	1.	Train a single density model p_\theta that supports \log p_\theta(x_U\mid x_O) for arbitrary (U,O) given by a mask m.
	2.	Implement an MI estimator that, for any (A,B), computes
\widehat I_\theta(A;B)=\frac{1}{M}\sum_{m=1}^M \big[\log p_\theta(x_A^{(m)}\!\mid x_B^{(m)})-\log p_\theta(x_A^{(m)})\big].
	3.	Provide a search routine (multi‑start, KL/FM‑style swaps with annealing/tabu) to minimize \widehat I_\theta(A;B).
	4.	Return the best partition, MI (nats and bits), 95% CI, and diagnostic plots.
	5.	All computations run locally; no remote downloads.

3.2 Non‑functional
	•	Reproducibility: Fixed seeds, saved splits, config logs, deterministic ops where practical.
	•	Data security: Operate on local paths; do not export datasets.
	•	Efficiency: Batched evaluation over candidate partitions (2 model passes per iteration).
	•	Correctness under selection: Final MI and CI reported on a held‑out test split untouched by the search.

⸻

4. Data & preprocessing

4.1 Input formats
	•	Primary: NumPy arrays / memmaps X.npy with shape [N,64] (one window → one row).
	•	If starting from raw time series: window into per‑window features (recommended: log bandpowers per channel) but keep a strict channel‑to‑feature mapping to define partitions over channels.

4.2 Windowing (if raw)
	•	Example: 256 Hz sampling; window length 1–2 s; 50% overlap.
	•	Featureization (recommended): per‑channel log power in canonical bands (δ, θ, α, β, γ), then aggregate back to one 64‑D vector via a stable 1:1 mapping per channel (e.g., weighted sum, PCA‑1 per channel). Record the mapping.

4.3 Cleaning & normalization
	•	Handle artifacts/NaNs: drop affected windows or interpolate locally; document policy.
	•	Per‑channel robust scaling on train (z‑score or median/IQR); apply to val/test.
	•	Do not whiten/mix across channels if the MIB is defined over original channels (mixing changes MI across sets). If whitening is used for optimization, explicitly state you are finding the MIB in the whitened basis.

4.4 Splits
	•	Train / Validation / Test by session or subject to reduce leakage (e.g., 70/15/15). Save indices and seed.

⸻

5. Model: Arbitrary‑Conditional Flow (ACFlow)

5.1 Purpose

Realize a single parameterization p_\theta capable of evaluating \log p_\theta(x_U\mid x_O) for any mask m. This ensures internal consistency across joint, marginal, and conditional terms.

5.2 Generative form

For a given mask m (observed O, unobserved U), define an invertible mapping in the U-subspace:
x_U = g_\theta(z_U; x_O, m),\quad z_U \sim \mathcal N(0,I_{|U|}),
with inverse z_U = g_\theta^{-1}(x_U; x_O, m). The conditional log-density is
\log p_\theta(x_U \mid x_O) = \log \mathcal N\big(g_\theta^{-1}(x_U; x_O, m)\big) + \log\left|\det J_{g_\theta^{-1}}(x_U; x_O, m)\right|.
When O=\varnothing, this reduces to \log p_\theta(x_U) (a true marginal).

5.3 Architecture

Coupling‑based ACFlow (RealNVP/Glow‑style with mask conditioning):
	•	Stack L coupling blocks (e.g., L=8–12).
	•	Within each block, split the current U into two parts U_1,U_2 (e.g., by a fixed 50/50 mask over U); transform U_2 conditioned on U_1, x_O, and m.
	•	Transformation (per block):
x_{U_2}’ = x_{U_2}\odot \exp s_\theta(U_1, x_O, m) + t_\theta(U_1, x_O, m),
x_{U_1}’ = x_{U_1} (pass‑through).
	•	Conditioner s_\theta,t_\theta: MLP with LayerNorm, residual MLP blocks, SiLU/ReLU; hidden size 256–512; light dropout (≤0.1).
	•	Mask conditioning: Concatenate m (or an embedding of m) and the observed values x_O to the conditioner inputs. Use zero‑imputation plus mask to indicate which entries are observed.
	•	Jacobian: Triangular in the U-ordering ⇒ log‑det is sum of s_\theta outputs over U_2.

Notes
	•	D = 64; use per‑channel affine base scaling inputs after normalization.
	•	Optional: Permute the U-ordering between blocks to improve mixing (per‑block fixed permutation of the 64 indices; at runtime, apply the subset restriction to the current U).

5.4 Alternatives (acceptable)
	•	Mask‑conditioned MADE/MAF where edges are gated by m so that observed dims precede unobserved dims. Ensure the same \theta is used for all masks.
	•	Mixture of K fixed‑order autoregressive flows (ensemble) used consistently for all terms with log‑sum‑exp; heavier and less elegant.

⸻

6. Training

6.1 Objective

Maximize conditional log‑likelihood for random masks:
\mathcal L(\theta) = \mathbb{E}{x\sim \text{train}}\ \mathbb{E}{m\sim \mathcal M}\big[ \log p_\theta(x_{U}\mid x_{O}) \big],
where \mathcal M is the mask sampler.

6.2 Mask sampler \mathcal M
	•	Sample a mixture of:
	•	Unconditional masks: O=\varnothing (train marginals).
	•	Bipartition‑like masks: O size distributed across \{8,\dots,56\} with emphasis on balanced sizes (e.g., triangular or uniform over sizes).
	•	Random subsets to cover diverse conditionals.
	•	Record the sampler seed; expose temperature/weights as config.

6.3 Optimization
	•	AdamW; lr =1\mathrm{e}{-3} (search [5\mathrm{e}{-4}, 2\mathrm{e}{-3}]), weight decay 10^{-6}–10^{-5}.
	•	Batch size: 512–2048 windows (GPU memory‑dependent).
	•	Gradient clipping (e.g., global‑norm 1.0); AMP mixed precision; Early stopping on validation NLL (patience 10–20).
	•	Save the best checkpoint (min val NLL) and full config.

⸻

7. Mutual information (math & estimator)

7.1 Definition under a single density

I_\theta(A;B)
= \mathbb{E}{x}\big[\log p\theta(x_A\mid x_B)-\log p_\theta(x_A)\big]
= \mathbb{E}{x}\big[\log p\theta(x_B\mid x_A)-\log p_\theta(x_B)\big].

7.2 Estimator (held‑out)

Given held‑out samples \{x^{(m)}\}{m=1}^M,
\widehat I\theta(A;B)=\frac{1}{M}\sum_{m=1}^M\left[\log p_\theta(x_A^{(m)}\!\mid x_B^{(m)})-\log p_\theta(x_A^{(m)})\right].
	•	Units: nats; convert to bits by dividing by \ln 2.
	•	Option: average the A\!\mid\!B and B\!\mid\!A forms to reduce variance.

7.3 Confidence intervals
	•	Use block/bootstrap appropriate to dependence structure:
	•	If windows are clustered by session/subject, resample at that granularity (session‑level bootstrap).
	•	CI method: percentile or BCa, 1000 replicates by default.
	•	Selection‑robust: Compute CI only on the test split (no reuse of the validation set used in search).

⸻

8. MIB search

8.1 Objective & constraints
	•	Primary objective: minimize \widehat I_\theta(A;B).
	•	To avoid degenerate tiny sets, choose one of:
	•	Size constraint: enforce |A|\in[k_{\min}, D-k_{\min}], e.g., k_{\min}=8.
	•	Normalized objective: report and (optionally) optimize \widehat I_\theta(A;B)/\min\{\widehat H_\theta(A), \widehat H_\theta(B)\}. (Always report raw MI as the main metric.)

8.2 Heuristics
	•	Multi‑start from balanced random splits (and optional anatomy‑informed seeds).
	•	Iterative swaps (Kernighan–Lin / Fiduccia–Mattheyses):
	•	Evaluate all 1‑swap moves in a batch; pick best gain.
	•	Occasional uphill moves via simulated annealing.
	•	Tabu list (size 5–10) to reduce cycling.
	•	Terminate when no improving move exists or max iterations reached.

8.3 Batched MI evaluation (efficiency)

For a set of candidate partitions \{(A_j,B_j)\}{j=1}^{N_c} and samples \{x^{(m)}\}{m=1}^{M}:
	•	Build two mask batches:
	•	Unconditional: m^{(j)}_{\text{un}}=\mathbf{0} for all j (targets A_j).
	•	Conditional: m^{(j)}_{\text{cd}} with ones on B_j (targets A_j).
	•	Two forward passes over the model (one for all unconditionals, one for all conditionals) returning per‑sample log‑probs for the requested target indices; then average over samples and subtract.
	•	Complexity per iteration: 2 model passes (independent of N_c) + cheap reductions.

⸻

9. System architecture & data flow

9.1 End‑to‑end flow
	1.	Data ingestion: Load preprocessed per‑window features from disk (NumPy memmap preferred) along with metadata (subject, session, state labels). Persist a manifest JSON/YAML capturing file hashes and channel order.
	2.	Splitting: Deterministically create train/val/test indices (subject‑ or session‑level blocking) and persist to `artifacts/splits/{run_id}.json`.
	3.	Normalization cache: Fit scalers on the train split only, store in `artifacts/scalers/{run_id}.pt`, and apply lazily during training/evaluation.
	4.	ACFlow training: Launch the conditional flow trainer (single script `run_analysis.py --mode train-flow`) that streams batches via PyTorch `IterableDataset`, samples masks, and checkpoints to `artifacts/checkpoints/{run_id}`.
	5.	Checkpoint selection: Pick the best validation NLL checkpoint and export a frozen evaluation bundle (`state_dict`, config, scaler, mask sampler spec).
	6.	MIB search: Invoke `run_analysis.py --mode search-mib --checkpoint <path>` which loads the frozen bundle, enumerates/searches partitions on the validation split, and produces candidate partitions with MI estimates.
	7.	Final evaluation: Re‑evaluate the top‐k partitions on the held‑out test split with bootstrap CI and store results under `artifacts/results/{timestamp}.parquet`.
	8.	Reporting & visualization: Generate plots/tables via `visualize_results.py` and embed into a markdown/HTML report for review.

9.2 Logical components
	•	Data layer: Dataset manifest, normalization cache, deterministic split registry.
	•	Model trainer: PyTorch module implementing ACFlow, mask sampler, optimization loop, early stopping, logging hooks.
	•	Evaluator: Utilities that given a checkpoint and batch of masks return per‑sample log densities (both conditional and marginal).
	•	Search engine: Kernighan–Lin/annealing hybrid operating on in‑memory bitmasks with batched log prob lookups and caching.
	•	Reporting layer: Result schema, persistence adapters (Parquet/JSON), plotting utilities (Matplotlib/Seaborn/Plotly).

9.3 Data movement & caching
	•	Primary tensors live in pinned host memory; GPU receives only the target subsets per batch to minimize PCIe load.
	•	Log prob outputs for each candidate partition are cached as `[partition_hash, log_p(X_A), log_p(X_A|X_B)]` arrays; incremental swaps reuse existing cache rows when only one channel is moved.
	•	Bootstrap resamples materialize only the partition scores (scalar per replicate) to keep memory bounded.

9.4 Observability & logging
	•	All stages log to both console and `artifacts/logs/<run_id>.jsonl` with structured entries (event, step, metrics, hash of config).
	•	Training metrics: global step, lr, train/val NLL, gradient norm, parameter norm, mask entropy.
	•	Search metrics: iteration, partition hash, MI (train/val/test), gain per swap, annealing temperature.
	•	CI metrics: bootstrap percentile bounds, number of resamples, seed.

⸻

10. Implementation details

10.1 Data interfaces & storage
	•	Input root: `data/processed/<dataset>/windows.npy` plus `windows.meta.json` describing sampling rate, window params, channel labels, subject/session IDs.
	•	Each run references data via a config object (`configs/mib_flow.yaml`) that includes paths, split keys, and preprocessing hashes; configs are versioned in git.
	•	Dataset loader exposes a `Batch` dataclass with tensors (`x`, `mask`, `conditioning_values`), metadata (subject, state), and normalization stats used.
	•	Use memory mapping (`np.memmap`) for `windows.npy` to stream large datasets without loading fully into RAM.

10.2 Flow training module
	•	Code organization: `src/eeg_analysis/flows/acflow.py` (model definition), `train_flow.py` (loop), `mask_sampler.py`.
	•	Mask sampler pseudocode:
```
def sample_mask(batch_size, dim=64):
    modes = np.random.choice(["uncond","bipart","random"], p=[0.2,0.5,0.3], size=batch_size)
    masks = np.zeros((batch_size, dim), dtype=bool)
    for i, mode in enumerate(modes):
        if mode == "uncond":
            continue
        elif mode == "bipart":
            k = triangular_sample(8, dim-8)
        else:
            k = np.random.randint(1, dim-1)
        observed = rng.choice(dim, size=dim-k, replace=False)
        masks[i, observed] = 1
    return masks
```
	•	Training loop steps: fetch batch → sample masks → split tensors into observed/unobserved sets via fancy indexing → forward through ACFlow (conditioner receives zero‑filled unobserved dims + mask) → compute loss (mean conditional log prob) → optimizer step → periodic validation on a fixed mask pool.
	•	Scheduler: cosine decay with warmup or ReduceLROnPlateau triggered by validation NLL stagnation.
	•	Checkpoint payload: model weights, optimizer, scheduler, scaler params, mask sampler config, git SHA, dependency versions.

10.3 MI evaluation & search module
	•	Represent partitions as 64‑bit bitmasks (Python `int` or `torch.bool` tensor); maintain both sets explicitly for logging.
	•	Precompute per‑partition tensors `target_idx` and `condition_idx` so the evaluator only does gather/scatter once per iteration.
	•	Batched MI evaluation API:
```
scores = evaluator.log_probs(target_idx_batch, condition_idx_batch, split="val")
mi = (scores["cond"] - scores["marg"]).mean(dim=0)
```
	•	Search loop outline:
		1.	Initialize `P` random balanced partitions (multi‑start).
		2.	For each partition, evaluate MI and populate a priority queue.
		3.	Iteratively pick the best partition, enumerate all 1‑swap neighbors (bounded by tabu list), evaluate via batched call, and accept the best improving (or annealed) move.
		4.	Track stagnation; when no improvement for `T` iterations, restart from a new seed using elite partitions plus random noise (channel shuffles).
	•	Annealing schedule: temperature T decays exponentially; uphill move probability `exp(-Δ/T)`; reset T on restart.

10.4 Orchestration & CLI
	•	Primary entry point `run_analysis.py` with subcommands/modes (`train-flow`, `search-mib`, `evaluate`, `report`).
	•	Configuration handled via OmegaConf/Hydra (or argparse + YAML) enabling overrides such as `python run_analysis.py mode=train-flow data.dataset=sedation flow.hidden=512`.
	•	All commands accept `--run-id` (default timestamp) to namespace artifacts.
	•	`visualize_results.py` loads the consolidated results parquet and generates:
		•	MI vs. iteration plots.
		•	Partition chord diagrams (channels grouped by anatomical label if provided).
		•	Bootstrap CI histograms.

10.5 Reproducibility hooks
	•	Global seeding (Python, NumPy, PyTorch, CUDA) at the start of every command; seed stored with run metadata.
	•	Config + git SHA + package versions serialized to `artifacts/config_snapshot.yaml`.
	•	Optional dry‑run mode that runs a miniature dataset (e.g., 128 windows, 32 channels) for CI.

⸻

11. Validation & testing

11.1 Data quality checks
	•	Schema validation on manifests (channel count, sampling rate, window length).
	•	Per‑channel summary stats (mean, std, missing ratio) compared against historical baselines with alert thresholds.
	•	Leakage guard: assert no subject/session overlap across splits; store hashes to guarantee immutability.

11.2 Model evaluation
	•	Track train/val NLL curves; stop if divergence >3 nats over 3 epochs.
	•	Calibration check: evaluate `\log p_\theta(x)` on a synthetic Gaussian dataset where the true log density is known to ensure the implementation is unbiased.
	•	Mask coverage diagnostic: log empirical distribution of observed set sizes; abort if any bucket (<5% of samples) is underrepresented.

11.3 MI estimator sanity
	•	Self‑consistency: verify `I_\theta(A;B)≈I_\theta(B;A)` within tolerance (≤1e-3 nats) for random partitions.
	•	Zero‑information control: shuffle samples independently across channels to destroy dependencies and confirm MI≈0.
	•	Known partition test: inject synthetic blocks with planted low/high MI to ensure the search recovers the planted split.

11.4 Testing strategy
	•	Unit tests for mask sampling, log prob API, caching, bootstrap utilities (PyTest under `tests/flows`, `tests/mib`).
	•	Integration test that runs a miniature flow (D=4) end‑to‑end through search and reporting; executed in CI.
	•	Regression suite capturing MI outputs for frozen checkpoints to detect future drift.

⸻

12. Compute & resource plan

12.1 Hardware
	•	Training: 1× NVIDIA RTX 4090 or A5000 (24 GB) or equivalent; mixed precision keeps memory <12 GB for batch 1024.
	•	Evaluation/search: GPU‐optional; CPU fallback uses vectorized PyTorch on MKL but ~5× slower. Target 32‑core CPU, 128 GB RAM for bootstrap heavy workloads.

12.2 Runtime estimates (for N=200k windows)
	•	Data loading + normalization cache: <10 min.
	•	Flow training (100 epochs, batch 1024): ~6 hrs on 4090.
	•	MIB search (50 multi‑starts, 200 iterations each): ~2 hrs GPU, ~8 hrs CPU.
	•	Bootstrap CI (1000 replicates, cached log probs): 30–45 min CPU.

12.3 Storage
	•	Dataset: ~5 GB per sedation/awake pair.
	•	Checkpoints: ≤500 MB per run (best + last + optimizer).
	•	Logs/results: <100 MB per run.
	•	Cleanup policy: retain best checkpoint + result summary; archive raw logs >30 days.

12.4 Failure handling
	•	Trainer resumes from the latest checkpoint (tracked in `artifacts/last.ckpt`).
	•	Evaluator is stateless; rerunning with the same `run-id` overwrites caches only after success to avoid corruption.
	•	Bootstrap supports chunking/resume by persisting partial replicate results.

⸻

13. Risks & mitigations
	1.	Model underfits or collapses on hard masks → Increase conditioner capacity, add mask‑dropout regularization, monitor per‑mask losses.
	2.	Data scarcity for certain states → Use subject‑level cross‑validation, augment with noise injection limited to within channels, or aggregate across compatible sessions.
	3.	Search stagnates in local minima → Increase multi‑starts, incorporate larger swap moves (2‑opt), tune annealing schedule, or hybridize with genetic search for diversification.
	4.	Estimator bias from model misspecification → Validate with synthetic datasets where ground truth MI is computable; if needed, ensemble multiple flow checkpoints and average MI.
	5.	Compute/resource constraints → Provide knobs for dimensionality reduction (subset of channels) and lower batch sizes; support CPU fallback albeit slower.
	6.	Reproducibility drift → Enforce config snapshots, hashed datasets, CI tests on miniature runs.

⸻

14. Milestones & deliverables

| Phase | Duration | Key outputs |
| --- | --- | --- |
| 0. Data readiness | 1 week | Finalized manifests, normalization cache, deterministic splits, QC report. |
| 1. ACFlow baseline | 2 weeks | Working training loop, mask sampler, baseline checkpoint with target val NLL, logging dashboard. |
| 2. MIB search engine | 2 weeks | Batched evaluator, swap/annealing search, caching layer, preliminary partitions on validation split. |
| 3. Test split + CI | 1 week | Held‑out MI estimates with bootstrap CI, reproducibility scripts. |
| 4. Reporting & polish | 1 week | Visualization bundle, markdown/HTML report, README updates, final tech spec review. |

Critical exit criteria:
	•	Validated ACFlow checkpoint meeting val NLL target.
	•	Search recovers low‑MI partitions consistently across seeds.
	•	Test split MI + CI reported with reproducibility artifacts.
	•	Documentation + automation ready for handoff.
