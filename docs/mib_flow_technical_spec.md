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

