# Domain-Weighting Function Design (frozen BEFORE test-set evaluation)

All hyperparameter candidates below were chosen from distributional inspection of the estimated pool and the 142 refrigerants (unlabeled structures/composition only), and selected using GLOBAL validation Tc RMSE (n=209) only. No frozen-test labels or the tiny refrigerant-like validation subgroup (16 MEDIUM, 0 HIGH of 209) were used for selection.

Anchors (fixed by construction, not searched): W_MIN=0.2 (= existing XGB-D uniform weight), W_MAX=1.0 (full experimental trust).

## D2 — similarity-based smooth weight
w_i = W_MIN + (W_MAX-W_MIN) * s_i^p,  s_i = max Morgan-Tanimoto(estimated molecule i, 142 refrigerants)
Candidates: p in [1.0, 2.0] (linear / quadratic).
Est-pool s_i distribution: median=0.1765, MEDIUM/HIGH count=75/5554.

## D3 — composition-domain distance, Gaussian kernel
z1=(halogen_fraction-mu_hf)/sd_hf, z2=(F_to_C-mu_fc)/sd_fc  [mu,sd from the 142-refrigerant population]
d_i = sqrt(z1^2+z2^2);  w_i = W_MIN + (W_MAX-W_MIN) * exp(-d_i^2 / (2*bw^2))
Refrigerant centroid: halogen_fraction=0.5047+-0.1895, F_to_C=1.5049+-0.9260.
Cl_to_C excluded from the metric: refrigerant-vs-train standardized difference was only -0.25 sigma (see 05_refrigerant_domain_audit.py output 02) — not a differentiating axis for this population, unlike halogen_fraction (+1.93 sigma) and F_to_C (+1.84 sigma).
Candidates: bandwidth in [1.5, 2.5] SD units.

## D4 — hard refrigerant-like subset
Same d_i as D3. Include estimated molecule i iff d_i <= radius; included rows get weight=W_MAX, excluded rows get weight=0 (equivalent to removal for this target's training pass).
Candidates: radius in [1.5, 2.0] SD units (1.0 SD was inspected and rejected pre-registration: only 21 estimated molecules, judged too small to train a stable subset).

## Winning candidates (by global validation RMSE)
- D2: 1.0
- D3: 2.5
- D4: 2.0