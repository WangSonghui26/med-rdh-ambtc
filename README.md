# MED-Predictive Coding for High-Capacity Reversible Data Hiding in AMBTC Compressed Images

Reference implementation and per-image experimental data for the manuscript
submitted to *IEEE Signal Processing Letters*.

All numbers reported in the paper are produced by the scripts in this
repository from the twelve standard benchmark images. No result is fabricated;
the JSON files are the raw outputs.

---

## 1. What this repository contains

| File | Role | Maps to |
|------|------|---------|
| `run_experiments.py` | Core AMBTC codec + MED/JNC/avg/left predictors + six-tier variable-length XOR code. Runs the main predictor ablation across 12 images. | Table (predictor ablation), Table (capacity), Fig. results |
| `ablation_code_tiers.py` | Controlled **code** ablation: faithful four-tier (Chang 2018) vs proposed six-tier, holding the MED predictor fixed. Includes an integrity check that the six-tier run reproduces `experiment_results.json`. | Code-tier ablation result (190,634 vs 196,902 bits, +3.3%) |
| `stego_payload_detectability.py` | Compressed-domain passive steganalysis: chi-square bit-balance + order-1 Markov entropy on the payload field, raw vs whitened. | Security analysis (raw 12/12 detected, whitened 0/12) |
| `security_analysis.py` | Pixel-domain statistics: histogram, chi-square, RS analysis, bitstream statistics. | Supplementary security discussion |
| `alg_ambtc.py`, `alg_ambtc_predict.py` | Standalone AMBTC compression / predictive-coding utilities (reference building blocks). | Method description |

Result data (already computed, committed so you can verify without running):

| File | Contents |
|------|----------|
| `experiment_results.json` | Ground-truth per-image capacity for the four predictor variants (jnc/med/avg/left) under the six-tier code. MED avg = 196,902.5 bits; JNC avg = 184,594.3 bits. |
| `ablation_code_tiers.json` | Per-image bits for six-tier vs four-tier under fixed MED. avg_six = 196,902; avg_four = 190,634; gain = +3.3%; all reversible. |
| `stego_payload_detectability.json` | Per-image chi-square and H1 for raw vs whitened payloads. |
| `security_results.json` | Pixel-domain security statistics. |

---

## 2. Requirements

- Python 3.8+
- `numpy`
- `Pillow` (imports as `PIL`)

```bash
pip install numpy Pillow
```

---

## 3. Test images

The scripts expect twelve 512x512 8-bit grayscale BMP images under `pics/`.
They are **not** redistributed here (copyright); see `pics/README.txt` for the
exact expected file names and layout. Place your copies there before running.

---

## 4. Running

From the repository root:

```bash
# Main predictor ablation (JNC vs MED vs avg vs left) -> experiment_results.json
python run_experiments.py

# Code-tier ablation (four-tier vs six-tier, MED fixed) -> ablation_code_tiers.json
python ablation_code_tiers.py

# Compressed-domain payload detectability -> stego_payload_detectability.json
python stego_payload_detectability.py

# Pixel-domain security statistics -> security_results.json
python security_analysis.py
```

Each script prints a summary to stdout and (re)writes its JSON file. The
`ablation_code_tiers.py` run performs a built-in integrity check: if the
six-tier MED result does not reproduce the ground-truth average in
`experiment_results.json`, it aborts rather than reporting a number.

---

## 5. Headline results (from the committed JSON)

- Across twelve 512x512 images, the proposed MED + six-tier code embeds
  **196,902 secret bits (0.7511 bpp)** vs **184,594 bits (0.7042 bpp)** for the
  otherwise-identical secret-driven-selection baseline — a **+6.7%** capacity
  gain while producing a *smaller* label stream, at bit-identical reconstruction.
- Holding MED fixed, refining the four-tier code to the proposed six-tier code
  adds **+3.3%** capacity (190,634 → 196,902 bits).
- Against the published JNC-XOR baseline on the five shared images, capacity
  rises by **13–21%**.
- Compressed-domain steganalysis: raw structured payloads are detected on
  **12/12** images (chi-square ≫ 3.841, H1 < 2), whitened payloads on **0/12**
  (chi-square ≈ 0.6, H1 ≈ 2.000) — the scheme provides no inherent payload
  confidentiality without whitening.

---

## 6. Notes

- Block size is fixed at n = 4 throughout; all results are averaged over the
  twelve images unless a per-image value is noted.
- Reversibility is verified in-code by exact cover reconstruction after
  extraction (bit-identical).

For questions, open an issue in this repository or contact the corresponding
author.
