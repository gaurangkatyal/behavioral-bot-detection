# Account-history features for social bot detection in the era of large language models

Research paper bundle for submission to the Journal of Online Trust and Safety (primary) or Cybersecurity / EPJ Data Science (secondary).

## Files

- `manuscript.docx` — Final manuscript, ready for journal submission (Times New Roman 12pt, ~6,500 words, 9 tables, 5 figures, 20 pages)
- `manuscript_preview.pdf` — Rendered preview for QA
- `code/` — All analysis and document-generation scripts
- `results/` — All 10 result CSVs plus summary JSONs
- `figures/` — All 5 figures as standalone PNGs

## Headline results

| Model | ROC-AUC (5-fold CV) | Brier | FPR | FNR |
|---|---|---|---|---|
| RF-Content | 0.830 (0.017) | 0.155 | 0.130 | 0.322 |
| RF-Behavioral | 0.977 (0.003) | 0.055 | 0.034 | 0.118 |
| RF-Fusion | **0.981 (0.003)** | 0.050 | 0.022 | 0.102 |

**DeLong's test (paired AUC):**
- Behavioral vs Content: z = 9.36, p < 0.001
- Fusion vs Behavioral: z = 2.67, p = 0.008 (Δ AUC = 0.003)

**Adversarial robustness (text-rewriting, severity 1.0):**
- Content: AUC 0.842 → 0.785
- Behavioral: 0.981 → 0.981 (invariant)
- Fusion: 0.984 → 0.984 (essentially invariant)

**Adversarial robustness (feature-space, upper bound):**
- Content: AUC 0.842 → 0.466 (below chance)
- Behavioral: 0.981 → 0.981 (invariant)

## Reproducing the analysis

```bash
cd code/
python3 run_analysis.py     # Tables 1-4, Figures 1-4, TwiBot-20 scoring
python3 run_extended.py     # Tables 5-10, Figure 5 (calibration)
python3 regen_figures.py    # Regenerate figures with paper-clean titles
node build_paper.js         # Generate manuscript.docx
```

Dependencies: Python 3 with pandas, scikit-learn, scipy, matplotlib, seaborn; Node.js with the `docx` package.

## Data sources

- **Primary (labeled):** `jubins/MachineLearning-Detecting-Twitter-Bots` on GitHub. 2,432 unique accounts, 43% bots, manual annotation. URL: https://raw.githubusercontent.com/jubins/MachineLearning-Detecting-Twitter-Bots/master/FinalProjectAndCode/kaggle_data/training_data_2_csv_UTF.csv
- **Secondary (unlabeled, cross-dataset):** `BunsenFeng/TwiBot-20` public sample on GitHub. 100 accounts. URL: https://raw.githubusercontent.com/BunsenFeng/TwiBot-20/main/TwiBot-20_sample.json

## Submission notes

Primary target: **Journal of Online Trust and Safety** (https://tsjournal.org). Open access, no APC, fast review cycles, indexed in Google Scholar, growing influence in the trust-and-safety research community. The paper's framing aligns with JOTS's stated scope.

Backup targets:
1. **Cybersecurity** (Springer Nature, indexed in SCIE, Q2)
2. **EPJ Data Science** (Springer Nature, indexed in SCIE, Q1)
3. **First Monday** (open access, accepted Ferrara's 2023 piece on the same topic)

## Author actions before submission

1. Fill in `[Author Name]`, `[Affiliation]`, and `[email]` on page 1
2. Replace `[Acknowledgments to be added upon de-anonymization.]` with real acknowledgments at camera-ready stage
3. Verify ORCID is registered
4. Deposit code in a public repository (GitHub + Zenodo for a DOI)
5. Pre-print to arXiv (cs.SI or cs.CR) on the day of submission to maximize citation accumulation in the review window
