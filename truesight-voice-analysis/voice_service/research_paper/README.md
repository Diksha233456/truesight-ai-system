# TRUESIGHT: Research Paper Package (IEEE Format)

This directory contains the complete, publication-ready research paper for the **TRUESIGHT** voice analysis and deception classification project.

---

## 1. Directory Structure

```text
research_paper/
├── main.tex                    # Complete IEEEtran LaTeX source code
├── references.bib              # Verified BibTeX bibliography (15 citations)
├── main.docx                   # Complete editable Microsoft Word document
├── README.md                   # Compilation, reproduction, and submission guide
└── figures/
    ├── fig1_architecture.png           # System architecture schematic
    ├── fig2_feature_pipeline.png       # 42-D feature extraction & VAD workflow
    ├── fig3_confusion_baseline.png     # Baseline linear model confusion matrix
    ├── fig4_confusion_production.png   # Production Random Forest confusion matrix
    └── fig5_model_comparison.png       # Evaluation comparison & language breakdown
```

---

## 2. Paper Metadata

- **Working Title:** *TRUESIGHT: An Experimental Acoustic Feature-Based Framework for Voice Analysis and Deception Classification*
- **Layout:** Standard IEEE two-column conference format (`\documentclass[conference]{IEEEtran}`)
- **Target Venue:** IEEE Conferences in Signal Processing, Audio Engineering, Computational Linguistics, or Applied AI
- **Primary Model Evaluated:** 200-Tree Random Forest Classifier (`app/models/deception_model.pkl`) using 42 acoustic-prosodic features
- **Baseline Model:** Standardized Linear Classifier (`StandardScaler` + `LogisticRegression(max_iter=1000)`)

---

## 3. How to Compile the LaTeX Paper

### Option A: Overleaf (Recommended)
1. Go to [Overleaf](https://www.overleaf.com) and create a **New Project** $\rightarrow$ **Blank Project**.
2. Upload:
   - `main.tex`
   - `references.bib`
   - The entire `figures/` folder (with `fig1_architecture.png`, `fig2_feature_pipeline.png`, etc.)
3. Ensure the compiler is set to **pdfLaTeX** or **XeLaTeX** (under *Menu* $\rightarrow$ *Compiler*).
4. Click **Recompile**. Overleaf will generate the final two-column PDF.

### Option B: Local Windows (MiKTeX or TeX Live)
If you have MiKTeX or TeX Live installed:
```powershell
# In PowerShell:
cd research_paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
Or using `latexmk`:
```powershell
latexmk -pdf main.tex
```

### Option C: Linux / macOS
```bash
cd research_paper
latexmk -pdf -interaction=nonstopmode main.tex
```

---

## 4. How to Edit and Update the Word Document (`main.docx`)

- **Direct Editing:** Open `main.docx` in Microsoft Word 2016+, Office 365, Google Docs, or LibreOffice Writer. The document features:
  - Standard IEEE margins (0.75 in top/bottom, 0.65 in left/right)
  - Times New Roman typography (10 pt body, 9.5 pt subsections, 9 pt abstract)
  - Two-column section layout for the body text
  - Formatted mathematical equations and Table I
  - Embedded publication-quality figures with captions
  - Complete numbered IEEE bibliography ([1] through [15])
- **Regenerating from Code:** If you modify `build_word_paper.py`, you can regenerate `main.docx` at any time:
  ```powershell
  .\.venv\Scripts\python.exe build_word_paper.py
  ```

---

## 5. Verified Evaluation Results Summary

| Evaluation Domain | Metric | Baseline Linear | TRUESIGHT (Production RF) | Change | 95% Confidence Interval |
|---|---|---:|---:|---:|---:|
| **CBU5201 External Test ($N=100$)** | Accuracy | 48.0% | **56.0%** | **+8.0 pp** | [46.2%, 65.3%] (Wilson) |
| | Balanced Accuracy | 48.0% | **56.0%** | **+8.0 pp** | [46.9%, 65.6%] (Bootstrap) |
| | Macro F1-Score | 0.403 | **0.560** | **+0.157** | [0.467, 0.650] (Bootstrap) |
| | ROC-AUC | 0.492 | **0.613** | **+0.121** | [0.505, 0.712] (Bootstrap) |
| | Deceptive Recall | 12.0% | **54.0%** | **+42.0 pp** | — |
| **CBU5201 English Subset ($N=50$)** | Accuracy | 44.0% | **62.0%** | **+18.0 pp** | [48.1%, 74.1%] |
| **CBU5201 Chinese Subset ($N=50$)** | Accuracy | 52.0% | **50.0%** | **-2.0 pp** | [36.6%, 63.4%] |
| **Dolos 5-Fold GroupKFold ($N=168$)** | Mean Accuracy | 62.5% | **64.7%** | **+2.2 pp** | $\pm 8.8\%$ (across folds) |
| | Mean Macro F1 | 0.618 | **0.640** | **+0.022** | — |
| **Indian-English Cohort ($N=50$)** | Extraction Success | 100% | 100% | 0 NaNs | Veracity **UNVALIDATED** |

---

## 6. Placeholders to Fill Before Submission

In `main.tex` and `main.docx`, update the following author and institution fields:
1. `Author 1`: Primary Student Author Name
2. `Author 2`: Co-Author Name (or remove if single author)
3. `Author 3`: Project Supervisor / Faculty Mentor Name
4. `Department`: e.g., *Department of Computer Science and Engineering*
5. `Institution`: Your University / College Name
6. `Emails`: Institutional email addresses
7. `Acknowledgment`: Add grant numbers or faculty acknowledgments if applicable.
