"""
build_word_paper.py

Generates research_paper/main.docx with identical content to research_paper/main.tex:
- Official IEEE conference typography (Times New Roman)
- Title, Authors, Abstract, Index Terms
- Two-column layout for body text
- Numbered sections, subsections, and subsubsections
- Equations, formatted tables, and inline figure insertions
- Complete bibliography with 15 citations
"""

import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

DOC_DIR = os.path.join(os.path.dirname(__file__), "research_paper")
FIG_DIR = os.path.join(DOC_DIR, "figures")
OUT_PATH = os.path.join(DOC_DIR, "main.docx")


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def add_two_column_section(doc):
    """Inserts a continuous section break with two-column layout."""
    section = doc.add_section()
    sectPr = section._sectPr
    cols = parse_xml(f'<w:cols {nsdecls("w")} w:num="2" w:space="720"/>')
    sectPr.append(cols)
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.75)
    section.left_margin = Inches(0.65)
    section.right_margin = Inches(0.65)
    return section


def style_paragraph(p, space_after=4, space_before=0, line_spacing=1.05, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.line_spacing = line_spacing
    p.alignment = align


def add_heading_1(doc, text):
    p = doc.add_paragraph()
    style_paragraph(p, space_before=12, space_after=4, align=WD_ALIGN_PARAGRAPH.CENTER)
    run = p.add_run(text.upper())
    run.font.name = 'Times New Roman'
    run.font.size = Pt(10)
    run.bold = True
    return p


def add_heading_2(doc, text):
    p = doc.add_paragraph()
    style_paragraph(p, space_before=8, space_after=3, align=WD_ALIGN_PARAGRAPH.LEFT)
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(9.5)
    run.italic = True
    run.bold = True
    return p


def add_body_p(doc, text, bold_prefix=None, italic_prefix=None):
    p = doc.add_paragraph()
    style_paragraph(p)
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = 'Times New Roman'
        r_pre.font.size = Pt(9.5)
        r_pre.bold = True
    if italic_prefix:
        r_pre = p.add_run(italic_prefix)
        r_pre.font.name = 'Times New Roman'
        r_pre.font.size = Pt(9.5)
        r_pre.italic = True
    r = p.add_run(text)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(9.5)
    return p


def add_equation_box(doc, eq_text, eq_num):
    p = doc.add_paragraph()
    style_paragraph(p, space_before=4, space_after=4, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_eq = p.add_run(eq_text)
    r_eq.font.name = 'Times New Roman'
    r_eq.font.size = Pt(9.5)
    r_eq.italic = True
    r_tab = p.add_run(f"    ({eq_num})")
    r_tab.font.name = 'Times New Roman'
    r_tab.font.size = Pt(9.5)
    r_tab.bold = False


def add_caption(doc, text):
    p = doc.add_paragraph()
    style_paragraph(p, space_before=4, space_after=8, align=WD_ALIGN_PARAGRAPH.CENTER)
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(8.5)
    run.bold = True


def build_paper():
    doc = Document()

    # 1. First Section (Title & Abstract, Single Column)
    first_sec = doc.sections[0]
    first_sec.top_margin = Inches(0.75)
    first_sec.bottom_margin = Inches(0.75)
    first_sec.left_margin = Inches(0.75)
    first_sec.right_margin = Inches(0.75)

    # Title
    p_title = doc.add_paragraph()
    style_paragraph(p_title, space_before=0, space_after=6, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_title = p_title.add_run("TRUESIGHT: An Experimental Acoustic Feature-Based Framework for Voice Analysis and Deception Classification")
    r_title.font.name = 'Times New Roman'
    r_title.font.size = Pt(17)
    r_title.bold = True

    # Authors
    p_auth = doc.add_paragraph()
    style_paragraph(p_auth, space_before=4, space_after=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_auth = p_auth.add_run("Diksha Sharma\u00b9, Co-Author Placeholder\u00b2, and Project Supervisor\u00b3\n")
    r_auth.font.name = 'Times New Roman'
    r_auth.font.size = Pt(10)
    r_auth.bold = True

    r_aff = p_auth.add_run("Department of Computer Science and Engineering\nInstitution / University Name Placeholder, City, Country\nEmail: {student, coauthor, supervisor}@institution.edu")
    r_aff.font.name = 'Times New Roman'
    r_aff.font.size = Pt(9)
    r_aff.italic = True

    # Abstract Box
    p_abs = doc.add_paragraph()
    style_paragraph(p_abs, space_before=6, space_after=6, line_spacing=1.05)
    r_abs_h = p_abs.add_run("Abstract\u2014")
    r_abs_h.font.name = 'Times New Roman'
    r_abs_h.font.size = Pt(9)
    r_abs_h.bold = True

    abs_text = (
        "Automated voice-based veracity and stress analysis represents a challenging frontier at the intersection of digital signal processing, "
        "computational phonetics, and machine learning. In high-stakes human communication, acoustic speech characteristics frequently exhibit "
        "micro-prosodic variations in response to cognitive load, autonomic arousal, or vocal tension. This paper presents TRUESIGHT, an experimental "
        "acoustic feature-based framework designed for automated voice analysis, prosodic inspection, and binary deception classification. TRUESIGHT "
        "integrates a resilient audio ingestion and preprocessing pipeline with dual feature extraction engines: Praat-Parselmouth for fundamental "
        "frequency (F0), micro-perturbations (local jitter and shimmer), and temporal pause/silence structures, coupled with Librosa for Mel-Frequency "
        "Cepstral Coefficients (MFCCs), spectral centroid, bandwidth, rolloff, and energy envelope dynamics, yielding a canonical 42-dimensional "
        "acoustic feature vector. Classification is driven by a 200-tree Random Forest ensemble paired with a deterministic, rule-based explanation "
        "engine that articulates phonetic observations without asserting absolute veracity verdicts. We conduct rigorous, empirical evaluations "
        "across distinct acoustic domains: (1) 5-fold speaker-independent GroupKFold cross-validation on the Dolos trial corpus (N=168), achieving "
        "64.7% \u00b1 8.8% accuracy (0.640 Macro F1) versus 62.5% \u00b1 7.1% for a standardized linear baseline; (2) strictly held-out external "
        "evaluation on the CBU5201 benchmark (N=100), where TRUESIGHT achieves 56.0% accuracy (0.560 Macro F1, 0.613 ROC-AUC), outperforming the baseline "
        "linear model (48.0% accuracy, 0.403 Macro F1) and lifting deceptive sensitivity from 12.0% to 54.0% (+42.0 pp gain); and (3) an acoustic audit "
        "across a 50-sample multi-speaker Indian-accented English cohort. Our findings demonstrate that while acoustic models capture genuine prosodic "
        "stress dynamics, veracity classification suffers cross-lingual degradation (-12.0 pp drop on non-English speech) and lacks statistical "
        "significance (p = 0.1698, McNemar test). We conclude with critical ethical safeguards, emphasizing that vocal acoustic indicators reflect "
        "physiological arousal rather than verified real-world deception."
    )
    r_abs = p_abs.add_run(abs_text)
    r_abs.font.name = 'Times New Roman'
    r_abs.font.size = Pt(9)

    # Keywords
    p_kw = doc.add_paragraph()
    style_paragraph(p_kw, space_before=2, space_after=14)
    r_kw_h = p_kw.add_run("Index Terms\u2014")
    r_kw_h.font.name = 'Times New Roman'
    r_kw_h.font.size = Pt(9)
    r_kw_h.bold = True
    r_kw = p_kw.add_run("Acoustic feature extraction, voice stress analysis, speech prosody, deception classification, machine learning, Random Forest, FastAPI, signal processing.")
    r_kw.font.name = 'Times New Roman'
    r_kw.font.size = Pt(9)

    # 2. Add Two-Column Section for Body
    add_two_column_section(doc)

    # I. INTRODUCTION
    add_heading_1(doc, "I. Introduction")
    add_body_p(doc, "Speech is the primary and most expressive modality of human interpersonal communication. Beyond semantic verbal content, vocal acoustics convey rich non-verbal information regarding speaker emotional state, cognitive effort, and physiological arousal [1]. Over the past two decades, forensic psychologists, criminologists, and computational linguists have investigated whether systematic acoustic anomalies\u2014such as elevated fundamental frequency (F0), micro-tremors in vocal fold oscillation (jitter and shimmer), altered speaking rate, and unnatural pause distributions\u2014can serve as empirical markers of vocal tension or deceptive intent [2], [3].")
    add_body_p(doc, "However, computational deception detection from vocal signals remains fraught with theoretical and empirical complexities. Conventional polygraph-style Voice Stress Analysis (VSA) systems historically claimed high veracity discrimination based on proprietary sub-audible micro-tremor filters; subsequent controlled psychophysiological evaluations revealed that such systems frequently performed no better than chance [3]. Modern machine-learning approaches have transitioned toward high-dimensional acoustic-prosodic feature extraction, capturing spectral envelope distributions and voice quality parameters [4], [5]. Despite algorithmic advancements, speech-based veracity prediction suffers from severe confounding factors, including inter-speaker anatomical differences, dialectal and accent variations, emotional state, ambient acoustic noise, and recording hardware heterogeneity [6].")
    add_body_p(doc, "To systematically investigate these phenomena within an inspectable, open-source software environment, we introduce TRUESIGHT, an experimental acoustic feature-based framework and analysis engine. TRUESIGHT does not purport to be a conclusive or forensic 'lie detector'; rather, it is designed as an investigative research prototype that extracts standardized acoustic biomarkers, performs ensemble classification, and couples raw model predictions with structured, rule-based explanations that contextualize vocal variability against normal human conversational baselines.")
    add_body_p(doc, "The primary technical contributions of this research are threefold: (1) A robust, multi-modal audio ingestion and preprocessing pipeline enforcing 16 kHz mono sinc-resampling, short-time RMS Voice Activity Detection (VAD), and SNR audits; (2) A unified dual-engine 42-dimensional acoustic extraction pipeline combining Praat-Parselmouth (pitch, perturbation, silence TextGrids) with Librosa (MFCCs, spectral moments); and (3) A leakage-free empirical evaluation audit evaluating TRUESIGHT against a standardized linear baseline across speaker-independent GroupKFold cross-validation, external unseen test data (CBU5201), and multi-speaker Indian-English speech cohorts.")

    # II. RELATED WORK
    add_heading_1(doc, "II. Related Work")
    add_heading_2(doc, "A. Acoustic Prosody and Voice Quality Analysis")
    add_body_p(doc, "Acoustic speech analysis relies on extracting mathematical representations that reflect vocal fold physiology and vocal tract filter characteristics. Fundamental frequency (F0), representing the rate of vocal cord vibration, is conventionally extracted via autocorrelation or normalized cross-correlation algorithms [7]. Perturbation measures, specifically jitter (cycle-to-cycle frequency variation) and shimmer (cycle-to-cycle amplitude variation), serve as classical indicators of pathological dysphonia or physiological vocal instability [8]. Spectral envelope modeling is predominantly accomplished via Mel-Frequency Cepstral Coefficients (MFCCs), introduced by Davis and Mermelstein [9], which approximate human auditory perception through non-linearly spaced triangular filter banks. Toolkits such as Praat [7], Parselmouth [8], and Librosa [10] have standardized the extraction of these prosodic and spectral parameters.")
    add_heading_2(doc, "B. Computational Voice-Based Deception Detection")
    add_body_p(doc, "Pioneering psychological meta-analyses by DePaulo et al. [2] established that vocal pitch tends to increase slightly during deception, accompanied by modest increases in speech hesitation and response latency. However, effect sizes across verbal and non-verbal cues are notoriously small and context-dependent. Hirschberg et al. [1] constructed the Columbia Deception Corpus, demonstrating that machine-learning classifiers trained on acoustic-prosodic features achieved classification accuracies between 58% and 64%, substantially below naive human expectations. P\u00e9rez-Rosas et al. [4] introduced the Real-Life Deception dataset collected from public courtroom trials, reporting multimodal accuracies exceeding 75% when combining facial action units, lexical transcripts, and acoustic descriptors. Subsequent investigations by Mendels et al. [5] demonstrated that acoustic features alone rarely exceed 60%\u201365% generalization accuracy on out-of-domain forensic data due to speaker memorization and acoustic environmental confounding.")
    add_heading_2(doc, "C. Dataset Bias, Speaker Leakage, and Cross-Cultural Variance")
    add_body_p(doc, "A prominent vulnerability in voice classification literature is data leakage induced by naive random train-test splitting. When audio segments from the same individual appear in both training and test partitions, classifiers tend to memorize speaker identity (vocal tract length, baseline pitch, accent) rather than task-generalizable deception cues. Levitan et al. [6] examined cross-cultural deception in intercultural dialogues, finding that acoustic manifestations of deceit vary drastically across native languages and cultural communication norms. Furthermore, regional speech varieties, such as Indian-accented English, possess distinct syllable-timed rhythmic structures, pitch excursion patterns, and phonotactic characteristics that conventional Western-calibrated models risk misinterpreting as vocal anomalies [6], [11].")

    # III. PROBLEM STATEMENT
    add_heading_1(doc, "III. Problem Statement and Objectives")
    add_body_p(doc, "The core research problem addressed by TRUESIGHT is the systematic evaluation of whether acoustic and prosodic speech features can provide reliable, generalizable indicators of deception in human speech without succumbing to speaker identity memorization, signal corruption, or cultural bias. Formally, given a digitized audio signal s(t), the framework aims to: (1) Determine voice presence via energy-based VAD; (2) Map s(t) into a canonical 42-dimensional feature vector v; (3) Compute the posterior class probability P(Deceptive | v) using an inspectable ensemble; and (4) Generate structured phonetic explanations that explicitly decouple vocal tension indicators from moral veracity assertions.")

    # IV. PROPOSED SYSTEM ARCHITECTURE
    add_heading_1(doc, "IV. Proposed System Architecture")
    add_body_p(doc, "The architecture of TRUESIGHT comprises five tightly decoupled modules: Audio Ingestion Layer (FastAPI REST and HTML5 WebSockets), Signal Preprocessing Engine (16 kHz resampling, mono downmixing, RMS VAD), Dual-Engine 42-D Extraction Pipeline (Praat + Librosa), Machine Learning Inference Engine (200-tree Random Forest), and Rule-Based Explanation Engine.")

    # Figure 1 Insertion
    fig1_path = os.path.join(FIG_DIR, "fig1_architecture.png")
    if os.path.exists(fig1_path):
        doc.add_picture(fig1_path, width=Inches(3.4))
        add_caption(doc, "Fig. 1. End-to-end architectural schematic of the TRUESIGHT voice analysis pipeline.")

    # V. METHODOLOGY
    add_heading_1(doc, "V. Methodology")
    add_heading_2(doc, "A. Audio Acquisition and Client Ingestion")
    add_body_p(doc, "TRUESIGHT supports two acquisition modes: (1) In-memory multipart REST file uploads (.wav, .mp3, .m4a, .webm, .flac) up to 25 MB processed in volatile RAM buffers to safeguard user privacy; and (2) Browser-based live microphone streaming using HTML5 MediaRecorder API transmitting chunked PCM slices (default: 15.0 s rolling window) to server endpoints.")

    add_heading_2(doc, "B. Signal Preprocessing and Voice Activity Detection")
    add_body_p(doc, "Incoming waveforms s[n] undergo four mandatory transformations: stereo-to-mono downmixing, polyphase sinc-resampling to 16,000 Hz, short-time RMS energy gating, and minimum duration validation. Voice Activity Detection (VAD) evaluates short-term RMS energy:")
    add_equation_box(doc, "E_RMS = sqrt( (1 / N) * sum_{n=1}^N s[n]^2 )", "1")
    add_body_p(doc, "Segments with peak amplitude < 0.01 or E_RMS below -45 dB relative to full scale are flagged as 'VOICE NOT DETECTED' and rejected prior to model inference. Audio clips shorter than 0.50 s are rejected due to insufficient glottal periods for perturbation analysis.")

    add_heading_2(doc, "C. 42-Dimensional Acoustic Feature Extraction")
    add_body_p(doc, "The extraction pipeline computes exactly 42 acoustic descriptors across six functional categories:")
    add_body_p(doc, "1) Fundamental Frequency (F0) Prosody (4 features): Mean, standard deviation, minimum, and maximum pitch extracted via Praat cross-correlation (75\u2013500 Hz), requiring at least 15 voiced frames (~150 ms phonation).")
    add_body_p(doc, "2) Glottal Micro-Perturbations (2 features): Local jitter (cycle-to-cycle frequency variation) and local shimmer (cycle-to-cycle amplitude variation) computed over PointProcess glottal cycles (minimum 25 cycles required):")
    add_equation_box(doc, "J_loc = [ (1 / (K-1)) * sum |T_i - T_{i+1}| ] / [ (1 / K) * sum T_i ]", "2")
    add_equation_box(doc, "S_loc = [ (1 / (K-1)) * sum |A_i - A_{i+1}| ] / [ (1 / K) * sum A_i ]", "3")
    add_body_p(doc, "3) Temporal and Pause Dynamics (4 features): Syllable onset speech tempo, pause ratio (silence fraction), discrete pause count, and cumulative silence duration using Praat's TextGrid silence detector (-25 dB threshold, min silence 150 ms).")
    add_body_p(doc, "4) Energy Dynamics and Zero-Crossing (3 features): RMS energy mean, RMS energy standard deviation, and Zero-Crossing Rate (ZCR):")
    add_equation_box(doc, "ZCR = (1 / (2(N-1))) * sum_{n=1}^{N-1} | sgn(s[n]) - sgn(s[n-1]) |", "4")
    add_body_p(doc, "5) Spectral Envelope Moments (3 features): Spectral centroid (center of spectral mass), spectral bandwidth, and spectral rolloff (85% energy frequency):")
    add_equation_box(doc, "f_c = [ sum_{k=0}^{M-1} f[k] * |S[k]| ] / [ sum_{k=0}^{M-1} |S[k]| ]", "5")
    add_body_p(doc, "6) Mel-Frequency Cepstral Coefficients (26 features): 13 MFCC temporal means and 13 MFCC temporal standard deviations across 40 triangular Mel filterbanks.")

    # Figure 2 Insertion
    fig2_path = os.path.join(FIG_DIR, "fig2_feature_pipeline.png")
    if os.path.exists(fig2_path):
        doc.add_picture(fig2_path, width=Inches(3.4))
        add_caption(doc, "Fig. 2. Dual-engine 42-dimensional acoustic extraction and VAD gating flowchart.")

    add_heading_2(doc, "D. Machine Learning Model")
    add_body_p(doc, "Inference is driven by an ensemble Random Forest classifier [12] with B = 200 trees trained on the Dolos dataset using Gini impurity splits. Class posterior probability P(Deceptive | v) is obtained via ensemble averaging, with decision threshold fixed at tau = 0.50 without post-hoc tuning on test sets.")

    add_heading_2(doc, "E. Explanation Module")
    add_body_p(doc, "The rule-based explanation module (app/explanation.py) inspects feature values against conservative clinical vocal thresholds (pitch std >= 35 Hz, pause ratio >= 0.30, jitter >= 0.02, shimmer >= 0.08). The module explicitly caveats each observation, separating acoustic vocal tension from deceit.")

    add_heading_2(doc, "F. Software Implementation")
    add_body_p(doc, "TRUESIGHT is implemented in Python 3.11/3.13, utilizing FastAPI for asynchronous HTTP/WebSocket routing, Praat-Parselmouth 0.4.7 for phonetics, Librosa 0.10.2 for spectral DSP, Scikit-Learn 1.9.0 for ML modeling, and vanilla HTML5/ES6 for the frontend.")

    # VI. DATASET DESCRIPTION
    add_heading_1(doc, "VI. Dataset Description and Experimental Setup")
    add_body_p(doc, "TRUESIGHT was evaluated across three distinct speech corpora: (1) Dolos Trial Corpus: 168 audio clips (84 truthful, 84 deceptive) across 122 unique courtroom speakers [4]; (2) CBU5201 External Benchmark: 100 audio clips (50 truthful, 50 deceptive; 50 English, 50 Chinese) recorded under laboratory storytelling conditions [13], reserved strictly as an unseen test benchmark; and (3) Indian-Accent Multi-Speaker Dataset: 50 conversational query clips across 46 speakers [11], used for acoustic extraction stability analysis with zero veracity labels.")

    # VII. EXPERIMENTAL DESIGN & METRICS
    add_heading_1(doc, "VII. Experimental Design and Evaluation Metrics")
    add_body_p(doc, "Performance is quantified via standard binary metrics: Accuracy, Balanced Accuracy, Macro Precision, Macro Recall, Macro F1-score, and ROC-AUC. 95% confidence intervals are computed using the Wilson score interval [14] for accuracy and 1,000 bootstrap resamples for non-linear metrics. Paired model significance on identical test samples is evaluated using McNemar's test with Edwards continuity correction [15]:")
    add_equation_box(doc, "chi^2 = ( |b - c| - 1 )^2 / (b + c)", "6")

    # VIII. RESULTS AND DISCUSSION
    add_heading_1(doc, "VIII. Results and Discussion")
    add_heading_2(doc, "A. External Benchmark Generalization on CBU5201")
    add_body_p(doc, "Table I presents the performance comparison between the Baseline Linear model and the Production Random Forest model evaluated on the unseen CBU5201 dataset (N=100).")

    # Insert Table I
    t = doc.add_table(rows=11, cols=5)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Metric", "Baseline", "TRUESIGHT", "Change", "95% CI"]
    for j, h in enumerate(headers):
        cell = t.cell(0, j)
        cell.text = h
        cell.paragraphs[0].runs[0].font.name = 'Times New Roman'
        cell.paragraphs[0].runs[0].font.size = Pt(8.5)
        cell.paragraphs[0].runs[0].bold = True
        set_cell_margins(cell, 80, 80, 100, 100)

    rows_data = [
        ("Accuracy", "48.0%", "56.0%", "+8.0 pp", "[46.2%, 65.3%]"),
        ("Balanced Acc", "48.0%", "56.0%", "+8.0 pp", "[46.9%, 65.6%]"),
        ("Macro Precision", "0.458", "0.560", "+0.102", "N/A"),
        ("Macro Recall", "0.480", "0.560", "+0.080", "N/A"),
        ("Macro F1", "0.403", "0.560", "+0.157", "[0.467, 0.650]"),
        ("ROC-AUC", "0.492", "0.613", "+0.121", "[0.505, 0.712]"),
        ("Truthful Prec", "0.488", "0.558", "+0.070", "\u2014"),
        ("Truthful Recall", "84.0%", "58.0%", "-26.0 pp", "\u2014"),
        ("Deceptive Prec", "0.429", "0.562", "+0.133", "\u2014"),
        ("Deceptive Recall", "12.0%", "54.0%", "+42.0 pp", "\u2014"),
    ]
    for i, rdata in enumerate(rows_data):
        for j, val in enumerate(rdata):
            cell = t.cell(i + 1, j)
            cell.text = val
            cell.paragraphs[0].runs[0].font.name = 'Times New Roman'
            cell.paragraphs[0].runs[0].font.size = Pt(8)
            set_cell_margins(cell, 60, 60, 80, 80)

    add_caption(doc, "TABLE I. External Evaluation on CBU5201 Benchmark (N=100)")

    add_body_p(doc, "As evidenced by the confusion matrices (Figs. 3 and 4), the Baseline Linear model suffers from severe mode collapse, predicting truthful for 86% of samples and detecting only 6 of 50 deceptive clips (12.0% recall). In contrast, the Random Forest model achieves balanced sensitivity, detecting 27 deceptive clips (54.0% recall) and 29 truthful clips (58.0% recall), lifting accuracy from 48.0% to 56.0% and Macro F1 from 0.403 to 0.560.")

    # Insert Fig 3 and Fig 4
    fig3_path = os.path.join(FIG_DIR, "fig3_confusion_baseline.png")
    fig4_path = os.path.join(FIG_DIR, "fig4_confusion_production.png")
    if os.path.exists(fig3_path):
        doc.add_picture(fig3_path, width=Inches(3.4))
        add_caption(doc, "Fig. 3. Baseline Linear Model Confusion Matrix on CBU5201 (N=100).")
    if os.path.exists(fig4_path):
        doc.add_picture(fig4_path, width=Inches(3.4))
        add_caption(doc, "Fig. 4. Production Random Forest Confusion Matrix on CBU5201 (N=100).")

    add_heading_2(doc, "B. Cross-Lingual Performance Gap")
    add_body_p(doc, "Stratification by language reveals a pronounced cross-lingual domain drop (Fig. 5): on the English partition (N=50), TRUESIGHT achieves 62.0% accuracy (0.619 Macro F1, 0.630 ROC-AUC), an 18.0 pp gain over baseline (44.0%). On the Chinese partition (N=50), accuracy drops to 50.0% (chance level). Mandarin Chinese is a tonal language where fundamental frequency encodes lexical meaning; acoustic models trained on Western non-tonal speech misinterpret natural tonal pitch excursions as stress anomalies.")

    # Insert Fig 5
    fig5_path = os.path.join(FIG_DIR, "fig5_model_comparison.png")
    if os.path.exists(fig5_path):
        doc.add_picture(fig5_path, width=Inches(3.4))
        add_caption(doc, "Fig. 5. Comparative performance profile on external benchmark and language partitions.")

    add_heading_2(doc, "C. Speaker-Independent Cross-Validation on Dolos")
    add_body_p(doc, "Under 5-fold GroupKFold cross-validation on the Dolos corpus (N=168, 122 speakers), TRUESIGHT achieves 64.7% \u00b1 8.8% mean accuracy (0.640 Macro F1) across folds, outperforming the baseline linear model (62.5% \u00b1 7.1%, 0.618 Macro F1), confirming robust multi-tree generalization under strict speaker separation.")

    add_heading_2(doc, "D. Resolution of Historical Metric Inconsistencies")
    add_body_p(doc, "Our audit resolved an earlier discrepancy in project drafts where some sections cited 48% vs 56% accuracy while others cited 55% vs 60%. The 48% vs 56% results represent the rigorous, reproducible evaluation executed across the full 100-sample CBU5201 manifest using the unified FeatureExtractor pipeline at default threshold tau = 0.50. The historical 55% vs 60% figures originated from an earlier ad-hoc prototype that applied post-hoc probability threshold tuning and alternative silence trimming. We retain the verified 48.0% vs 56.0% figures as the authentic scientific benchmark.")

    add_heading_2(doc, "E. Statistical Significance")
    add_body_p(doc, "Evaluating paired predictions on the 100 CBU5201 samples yields b = 17 samples (RF correct, Linear incorrect) and c = 9 samples (Linear correct, RF incorrect). Applying McNemar's test with continuity correction yields chi^2 = 1.8846 with p = 0.1698. Because p > 0.05, the observed numerical gain (+8.0 pp) cannot be claimed as statistically significant at the 95% confidence level due to test sample size constraints (N=100). The result represents a strong empirical trend rather than mathematical certainty.")

    add_heading_2(doc, "F. Indian-English Speech Stability Audit")
    add_body_p(doc, "Evaluation across 50 Indian-English recordings confirmed 100% pipeline extraction success with zero NaNs. Mean uncalibrated score was 0.478 \u00b1 0.086 (range: 0.310\u20130.630). Because no public Indian-accented speech dataset contains verified veracity ground truth, TRUESIGHT strictly reports deception classification on Indian English as UNVALIDATED.")

    # IX. LIMITATIONS AND ETHICAL CONSIDERATIONS
    add_heading_1(doc, "IX. Limitations and Ethical Considerations")
    add_body_p(doc, "1) Acoustic Multi-Causality: Physiological vocal tension reflects anxiety, unfamiliarity with recording settings, cognitive effort, or fatigue\u2014not deceit alone. Attributing deception to an acoustic anomaly is scientifically invalid.")
    add_body_p(doc, "2) Risk of False Accusations: False positive classifications (FP=21 in our external test) pose grave ethical risks. TRUESIGHT enforces neutral framing ('Experimental Model Output') and prohibits definitive truth/lie assertions.")
    add_body_p(doc, "3) Microphone Hardware Diversity: Uncalibrated consumer microphones distort spectral rolloff and shimmer measurements independently of speaker physiology.")
    add_body_p(doc, "4) Privacy Safeguards: Audio waveforms are processed in volatile RAM and immediately discarded; no voiceprints or raw recordings are retained.")

    # X. CONCLUSION
    add_heading_1(doc, "X. Conclusion and Future Work")
    add_body_p(doc, "This paper presented TRUESIGHT, an experimental acoustic feature-based framework for speech analysis and deception classification. By integrating a 42-dimensional feature extraction engine combining Praat-Parselmouth and Librosa with a 200-tree Random Forest classifier, TRUESIGHT demonstrates measurable empirical improvements over linear baselines on external unseen data (56.0% vs. 48.0% accuracy), substantially mitigating minority-class mode collapse. However, rigorous statistical testing (p = 0.1698) and cross-lingual degradation (-12.0 pp drop on tonal speech) underline that acoustic markers alone cannot provide definitive lie detection. Future work will explore multimodal integration with ASR transcripts, self-supervised Wav2Vec 2.0 representations, and cross-cultural dataset curation with verified ground truth.")

    # ACKNOWLEDGMENT
    add_heading_1(doc, "Acknowledgment")
    add_body_p(doc, "The authors gratefully acknowledge the creators of the open-source Dolos and CBU5201 corpora, as well as the developers of Praat, Parselmouth, Librosa, and scikit-learn, whose tools enabled this empirical audit.")

    # REFERENCES
    add_heading_1(doc, "References")
    references = [
        "[1] J. Hirschberg, S. Benus, J. M. Brenier, F. Enos, S. Friedman, S. Levitan, D. I. Moldovan, and A. Pelikan, 'Distinguishing deceptive from non-deceptive speech,' in Proc. Interspeech 2005, Lisbon, Portugal, 2005, pp. 1833\u20131836.",
        "[2] B. M. DePaulo, J. J. Lindsay, B. E. Malone, L. Muhlenbruck, K. Charlton, and H. Cooper, 'Cues to deception,' Psychological Bulletin, vol. 129, no. 1, pp. 74\u2013118, 2003.",
        "[3] A. Vrij, R. P. Fisher, and H. Blank, 'A cognitive approach to lie detection: A meta-analysis,' Legal and Criminological Psychology, vol. 22, no. 1, pp. 1\u201321, 2017.",
        "[4] V. P\u00e9rez-Rosas, M. Abouelenien, R. Mihalcea, Y. Xiao, C. J. Linting, and M. Burzo, 'Deception detection using real-life trial data,' in Proc. 2015 ACM ICMI, 2015, pp. 59\u201366.",
        "[5] G. Mendels, M. Chalamandaris, and J. Hirschberg, 'Hybrid acoustic-lexical deception detection,' in Proc. Interspeech 2017, Stockholm, Sweden, 2017, pp. 2001\u20132005.",
        "[6] S. I. Levitan, A. Maredia, and J. Hirschberg, 'Acoustic-prosodic and lexical cues to deception and trust in cross-cultural dialogue,' in Proc. EMNLP 2018, Brussels, Belgium, 2018, pp. 1566\u20131576.",
        "[7] P. Boersma and D. Weenink, 'Praat: doing phonetics by computer,' Glot International, vol. 5, no. 9/10, pp. 341\u2013345, 2001.",
        "[8] Y. Jadoul, B. Thompson, and B. de Boer, 'Introducing Parselmouth: A Python interface to Praat,' Journal of Phonetics, vol. 71, pp. 1\u201315, 2018.",
        "[9] S. B. Davis and P. Mermelstein, 'Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences,' IEEE Trans. ASSP, vol. 28, no. 4, pp. 357\u2013366, 1980.",
        "[10] B. McFee, C. Raffel, D. Liang, D. P. W. Ellis, M. McVicar, E. Battenberg, and O. Nieto, 'librosa: Audio and music signal analysis in Python,' in Proc. SciPy 2015, 2015, pp. 18\u201325.",
        "[11] S. Kumar, 'Indian Accent Dataset: A multi-speaker speech corpus,' Hugging Face Datasets, 2023. [Online]. Available: https://huggingface.co/datasets/Santhosh-kumar/Indian-Accent-Dataset",
        "[12] L. Breiman, 'Random forests,' Machine Learning, vol. 45, no. 1, pp. 5\u201332, 2001.",
        "[13] T. Tiendrebeogo et al., 'CBU5201 Deception Dataset: An annotated audio-visual corpus of true and deceptive anecdotes,' 2022. [Online]. Available: https://github.com/CBU5201Datasets/Deception",
        "[14] E. B. Wilson, 'Probable inference, the law of succession, and statistical inference,' J. Amer. Statist. Assoc., vol. 22, no. 158, pp. 209\u2013212, 1927.",
        "[15] Q. McNemar, 'Note on the sampling error of the difference between correlated proportions or percentages,' Psychometrika, vol. 12, no. 2, pp. 153\u2013157, 1947."
    ]

    for ref in references:
        p_ref = doc.add_paragraph()
        style_paragraph(p_ref, space_before=2, space_after=2, align=WD_ALIGN_PARAGRAPH.LEFT)
        r_ref = p_ref.add_run(ref)
        r_ref.font.name = 'Times New Roman'
        r_ref.font.size = Pt(8.5)

    doc.save(OUT_PATH)
    print(f"Successfully generated complete Word paper at {OUT_PATH}")


if __name__ == "__main__":
    build_paper()
