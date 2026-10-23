import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#718096"))
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 40, 30, page_text)
        self.drawString(40, 30, "TRUSIGHT Voice Service — Implementation & Architecture Guide")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(40, 42, 612 - 40, 42)
        self.restoreState()

def build_pdf(filename):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=45,
        bottomMargin=55
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#2563EB'),
        spaceAfter=14
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=12,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#2563EB'),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#334155'),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13.5,
        textColor=colors.HexColor('#334155'),
        leftIndent=15,
        spaceAfter=4
    )

    callout_style = ParagraphStyle(
        'Callout_Text',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#1E3A8A'),
    )

    table_header_style = ParagraphStyle(
        'TH',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        'TC',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#1E293B')
    )

    table_cell_bold = ParagraphStyle(
        'TCB',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#0F172A')
    )

    story = []

    # Title block
    story.append(Paragraph("TRUSIGHT Voice Analysis System", title_style))
    story.append(Paragraph("Comprehensive Implementation Guide: Datasets, Architecture, Feature Extraction & Machine Learning Models", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563EB'), spaceBefore=0, spaceAfter=12))

    # Executive Overview Callout
    summary_html = "<b>Executive Summary:</b> TRUSIGHT Voice Service is an intelligent multimodal microservice designed to analyze spoken audio for signs of vocal stress and deception. It extracts 814 dimensional acoustic, neural, and verbal features, feeds them through an ANOVA feature selector, and classifies truthfulness using a Soft Voting Ensemble model (Random Forest + Gradient Boosting)."
    callout_table = Table([[Paragraph(summary_html, callout_style)]], colWidths=[532])
    callout_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#EFF6FF')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#93C5FD')),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(callout_table)
    story.append(Spacer(1, 10))

    # Section 1: End-to-End Implementation Process
    story.append(Paragraph("1. End-to-End Implementation Workflow", h1_style))
    story.append(Paragraph("The system follows a clean 5-stage pipeline from raw audio ingestion to actionable deception insights:", body_style))

    pipeline_steps = [
        ("Step 1: Audio Preprocessing & Normalization", "Uploaded audio files (.wav, .mp3, .m4a, .flac) are decoded using Soundfile and Librosa, converted to 16 kHz mono format, and normalized using peak audio scaling."),
        ("Step 2: Triple-Stream Feature Extraction (814 Features)", "Raw audio is processed simultaneously across 3 streams: 42 Classical Acoustic cues (Praat/Librosa), 768 Deep Neural transformer embeddings (Meta's Wav2Vec 2.0), and 4 Verbal cues via CTC ASR."),
        ("Step 3: Feature Scaling & Top-K ANOVA Feature Selection", "The 814 features are standardized via StandardScaler. SelectKBest (ANOVA F-test) automatically selects the top 50 most discriminative features to eliminate noise and prevent overfitting."),
        ("Step 4: Machine Learning Inference via Soft Voting Ensemble", "The optimized 50 features are fed into a soft-voting ensemble combining a 300-tree Random Forest and a 200-estimator Gradient Boosting Classifier to compute truth/deception probabilities."),
        ("Step 5: Explainable AI & Human-Readable Observations", "In addition to a numerical verdict, rule-based acoustic interpretation (Praat thresholds) generates instant explanations: pitch instability, vocal fry/shimmer, hesitation pauses, and speech rate.")
    ]

    for title, desc in pipeline_steps:
        p_text = f"<b>• {title}:</b> {desc}"
        story.append(Paragraph(p_text, bullet_style))

    story.append(Spacer(1, 10))

    # Section 2: Datasets Used
    story.append(Paragraph("2. Datasets Used & Corpus Formulation", h1_style))
    story.append(Paragraph("To ensure the model trains on authentic vocal stress rather than synthetic or scripted laboratory speech, two real-world deception datasets were integrated into a strictly balanced training corpus:", body_style))

    # Dataset Table
    ds_data = [
        [Paragraph("Dataset Name", table_header_style), Paragraph("Source Reference", table_header_style), Paragraph("Modality", table_header_style), Paragraph("Clips Extracted", table_header_style), Paragraph("Labels", table_header_style), Paragraph("Core Purpose & Characteristics", table_header_style)],
        [
            Paragraph("<b>Real-Life Trial</b>", table_cell_bold),
            Paragraph("Perez-Rosas et al.<br/>(Univ of Michigan)", table_cell_style),
            Paragraph("Voice (Extracted from MP4)", table_cell_style),
            Paragraph("<b>121 clips</b>", table_cell_bold),
            Paragraph("60 Truthful<br/>61 Deceptive", table_cell_style),
            Paragraph("High-stakes courtroom testimony video clips with verified legal exonerations and court verdicts. Genuine stress responses under oath.", table_cell_style)
        ],
        [
            Paragraph("<b>DOLOS / TV Corpus</b>", table_cell_bold),
            Paragraph("ICCV 2023 Benchmark<br/>(Dolos team)", table_cell_style),
            Paragraph("Voice (Extracted from Video)", table_cell_style),
            Paragraph("<b>47 clips</b>", table_cell_bold),
            Paragraph("24 Truthful<br/>23 Deceptive", table_cell_style),
            Paragraph("Televised interrogations, high-profile interviews, and press conferences across varied recording equipment, acoustics, and demographics.", table_cell_style)
        ],
        [
            Paragraph("<b>Combined Voice Corpus</b>", table_cell_bold),
            Paragraph("<b>Total Project Benchmark</b>", table_cell_bold),
            Paragraph("<b>Voice (16kHz WAV)</b>", table_cell_bold),
            Paragraph("<b>168 clips total</b>", table_cell_bold),
            Paragraph("<b>84 Truthful<br/>84 Deceptive</b>", table_cell_bold),
            Paragraph("<b>Balanced 50/50 ratio</b> preventing machine learning class bias while maintaining genuine diversity across pitch, age, and accents.", table_cell_bold)
        ]
    ]

    col_widths = [80, 85, 65, 65, 75, 162]
    ds_table = Table(ds_data, colWidths=col_widths, repeatRows=1)
    ds_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#FFFFFF')),
        ('BACKGROUND', (0,2), (-1,2), colors.HexColor('#F8FAFC')),
        ('BACKGROUND', (0,3), (-1,3), colors.HexColor('#EEF2FF')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(ds_table)
    story.append(Spacer(1, 12))

    # Page Break for clean reading
    story.append(PageBreak())

    # Section 3: Feature Extraction (814 Features)
    story.append(Paragraph("3. Feature Extraction Architecture (814 Total Features)", h1_style))
    story.append(Paragraph("Deception manifests in subtle vocal shifts, subconscious pauses, and lexical choices. The pipeline extracts 814 distinct features spanning 3 complementary layers:", body_style))

    feat_data = [
        [Paragraph("Feature Group", table_header_style), Paragraph("Count", table_header_style), Paragraph("Library / Tool", table_header_style), Paragraph("Significance in Deception & Voice Analysis", table_header_style)],
        [
            Paragraph("<b>Pitch & Fundamental Frequency (F0)</b>", table_cell_bold),
            Paragraph("4", table_cell_style),
            Paragraph("Praat-Parselmouth", table_cell_style),
            Paragraph("Mean F0, Pitch Std, Min F0, Max F0. High stress/arousal contracts laryngeal muscles, causing sudden pitch elevation and volatility.", table_cell_style)
        ],
        [
            Paragraph("<b>Voice Perturbation (Jitter & Shimmer)</b>", table_cell_bold),
            Paragraph("2", table_cell_style),
            Paragraph("Praat-Parselmouth", table_cell_style),
            Paragraph("Local pitch variability (Jitter %) and micro-amplitude variation (Shimmer %). Measures vocal cord tremor and stability under cognitive strain.", table_cell_style)
        ],
        [
            Paragraph("<b>Temporal Dynamics & Pauses</b>", table_cell_bold),
            Paragraph("4", table_cell_style),
            Paragraph("Praat + Librosa", table_cell_style),
            Paragraph("Total duration, active speaking time, speech-to-pause ratio, and hesitation pause counts. Liars frequently exhibit cognitive latency pauses.", table_cell_style)
        ],
        [
            Paragraph("<b>Spectral Energy & Dynamics</b>", table_cell_bold),
            Paragraph("6", table_cell_style),
            Paragraph("Librosa", table_cell_style),
            Paragraph("RMS energy (mean & std), Spectral Centroid (brightness/sharpness), and Spectral Flux (rate of spectral change across time frames).", table_cell_style)
        ],
        [
            Paragraph("<b>Mel-Frequency Cepstral Coeffs (MFCCs)</b>", table_cell_bold),
            Paragraph("26", table_cell_style),
            Paragraph("Librosa", table_cell_style),
            Paragraph("13 MFCC means + 13 MFCC standard deviations. Captures vocal tract shape, timbre, resonance, and articulation changes.", table_cell_style)
        ],
        [
            Paragraph("<b>Deep Audio Embeddings</b>", table_cell_bold),
            Paragraph("<b>768</b>", table_cell_bold),
            Paragraph("Hugging Face / Meta<br/>Wav2Vec 2.0", table_cell_style),
            Paragraph("Mean-pooled latent hidden state from Meta's facebook/wav2vec2-base-960h transformer. Captures complex temporal speech patterns.", table_cell_style)
        ],
        [
            Paragraph("<b>Verbal / Linguistic Cues (ASR)</b>", table_cell_bold),
            Paragraph("4", table_cell_style),
            Paragraph("Wav2Vec2 CTC Decoder", table_cell_style),
            Paragraph("1st-person pronoun density ('I', 'we', 'my'), hedge-word density ('maybe', 'guess', 'think'), word count, and average word length.", table_cell_style)
        ]
    ]

    feat_table = Table(feat_data, colWidths=[120, 45, 95, 272], repeatRows=1)
    feat_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#FFFFFF')),
        ('BACKGROUND', (0,2), (-1,2), colors.HexColor('#F8FAFC')),
        ('BACKGROUND', (0,3), (-1,3), colors.HexColor('#FFFFFF')),
        ('BACKGROUND', (0,4), (-1,4), colors.HexColor('#F8FAFC')),
        ('BACKGROUND', (0,5), (-1,5), colors.HexColor('#FFFFFF')),
        ('BACKGROUND', (0,6), (-1,6), colors.HexColor('#EEF2FF')),
        ('BACKGROUND', (0,7), (-1,7), colors.HexColor('#FFFFFF')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(feat_table)
    story.append(Spacer(1, 12))

    # Section 4: Models & Machine Learning Architecture
    story.append(Paragraph("4. Models & Algorithms Used", h1_style))
    story.append(Paragraph("The system employs a dual-stage architecture combining deep self-supervised neural networks for representation learning with ensemble tree algorithms for robust classification:", body_style))

    story.append(Paragraph("A. Feature Representation & ASR Model: Meta Wav2Vec 2.0", h2_style))
    story.append(Paragraph("• <b>Model Identifier:</b> <code>facebook/wav2vec2-base-960h</code> (PyTorch + Hugging Face Transformers)<br/>"
                           "• <b>Acoustic Representation:</b> Feeds 16 kHz raw waveforms through multi-layer convolutional feature encoders followed by 12 transformer blocks, extracting a 768-dimensional context vector via temporal average pooling.<br/>"
                           "• <b>Built-in Speech-to-Text:</b> Utilizes the integrated CTC (Connectionist Temporal Classification) head to generate automatic speech transcripts locally without third-party API dependencies.", bullet_style))

    story.append(Paragraph("B. Feature Normalization & Dimensionality Reduction", h2_style))
    story.append(Paragraph("• <b>StandardScaler:</b> Standardizes all 814 features to zero mean and unit variance.<br/>"
                           "• <b>SelectKBest (ANOVA F-value):</b> Reduces the 814 features down to the <b>top 50 most statistically significant features</b>. This prevents the curse of dimensionality and guards against overfitting.", bullet_style))

    story.append(Paragraph("C. Decision Classifier: Soft-Voting Ensemble", h2_style))
    story.append(Paragraph("• <b>Random Forest Classifier:</b> 300 decision trees, <code>max_depth=8</code>, <code>min_samples_leaf=2</code>. Handles non-linear feature interactions and provides high resistance to outliers.<br/>"
                           "• <b>Gradient Boosting Classifier:</b> 200 estimators, <code>max_depth=3</code>, <code>learning_rate=0.03</code>. Iteratively corrects residual errors and focuses on hard-to-classify borderline vocal patterns.<br/>"
                           "• <b>Soft Voting Mechanism:</b> Averages the predicted class probabilities of both models to output a calibrated deception probability score between 0.0 and 1.0.", bullet_style))

    story.append(Spacer(1, 10))

    # Section 5: Software & Technology Stack
    story.append(Paragraph("5. Complete Technology & Software Stack", h1_style))

    tech_items = [
        ("Backend Web Framework", "FastAPI + Uvicorn (Asynchronous REST API, multipart audio streaming, dynamic model reloading)"),
        ("Acoustic Feature Extraction", "Praat-Parselmouth (C++ Praat phonetics engine wrapper) & Librosa (DSP, STFT, MFCCs)"),
        ("Deep Learning Framework", "PyTorch + Hugging Face Transformers (Wav2Vec2 neural audio processing)"),
        ("Machine Learning & Statistics", "Scikit-Learn (Pipelines, StandardScaler, SelectKBest, VotingClassifier, Joblib serialization)"),
        ("Audio & Multimedia Processing", "FFmpeg & Soundfile (Automatic conversion from MP4/video to 16 kHz mono WAV)"),
        ("Frontend Interactive UI", "Modern HTML5 Dashboard with CSS3 Glassmorphism, Web Audio API waveform visualizer, and dynamic risk meters")
    ]

    for cat, tools in tech_items:
        story.append(Paragraph(f"<b>• {cat}:</b> {tools}", bullet_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF at: {filename}")

if __name__ == "__main__":
    out_pdf = r"c:\Users\diksh\Downloads\voice_service\voice_service\TRUSIGHT_Voice_Implementation_Guide.pdf"
    build_pdf(out_pdf)
