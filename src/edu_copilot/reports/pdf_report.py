from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from edu_copilot.reports.report_data import ReportData

def generate_pdf_report(data: ReportData, output_stream: BytesIO) -> None:
    """
    Generates a professional academic intervention report in PDF format
    from a ReportData instance and writes it to the output stream.
    
    Args:
        data (ReportData): Encapsulated report data.
        output_stream (BytesIO): Stream to write PDF binary to.
    """
    # Create the document with standard letter page size and margins
    doc = SimpleDocTemplate(
        output_stream,
        pagesize=letter,
        rightMargin=54, 
        leftMargin=54, 
        topMargin=54, 
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Define custom color-harmonious styles
    primary_color = colors.HexColor('#1E3A8A')   # Navy Blue
    secondary_color = colors.HexColor('#2563EB') # Royal Blue
    text_color = colors.HexColor('#1F2937')      # Dark Charcoal
    bg_light = colors.HexColor('#F3F4F6')        # Light Grey
    border_color = colors.HexColor('#E5E7EB')    # Subtle Grey
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=primary_color,
        spaceAfter=15
    )
    
    section_title_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=secondary_color,
        spaceBefore=14,
        spaceAfter=8
    )
    
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=text_color,
        spaceAfter=8
    )
    
    bullet_style = ParagraphStyle(
        'DocBullet',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=6
    )
    
    story = []
    
    # 1. Main Title Header
    story.append(Paragraph("Student Academic Intervention Report", title_style))
    story.append(Spacer(1, 5))
    
    # 2. Metadata Grid Table
    meta_table_content = [
        [
            Paragraph(f"<b>Student Name:</b> {data.student_name}", body_style), 
            Paragraph(f"<b>Student ID:</b> {data.student_id}", body_style)
        ],
        [
            Paragraph(f"<b>Grade Level:</b> {data.grade_level}", body_style), 
            Paragraph(f"<b>Review Status:</b> {data.reviewed_status}", body_style)
        ],
        [
            Paragraph(f"<b>Assessed Risk Level:</b> {data.predicted_risk}", body_style), 
            Paragraph(f"<b>Model Prediction Confidence:</b> {data.confidence_score:.1%}", body_style)
        ]
    ]
    
    meta_table = Table(meta_table_content, colWidths=[250, 250])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg_light),
        ('BOX', (0,0), (-1,-1), 1, border_color),
        ('INNERGRID', (0,0), (-1,-1), 0.5, border_color),
        ('PADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 15))
    
    # 3. Section 1: Executive Summary
    story.append(Paragraph("1. EXECUTIVE SUMMARY", section_title_style))
    story.append(Paragraph(data.summary, body_style))
    story.append(Spacer(1, 10))
    
    # 4. Section 2: Diagnostics & Metrics Table
    story.append(Paragraph("2. CORE TRIGGERS & DIAGNOSTICS", section_title_style))
    story.append(Paragraph(data.reasoning, body_style))
    story.append(Spacer(1, 5))
    
    # Detailed quantitative stats grid
    metrics_data = [
        [Paragraph("<b>Academic Indicator</b>", body_style), Paragraph("<b>Observed Value</b>", body_style)],
        [Paragraph("Cumulative GPA", body_style), Paragraph(f"{data.gpa:.2f}", body_style)],
        [Paragraph("Attendance Rate", body_style), Paragraph(f"{data.attendance_rate:.1%}", body_style)],
        [Paragraph("Weekly Study Commitment", body_style), Paragraph(f"{data.study_hours:.1f} hours", body_style)],
        [Paragraph("Prior Exam Grade", body_style), Paragraph(f"{data.previous_grade:.1f}%", body_style)],
        [Paragraph("Parental Involvement Level", body_style), Paragraph(data.parental_involvement, body_style)]
    ]
    
    metrics_table = Table(metrics_data, colWidths=[200, 200])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E5E7EB')),
        ('BOX', (0,0), (-1,-1), 1, border_color),
        ('INNERGRID', (0,0), (-1,-1), 0.5, border_color),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 15))
    
    # 4.1. Multimodal Sub-scores Table
    story.append(Paragraph("MODEL MULTIMODAL SUB-SCORES", section_title_style))
    story.append(Paragraph(
        "Below are the individual risk and quality probabilities extracted from each data modality. "
        "These are normalized and fused into the final composite risk score.", body_style
    ))
    story.append(Spacer(1, 5))
    
    cnn_mean = (data.cnn_legibility + data.cnn_correctness + data.cnn_completeness) / 3.0 if data.cnn_legibility is not None else None
    subscores_data = [
        [Paragraph("<b>Predictive Modality</b>", body_style), Paragraph("<b>Sub-Score Value</b>", body_style)],
        [Paragraph("Tabular ANN Student Risk", body_style), Paragraph(f"{data.tabular_score:.1%}" if data.tabular_score is not None else "N/A", body_style)],
        [Paragraph("Worksheet CNN Quality (Mean)", body_style), Paragraph(f"{cnn_mean:.1%} (Legibility: {data.cnn_legibility:.0%}, Correctness: {data.cnn_correctness:.0%}, Completeness: {data.cnn_completeness:.0%})" if cnn_mean is not None else "N/A", body_style)],
        [Paragraph("Weekly Activity LSTM Risk", body_style), Paragraph(f"{data.timeseries_score:.1%}" if data.timeseries_score is not None else "N/A", body_style)],
        [Paragraph("Oral Exam GRU Fluency", body_style), Paragraph(f"{data.audio_score:.1%}" if data.audio_score is not None else "N/A", body_style)],
        [Paragraph("<b>Fused Composite Recommendation Risk</b>", body_style), Paragraph(f"<b>{data.fused_score:.1%}</b>" if data.fused_score is not None else "N/A", body_style)]
    ]
    
    subscores_table = Table(subscores_data, colWidths=[180, 260])
    subscores_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E5E7EB')),
        ('BOX', (0,0), (-1,-1), 1, border_color),
        ('INNERGRID', (0,0), (-1,-1), 0.5, border_color),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(subscores_table)
    story.append(Spacer(1, 15))
    
    # 5. Section 3: Action Plan
    story.append(Paragraph("3. ACTIONABLE INTERVENTION PLAN", section_title_style))
    for i, item in enumerate(data.recommendations):
        story.append(Paragraph(f"<b>{i+1}.</b> {item}", bullet_style))
        
    # 5.1. Section 4: Explainability Maps
    import os
    from reportlab.platypus import Image as RLImage
    
    images_flowables = []
    
    # Check and add Grad-CAM image
    if data.gradcam_image_path and os.path.exists(data.gradcam_image_path):
        try:
            img1 = RLImage(data.gradcam_image_path, width=130, height=130)
            images_flowables.append(img1)
        except Exception:
            pass
            
    # Check and add time-series attention chart
    if data.timeseries_attention_path and os.path.exists(data.timeseries_attention_path):
        try:
            img2 = RLImage(data.timeseries_attention_path, width=140, height=80)
            images_flowables.append(img2)
        except Exception:
            pass
            
    # Check and add audio attention chart
    if data.audio_attention_path and os.path.exists(data.audio_attention_path):
        try:
            img3 = RLImage(data.audio_attention_path, width=140, height=80)
            images_flowables.append(img3)
        except Exception:
            pass
            
    if images_flowables:
        story.append(Spacer(1, 10))
        story.append(Paragraph("4. MULTIMODAL EXPLAINABILITY ARTIFACTS", section_title_style))
        story.append(Paragraph("Saliency mapping (CNN Grad-CAM) and temporal attention weighting charts (Time Series & Audio) highlighting key inference regions:", body_style))
        story.append(Spacer(1, 8))
        
        # Place images in a row layout table
        images_table = Table([images_flowables], colWidths=[150] * len(images_flowables))
        images_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('PADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(images_table)
        
    # Optional Reviewer Notes Block
    if data.reviewer_notes:
        story.append(Spacer(1, 10))
        story.append(Paragraph("HUMAN REVIEW AUDIT NOTES", section_title_style))
        story.append(Paragraph(data.reviewer_notes, body_style))
        
    # Build PDF
    doc.build(story)
