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
    
    # 5. Section 3: Action Plan
    story.append(Paragraph("3. ACTIONABLE INTERVENTION PLAN", section_title_style))
    for i, item in enumerate(data.recommendations):
        story.append(Paragraph(f"<b>{i+1}.</b> {item}", bullet_style))
        
    # Optional Reviewer Notes Block
    if data.reviewer_notes:
        story.append(Spacer(1, 10))
        story.append(Paragraph("HUMAN REVIEW AUDIT NOTES", section_title_style))
        story.append(Paragraph(data.reviewer_notes, body_style))
        
    # Build PDF
    doc.build(story)
