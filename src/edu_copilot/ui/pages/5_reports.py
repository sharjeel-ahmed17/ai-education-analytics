from io import BytesIO
import streamlit as st
import pandas as pd
from edu_copilot.db.session import SessionLocal
from edu_copilot.db.models import Student, StudentTabular, PredictionRecord, ReviewRecord, ReportRecord
from edu_copilot.reports.report_data import ReportData
from edu_copilot.reports.pdf_report import generate_pdf_report
from edu_copilot.reports.docx_report import generate_docx_report

st.set_page_config(layout="wide")

st.title("Export Academic Reports")
st.markdown("Download human-audited student intervention reports in PDF and Word formats.")

# Establish DB connection
db = SessionLocal()
try:
    # Query finalized reports
    finalized_reports = db.query(ReportRecord).all()
    
    if not finalized_reports:
        st.info("ℹ️ No audited reports are available yet. Complete reviews in the **HITL Review Queue** first.")
    else:
        st.subheader("Select a Finalized Report to Export")
        
        # Build options mapping
        report_options = {}
        for rep in finalized_reports:
            student = db.query(Student).filter(Student.student_id == rep.student_id).first()
            if student:
                report_options[f"{student.name} ({student.student_id}) - Finalized on {rep.created_at.strftime('%Y-%m-%d %H:%M')}"] = rep.id
                
        selected_label = st.selectbox("Select Report Profile", list(report_options.keys()))
        selected_report_id = report_options[selected_label]
        
        # Query target record
        report_rec = db.query(ReportRecord).filter(ReportRecord.id == selected_report_id).first()
        student = db.query(Student).filter(Student.student_id == report_rec.student_id).first()
        prediction = db.query(PredictionRecord).filter(PredictionRecord.id == report_rec.prediction_id).first()
        review = db.query(ReviewRecord).filter(ReviewRecord.id == report_rec.review_id).first()
        tabular = db.query(StudentTabular).filter(StudentTabular.student_id == report_rec.student_id).first()
        
        # Load compiled report data
        recs_list = [r.strip() for r in report_rec.recommendations.split("\n") if r.strip()]
        
        report_data = ReportData(
            student_id=student.student_id,
            student_name=student.name,
            grade_level=student.grade_level,
            predicted_risk="At Risk" if prediction.predicted_class == 1 else "On Track",
            risk_probability=prediction.predicted_prob,
            confidence_score=prediction.confidence_score,
            gpa=tabular.gpa if tabular else 0.0,
            attendance_rate=tabular.attendance_rate if tabular else 0.0,
            study_hours=tabular.study_hours_weekly if tabular else 0.0,
            parental_involvement=tabular.parental_involvement if tabular else "Medium",
            previous_grade=tabular.previous_grade if tabular else 0.0,
            summary=report_rec.summary,
            reasoning=report_rec.reasoning,
            recommendations=recs_list,
            reviewed_status=review.status if review else "Approved",
            reviewer_notes=review.reviewer_notes if review else None
        )
        
        # Render the report inside Streamlit for confirmation
        st.markdown("---")
        st.subheader("Report Preview")
        
        st.markdown(f"### Student: {report_data.student_name} (ID: {report_data.student_id})")
        st.markdown(f"**Assessed Risk Level**: {report_data.predicted_risk} | **Prediction Confidence**: {report_data.confidence_score:.2%}")
        
        st.markdown("#### 1. Executive Summary")
        st.write(report_data.summary)
        
        st.markdown("#### 2. Core Triggers & Diagnostics")
        st.write(report_data.reasoning)
        
        st.markdown("#### 3. Actionable Intervention Plan")
        for i, item in enumerate(report_data.recommendations):
            st.write(f"{i+1}. {item}")
            
        if report_data.reviewer_notes:
            st.markdown("#### Human Review Notes")
            st.write(report_data.reviewer_notes)
            
        st.markdown("---")
        
        # Exporter buttons row
        col_pdf, col_docx = st.columns(2)
        
        # 1. Compile PDF stream
        with col_pdf:
            pdf_stream = BytesIO()
            generate_pdf_report(report_data, pdf_stream)
            pdf_stream.seek(0)
            
            st.download_button(
                label="📥 Download PDF Document",
                data=pdf_stream,
                file_name=f"Intervention_Report_{report_data.student_id}.pdf",
                mime="application/pdf",
                width='stretch'
            )
            
        # 2. Compile DOCX stream
        with col_docx:
            docx_stream = BytesIO()
            generate_docx_report(report_data, docx_stream)
            docx_stream.seek(0)
            
            st.download_button(
                label="📥 Download Word Document (DOCX)",
                data=docx_stream,
                file_name=f"Intervention_Report_{report_data.student_id}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                width='stretch'
            )
except Exception as e:
    st.error(f"Error compiling download streams: {e}")
finally:
    db.close()
