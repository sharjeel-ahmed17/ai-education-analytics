import os
import numpy as np
import pandas as pd
from io import BytesIO
from sqlalchemy.orm import Session
import onnxruntime as ort
from edu_copilot.db.models import Student, StudentTabular, PredictionRecord, ReviewRecord, ReportRecord
from edu_copilot.data.preprocessing.tabular_preprocessing import TabularPreprocessor
from edu_copilot.xai.confidence import calculate_prediction_confidence
from edu_copilot.hitl.workflow import create_pending_review, approve_recommendation
from edu_copilot.reports.report_data import ReportData
from edu_copilot.reports.pdf_report import generate_pdf_report
from edu_copilot.reports.docx_report import generate_docx_report

def test_full_pipeline_end_to_end(db_session: Session, temp_dir: str) -> None:
    """
    Integration test checking data ingestion, ONNX inference, 
    relational schema logging, review queue transitions, and 
    document compilation.
    """
    # 1. Ingest Student and Tabular values
    student = Student(
        student_id="I101",
        name="Ingrid Integration",
        grade_level="12th Grade"
    )
    db_session.add(student)
    db_session.flush()
    
    tabular = StudentTabular(
        student_id="I101",
        gpa=3.1,
        attendance_rate=0.88,
        study_hours_weekly=8.0,
        parental_involvement="Medium",
        extracurricular_activities=True,
        sleep_hours=7.0,
        previous_grade=82.0,
        family_income="Medium",
        internet_access=True
    )
    db_session.add(tabular)
    db_session.commit()
    
    assert db_session.query(Student).count() == 1
    assert db_session.query(StudentTabular).count() == 1
    
    # 2. Run Preprocessor and ONNX prediction
    artifacts_dir = "src/edu_copilot/models/artifacts"
    preprocessor_path = os.path.join(artifacts_dir, "preprocessor.pkl")
    onnx_path = os.path.join(artifacts_dir, "student_model.onnx")
    
    assert os.path.exists(preprocessor_path), "Missing preprocessor pickle"
    assert os.path.exists(onnx_path), "Missing ONNX model weight"
    
    # Load and scale
    preprocessor = TabularPreprocessor.load(preprocessor_path)
    
    student_df = pd.DataFrame([{
        "student_id": "I101",
        "gpa": 3.1,
        "attendance_rate": 0.88,
        "study_hours_weekly": 8.0,
        "parental_involvement": "Medium",
        "extracurricular_activities": True,
        "sleep_hours": 7.0,
        "previous_grade": 82.0,
        "family_income": "Medium",
        "internet_access": True
    }])
    
    X_scaled = preprocessor.transform(student_df)
    
    # Load ONNX model and run inference
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    raw_prob = float(session.run([output_name], {input_name: X_scaled.astype(np.float32)})[0][0][0])
    predicted_class = 1 if raw_prob >= 0.5 else 0
    confidence = calculate_prediction_confidence(raw_prob)
    
    # 3. Log Prediction to Database
    pred_record = PredictionRecord(
        student_id="I101",
        predicted_prob=raw_prob,
        confidence_score=confidence,
        predicted_class=predicted_class
    )
    db_session.add(pred_record)
    db_session.flush()
    
    # 4. Trigger review queue transition
    review = create_pending_review(db_session, student_id="I101", prediction_id=pred_record.id)
    assert review.status == "Pending"
    
    approve_recommendation(db_session, review_id=review.id, notes="Approved during integration run")
    assert review.status == "Approved"
    
    # Save finalized ReportRecord in database
    report_record = ReportRecord(
        student_id="I101",
        prediction_id=pred_record.id,
        review_id=review.id,
        summary="Integration student is doing okay, minor warnings.",
        reasoning="Attendance and GPA are reasonable, minor study hours drop.",
        recommendations="Monitor daily attendance\nWeekly tutor sessions"
    )
    db_session.add(report_record)
    db_session.commit()
    
    # 5. Compile binary reports
    report_data = ReportData(
        student_id="I101",
        student_name="Ingrid Integration",
        grade_level="12th Grade",
        predicted_risk="At Risk" if predicted_class == 1 else "On Track",
        risk_probability=raw_prob,
        confidence_score=confidence,
        gpa=3.1,
        attendance_rate=0.88,
        study_hours=8.0,
        parental_involvement="Medium",
        previous_grade=82.0,
        summary="Integration student is doing okay, minor warnings.",
        reasoning="Attendance and GPA are reasonable, minor study hours drop.",
        recommendations=["Monitor daily attendance", "Weekly tutor sessions"],
        reviewed_status="Approved",
        reviewer_notes="Approved during integration run"
    )
    
    # PDF Compilation check
    pdf_stream = BytesIO()
    generate_pdf_report(report_data, pdf_stream)
    pdf_bytes = pdf_stream.getvalue()
    assert len(pdf_bytes) > 0
    assert pdf_bytes.startswith(b"%PDF"), "Output stream should contain PDF signature"
    
    # DOCX Compilation check
    docx_stream = BytesIO()
    generate_docx_report(report_data, docx_stream)
    docx_bytes = docx_stream.getvalue()
    assert len(docx_bytes) > 0
    assert docx_bytes.startswith(b"PK"), "Output stream should contain DOCX Zip signature"
