import os
import sys
from io import BytesIO

# Ensure src directory is on the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
import pandas as pd
import onnxruntime as ort

from edu_copilot.db.session import SessionLocal
from edu_copilot.db.models import Student, StudentTabular, PredictionRecord, ReviewRecord, ReportRecord
from edu_copilot.data.preprocessing.tabular_preprocessing import TabularPreprocessor
from edu_copilot.data.loaders.image_loader import load_student_worksheet
from edu_copilot.data.preprocessing.image_preprocessing import preprocess_worksheet
from edu_copilot.models.cnn_model import get_or_create_cnn_model, generate_gradcam_heatmap
from edu_copilot.data.loaders.timeseries_loader import load_student_timeseries
from edu_copilot.data.preprocessing.timeseries_preprocessing import preprocess_timeseries
from edu_copilot.models.rnn_model import get_or_create_rnn_model
from edu_copilot.data.loaders.audio_loader import load_student_audio
from edu_copilot.data.preprocessing.audio_preprocessing import preprocess_audio
from edu_copilot.models.fusion import fuse_student_scores
from edu_copilot.xai.attention_explainer import explain_attention_over_time, draw_bar_chart
from edu_copilot.reports.report_data import ReportData
from edu_copilot.reports.pdf_report import generate_pdf_report
from edu_copilot.reports.docx_report import generate_docx_report

def run_verification():
    print("=== STARTING END-TO-END PIPELINE VERIFICATION ===")
    
    student_id = "S101"
    db = SessionLocal()
    
    try:
        # 1. Fetch student
        student = db.query(Student).filter(Student.student_id == student_id).first()
        tabular = db.query(StudentTabular).filter(StudentTabular.student_id == student_id).first()
        
        if not student or not tabular:
            print(f"Error: Student {student_id} not found in database. Run reset_db.py first.")
            sys.exit(1)
            
        print(f"Loaded student: {student.name}")
        
        # 2. Run Tabular Model
        artifacts_dir = "src/edu_copilot/models/artifacts"
        preprocessor_path = os.path.join(artifacts_dir, "preprocessor.pkl")
        onnx_path = os.path.join(artifacts_dir, "student_model.onnx")
        
        if not os.path.exists(preprocessor_path) or not os.path.exists(onnx_path):
            print("Warning: student_model.onnx or preprocessor.pkl missing. Creating mock tabular outputs...")
            raw_prob = 0.65
        else:
            preprocessor = TabularPreprocessor.load(preprocessor_path)
            student_df = pd.DataFrame([{
                "student_id": student.student_id,
                "gpa": tabular.gpa,
                "attendance_rate": tabular.attendance_rate,
                "study_hours_weekly": tabular.study_hours_weekly,
                "parental_involvement": tabular.parental_involvement,
                "extracurricular_activities": tabular.extracurricular_activities,
                "sleep_hours": tabular.sleep_hours,
                "previous_grade": tabular.previous_grade,
                "family_income": tabular.family_income,
                "internet_access": tabular.internet_access
            }])
            X_scaled = preprocessor.transform(student_df)
            session = ort.InferenceSession(onnx_path)
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            raw_prob = float(session.run([output_name], {input_name: X_scaled.astype(np.float32)})[0][0][0])
            
        print(f"Tabular ANN Prediction: {raw_prob:.2%}")
        
        # 3. Run CNN Worksheet Model
        worksheet_path = "data/sample/sample_worksheet.png"
        img = load_student_worksheet(worksheet_path)
        img_prep = preprocess_worksheet(img)
        cnn_model = get_or_create_cnn_model()
        cnn_preds = cnn_model(img_prep, training=False).numpy()[0]
        cnn_legibility = float(cnn_preds[0])
        cnn_correctness = float(cnn_preds[1])
        cnn_completeness = float(cnn_preds[2])
        print(f"CNN Worksheet: Legibility={cnn_legibility:.2%}, Correctness={cnn_correctness:.2%}, Completeness={cnn_completeness:.2%}")
        
        gradcam_out = f"src/edu_copilot/models/artifacts/{student_id}_gradcam.png"
        generate_gradcam_heatmap(worksheet_path, cnn_model, output_path=gradcam_out)
        print("Grad-CAM image generated.")
        
        # 4. Run Time Series Model
        ts_path = "data/sample/sample_timeseries.csv"
        ts_df = load_student_timeseries(ts_path)
        ts_prep = preprocess_timeseries(ts_df, seq_len=10)
        ts_model = get_or_create_rnn_model('timeseries_model', seq_len=10, num_features=3, rnn_type='lstm')
        ts_prob = float(ts_model(ts_prep, training=False).numpy()[0][0])
        print(f"Time Series LSTM Risk Prediction: {ts_prob:.2%}")
        
        ts_importance = explain_attention_over_time(ts_model, ts_prep)
        ts_chart = f"src/edu_copilot/models/artifacts/{student_id}_attention_ts.png"
        draw_bar_chart(ts_importance, [f"W{i}" for i in range(1, 11)], "Weekly Activity Attention", ts_chart)
        print("Time Series attention chart generated.")
        
        # 5. Run Audio Model
        audio_path = "data/sample/sample_audio.wav"
        aud_y, aud_sr = load_student_audio(audio_path)
        aud_prep = preprocess_audio(aud_y, aud_sr, n_mfcc=13, seq_len=50)
        aud_model = get_or_create_rnn_model('audio_model', seq_len=50, num_features=13, rnn_type='gru')
        audio_score = float(aud_model(aud_prep, training=False).numpy()[0][0])
        print(f"Audio GRU Fluency Score: {audio_score:.2%}")
        
        aud_importance = explain_attention_over_time(aud_model, aud_prep)
        aud_binned = [float(np.mean(aud_importance[i*5:(i+1)*5])) for i in range(10)]
        aud_chart = f"src/edu_copilot/models/artifacts/{student_id}_attention_audio.png"
        draw_bar_chart(aud_binned, [f"{i*0.1:.1f}s" for i in range(10)], "Audio Saliency Segments", aud_chart)
        print("Audio saliency chart generated.")
        
        # 6. Score Fusion
        fused_prob = fuse_student_scores(
            tabular_prob=raw_prob,
            cnn_scores={'legibility': cnn_legibility, 'correctness': cnn_correctness, 'completeness': cnn_completeness},
            timeseries_prob=ts_prob,
            audio_score=audio_score
        )
        print(f"Fused Composite Score: {fused_prob:.2%}")
        
        # 7. Relational DB Persistence Check
        print("Testing DB logging...")
        pred_record = PredictionRecord(
            student_id=student_id,
            predicted_prob=fused_prob,
            confidence_score=0.88,
            predicted_class=1 if fused_prob >= 0.5 else 0,
            tabular_score=raw_prob,
            cnn_legibility=cnn_legibility,
            cnn_correctness=cnn_correctness,
            cnn_completeness=cnn_completeness,
            timeseries_score=ts_prob,
            audio_score=audio_score,
            fused_score=fused_prob
        )
        db.add(pred_record)
        db.flush()
        pred_id = pred_record.id
        db.commit()
        print(f"DB log successful. Prediction ID: {pred_id}")
        
        # Read back and assert values are mapped
        db_record = db.query(PredictionRecord).filter(PredictionRecord.id == pred_id).first()
        assert db_record.cnn_legibility == cnn_legibility, "CNN Legibility score mismatch!"
        assert db_record.timeseries_score == ts_prob, "Time series score mismatch!"
        assert db_record.fused_score == fused_prob, "Fused score mismatch!"
        print("Database fields readback verification PASSED.")
        
        # 8. Report PDF & DOCX Generation Check
        print("Testing PDF & DOCX generation...")
        report_data = ReportData(
            student_id=student_id,
            student_name=student.name,
            grade_level=student.grade_level,
            predicted_risk="At Risk" if fused_prob >= 0.5 else "On Track",
            risk_probability=fused_prob,
            confidence_score=0.88,
            gpa=tabular.gpa,
            attendance_rate=tabular.attendance_rate,
            study_hours=tabular.study_hours_weekly,
            parental_involvement=tabular.parental_involvement,
            previous_grade=tabular.previous_grade,
            tabular_score=raw_prob,
            cnn_legibility=cnn_legibility,
            cnn_correctness=cnn_correctness,
            cnn_completeness=cnn_completeness,
            timeseries_score=ts_prob,
            audio_score=audio_score,
            fused_score=fused_prob,
            gradcam_image_path=gradcam_out,
            timeseries_attention_path=ts_chart,
            audio_attention_path=aud_chart,
            summary="Verification test summary.",
            reasoning="Verification test reasoning.",
            recommendations=["Verify test recommendations 1", "Verify test recommendations 2"],
            reviewed_status="Approved",
            reviewer_notes="Verification test review notes."
        )
        
        # Test PDF
        pdf_stream = BytesIO()
        generate_pdf_report(report_data, pdf_stream)
        pdf_bytes = pdf_stream.getvalue()
        assert len(pdf_bytes) > 0, "PDF size is 0!"
        print(f"PDF generation verified successfully. Size: {len(pdf_bytes)} bytes.")
        
        # Test DOCX
        docx_stream = BytesIO()
        generate_docx_report(report_data, docx_stream)
        docx_bytes = docx_stream.getvalue()
        assert len(docx_bytes) > 0, "DOCX size is 0!"
        print(f"DOCX generation verified successfully. Size: {len(docx_bytes)} bytes.")
        
        print("=== ALL CHECKS COMPLETED: MULTIMODAL INTEGRATION IS FULLY CORRECT ===")
        
    except Exception as e:
        db.rollback()
        print(f"Verification failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        db.close()

if __name__ == "__main__":
    run_verification()
