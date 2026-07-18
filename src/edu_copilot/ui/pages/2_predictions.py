import os
import numpy as np
import pandas as pd
import streamlit as st
import onnxruntime as ort
from edu_copilot.db.session import SessionLocal
from edu_copilot.db.models import Student, StudentTabular, PredictionRecord, ReviewRecord
from edu_copilot.data.preprocessing.tabular_preprocessing import TabularPreprocessor
from edu_copilot.xai.confidence import calculate_prediction_confidence
from edu_copilot.xai.shap_explainer import get_shap_explanations
from edu_copilot.vectorstore.qdrant_client import QdrantVectorStore
from edu_copilot.llm.embeddings import get_embeddings_model
from edu_copilot.llm.providers import get_llm
from edu_copilot.llm.chains.summarization_chain import get_summarization_chain
from edu_copilot.llm.chains.reasoning_chain import get_reasoning_chain
from edu_copilot.llm.chains.report_chain import get_report_chain

# New imports for Image, Time-series, Audio modalities
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


st.set_page_config(layout="wide")

st.title("Run AI Co-Pilot Inference")
st.markdown("Select a student to run our primary ANN model (via ONNX runtime) and generate LangChain reasoning.")

# Load database connections
db = SessionLocal()
try:
    students = db.query(Student).all()
finally:
    db.close()

if not students:
    st.warning("No students found in the database. Please upload data first.")
else:
    # Build dropdown selections
    options_map = {f"{s.name} ({s.student_id})": s.student_id for s in students}
    selected_label = st.selectbox("Select Student Profile", list(options_map.keys()))
    student_id = options_map[selected_label]
    
    # Fetch student details
    db = SessionLocal()
    try:
        student = db.query(Student).filter(Student.student_id == student_id).first()
        tabular = db.query(StudentTabular).filter(StudentTabular.student_id == student_id).first()
        
        if not tabular:
            st.error("This student does not have tabular performance data. Cannot run prediction.")
        else:
            st.subheader(f"Academic Metrics for {student.name}")
            
            # Map tabular data back to a dictionary and show in UI
            metrics_dict = {
                "GPA": [tabular.gpa],
                "Attendance Rate": [tabular.attendance_rate],
                "Weekly Study Hours": [tabular.study_hours_weekly],
                "Sleep Hours": [tabular.sleep_hours],
                "Previous Course Grade": [tabular.previous_grade],
                "Parental Involvement": [tabular.parental_involvement],
                "Extracurriculars": [tabular.extracurricular_activities],
                "Family Income": [tabular.family_income],
                "Internet Access": [tabular.internet_access]
            }
            
            df_metrics = pd.DataFrame(metrics_dict)
            st.dataframe(df_metrics, width='stretch')
            
            # Button to trigger pipeline
            if st.button("Generate Diagnostic & Prediction"):
                with st.spinner("Executing ONNX predictive inference and RAG reasoning..."):
                    # 1. Prepare data for model
                    artifacts_dir = "src/edu_copilot/models/artifacts"
                    preprocessor_path = os.path.join(artifacts_dir, "preprocessor.pkl")
                    onnx_path = os.path.join(artifacts_dir, "student_model.onnx")
                    
                    if not os.path.exists(preprocessor_path) or not os.path.exists(onnx_path):
                        st.error("Model artifacts (preprocessor.pkl / student_model.onnx) are missing. Please run model training first.")
                    else:
                        # Load Preprocessor and run transformation
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
                        
                        # 2. Run ONNX Inference Session (Tabular ANN)
                        session = ort.InferenceSession(onnx_path)
                        input_name = session.get_inputs()[0].name
                        output_name = session.get_outputs()[0].name
                        raw_prob = float(session.run([output_name], {input_name: X_scaled.astype(np.float32)})[0][0][0])
                        
                        # 2.1 Run CNN Image Model (Scanned Worksheet)
                        student_worksheet_path = f"data/sample/students/{student_id}/worksheet.png"
                        if not os.path.exists(student_worksheet_path):
                            student_worksheet_path = "data/sample/sample_worksheet.png"
                            
                        img = load_student_worksheet(student_worksheet_path)
                        img_preprocessed = preprocess_worksheet(img)
                        cnn_model = get_or_create_cnn_model()
                        cnn_preds = cnn_model(img_preprocessed, training=False).numpy()[0]
                        cnn_legibility = float(cnn_preds[0])
                        cnn_correctness = float(cnn_preds[1])
                        cnn_completeness = float(cnn_preds[2])
                        
                        gradcam_out_path = f"src/edu_copilot/models/artifacts/{student_id}_gradcam.png"
                        generate_gradcam_heatmap(student_worksheet_path, cnn_model, output_path=gradcam_out_path)
                        
                        # 2.2 Run LSTM Time Series Model (Weekly Logs)
                        student_ts_path = f"data/sample/students/{student_id}/timeseries.csv"
                        if not os.path.exists(student_ts_path):
                            student_ts_path = "data/sample/sample_timeseries.csv"
                            
                        ts_df = load_student_timeseries(student_ts_path)
                        ts_preprocessed = preprocess_timeseries(ts_df, seq_len=10)
                        ts_model = get_or_create_rnn_model('timeseries_model', seq_len=10, num_features=3, rnn_type='lstm')
                        ts_prob = float(ts_model(ts_preprocessed, training=False).numpy()[0][0])
                        
                        ts_importance = explain_attention_over_time(ts_model, ts_preprocessed)
                        ts_chart_path = f"src/edu_copilot/models/artifacts/{student_id}_attention_ts.png"
                        draw_bar_chart(ts_importance, [f"W{i}" for i in range(1, 11)], "Time Series Weekly Attention", ts_chart_path)
                        
                        # 2.3 Run GRU Audio Model (Oral Response)
                        student_aud_path = f"data/sample/students/{student_id}/audio.wav"
                        if not os.path.exists(student_aud_path):
                            student_aud_path = "data/sample/sample_audio.wav"
                            
                        aud_y, aud_sr = load_student_audio(student_aud_path)
                        aud_preprocessed = preprocess_audio(aud_y, aud_sr, n_mfcc=13, seq_len=50)
                        aud_model = get_or_create_rnn_model('audio_model', seq_len=50, num_features=13, rnn_type='gru')
                        audio_score = float(aud_model(aud_preprocessed, training=False).numpy()[0][0])
                        
                        aud_importance = explain_attention_over_time(aud_model, aud_preprocessed)
                        aud_binned = [float(np.mean(aud_importance[i*5:(i+1)*5])) for i in range(10)]
                        aud_labels = [f"{i*0.1:.1f}s" for i in range(10)]
                        aud_chart_path = f"src/edu_copilot/models/artifacts/{student_id}_attention_audio.png"
                        draw_bar_chart(aud_binned, aud_labels, "Audio Saliency Segments", aud_chart_path)
                        
                        # 2.4 Score Fusion
                        fused_prob = fuse_student_scores(
                            tabular_prob=raw_prob,
                            cnn_scores={'legibility': cnn_legibility, 'correctness': cnn_correctness, 'completeness': cnn_completeness},
                            timeseries_prob=ts_prob,
                            audio_score=audio_score
                        )
                        
                        predicted_class = 1 if fused_prob >= 0.5 else 0
                        predicted_class_name = "At Risk" if predicted_class == 1 else "On Track"
                        confidence = calculate_prediction_confidence(fused_prob)
                        
                        st.success(f"Multi-modal Fusion inference completed. Predicted Fused Score: **{predicted_class_name}** with **{fused_prob:.2%}** probability (Confidence: **{confidence:.2%}**).")
                        
                        # Render sub-scores in nice UI cards
                        cols = st.columns(4)
                        with cols[0]:
                            st.metric("Tabular ANN Risk", f"{raw_prob:.1%}")
                        with cols[1]:
                            st.metric("Image CNN Quality", f"{np.mean([cnn_legibility, cnn_correctness, cnn_completeness]):.1%}",
                                      help=f"Legibility: {cnn_legibility:.1%}\nCorrectness: {cnn_correctness:.1%}\nCompleteness: {cnn_completeness:.1%}")
                        with cols[2]:
                            st.metric("Time Series Risk", f"{ts_prob:.1%}")
                        with cols[3]:
                            st.metric("Audio Fluency Score", f"{audio_score:.1%}")
                        
                        # 4. Save Prediction record with all sub-scores to Relational DB
                        pred_record = PredictionRecord(
                            student_id=student_id,
                            predicted_prob=fused_prob,
                            confidence_score=confidence,
                            predicted_class=predicted_class,
                            tabular_score=raw_prob,
                            cnn_legibility=cnn_legibility,
                            cnn_correctness=cnn_correctness,
                            cnn_completeness=cnn_completeness,
                            timeseries_score=ts_prob,
                            audio_score=audio_score,
                            fused_score=fused_prob
                        )
                        db.add(pred_record)
                        db.flush() # Populate pred_record.id
                        
                        # Create default pending human review log entry
                        existing_review = db.query(ReviewRecord).filter(
                            ReviewRecord.student_id == student_id,
                            ReviewRecord.status == "Pending"
                        ).first()
                        
                        if not existing_review:
                            new_review = ReviewRecord(
                                student_id=student_id,
                                prediction_id=pred_record.id,
                                status="Pending"
                            )
                            db.add(new_review)
                            
                        # Commit DB transactions
                        db.commit()
                        
                        # Render Explanation Plots
                        st.markdown("### Multimodal Explainability Diagnostics")
                        exp_cols = st.columns(3)
                        with exp_cols[0]:
                            st.subheader("Worksheet Grad-CAM Saliency")
                            st.image(gradcam_out_path, use_container_width=True)
                        with exp_cols[1]:
                            st.subheader("LSTM Time Series Attention")
                            st.image(ts_chart_path, use_container_width=True)
                        with exp_cols[2]:
                            st.subheader("GRU Audio Segment Attention")
                            st.image(aud_chart_path, use_container_width=True)
                        
                        # 5. Retrieve qualitative documents from Qdrant Vector Store
                        embeddings = get_embeddings_model()
                        vstore = QdrantVectorStore(embeddings)
                        
                        # Fetch relevant documents (notes, report cards)
                        retrieved_docs = vstore.search_student_docs(
                            query="academic behavior attendance difficulties strengths",
                            student_id=student_id,
                            k=3
                        )
                        
                        st.info(f"Retrieved {len(retrieved_docs)} qualitative text segments from Qdrant Vector Store.")
                        
                        # Merge document chunks
                        notes_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                        if not notes_context:
                            notes_context = "[No qualitative teacher logs or PDFs found for this student]"
                            
                        # 6. LangChain Reasoning over Predictions
                        llm = get_llm()
                        
                        # Generate notes summary
                        summarize_chain = get_summarization_chain(llm)
                        summary_result = summarize_chain.invoke({
                            "student_id": student_id,
                            "document_text": notes_context
                        })
                        
                        # Augment summary with multimodal details so LLM reasoning incorporates it
                        multimodal_summary = (
                            summary_result + 
                            f"\n\nMULTIMODAL DIAGNOSTICS DETECTED:\n"
                            f"- Primary Tabular ANN Risk prediction: {raw_prob:.1%}\n"
                            f"- Scanned Handwritten Worksheet CNN quality metrics: Legibility={cnn_legibility:.1%}, Correctness={cnn_correctness:.1%}, Completeness={cnn_completeness:.1%}\n"
                            f"- Weekly Student LMS engagement Time Series LSTM forecasted risk: {ts_prob:.1%}\n"
                            f"- Speech response Audio GRU Fluency/Confidence score: {audio_score:.1%}\n"
                            f"- Fused Composite Decision Risk Index: {fused_prob:.1%}"
                        )
                        
                        # Fetch SHAP attributions for the prompt
                        background_tabular_records = db.query(StudentTabular).all()
                        background_df = pd.DataFrame([{
                            "student_id": b.student_id,
                            "gpa": b.gpa,
                            "attendance_rate": b.attendance_rate,
                            "study_hours_weekly": b.study_hours_weekly,
                            "parental_involvement": b.parental_involvement,
                            "extracurricular_activities": b.extracurricular_activities,
                            "sleep_hours": b.sleep_hours,
                            "previous_grade": b.previous_grade,
                            "family_income": b.family_income,
                            "internet_access": b.internet_access
                        } for b in background_tabular_records])
                        
                        shap_explanation = get_shap_explanations(student_df, background_df, artifacts_dir)
                        shap_text = "\n".join([f"- {k}: {v:.4f}" for k, v in shap_explanation['attributions'].items()])
                        
                        # Run reasoning explanation
                        reasoning_chain = get_reasoning_chain(llm)
                        reasoning_result = reasoning_chain.invoke({
                            "student_id": student_id,
                            "predicted_class_name": predicted_class_name,
                            "predicted_prob": fused_prob,
                            "confidence_score": confidence,
                            "gpa": tabular.gpa,
                            "attendance_rate": tabular.attendance_rate,
                            "study_hours_weekly": tabular.study_hours_weekly,
                            "previous_grade": tabular.previous_grade,
                            "parental_involvement": tabular.parental_involvement,
                            "extracurricular_activities": str(tabular.extracurricular_activities),
                            "sleep_hours": tabular.sleep_hours,
                            "family_income": tabular.family_income,
                            "internet_access": str(tabular.internet_access),
                            "shap_attributions": shap_text
                        })
                        
                        # Run final draft report generation
                        report_chain = get_report_chain(llm)
                        report_draft = report_chain.invoke({
                            "student_name": student.name,
                            "student_id": student_id,
                            "predicted_class_name": predicted_class_name,
                            "predicted_prob": fused_prob,
                            "confidence_score": confidence,
                            "xai_explanation": reasoning_result,
                            "qualitative_summary": multimodal_summary
                        })
                        
                        # Cache reasoning & summary in Streamlit session state
                        st.session_state["last_run_student"] = student_id
                        st.session_state["last_run_summary"] = multimodal_summary
                        st.session_state["last_run_reasoning"] = reasoning_result
                        st.session_state["last_run_report"] = report_draft
                        
                        st.markdown("### Generated Multimodal Intervention Draft")
                        st.markdown(report_draft)
                        
                        st.info("💡 This draft has been sent to the **Human-in-the-Loop Review Queue**. Go to the next tab to edit/finalize.")
    except Exception as e:
        st.error(f"Error during inference execution: {e}")
    finally:
        db.close()
