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
            st.dataframe(df_metrics, use_container_width=True)
            
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
                        
                        # 2. Run ONNX Inference Session
                        session = ort.InferenceSession(onnx_path)
                        input_name = session.get_inputs()[0].name
                        output_name = session.get_outputs()[0].name
                        
                        # Predict
                        raw_prob = float(session.run([output_name], {input_name: X_scaled.astype(np.float32)})[0][0][0])
                        predicted_class = 1 if raw_prob >= 0.5 else 0
                        predicted_class_name = "At Risk" if predicted_class == 1 else "On Track"
                        
                        # 3. Calculate Confidence
                        confidence = calculate_prediction_confidence(raw_prob)
                        
                        st.success(f"Primary ANN inference completed. Predicted: **{predicted_class_name}** with **{raw_prob:.2%}** probability (Confidence: **{confidence:.2%}**).")
                        
                        # 4. Save Prediction record to Relational DB
                        pred_record = PredictionRecord(
                            student_id=student_id,
                            predicted_prob=raw_prob,
                            confidence_score=confidence,
                            predicted_class=predicted_class
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
                        
                        # Fetch SHAP attributions for the prompt
                        # We use background data from DB to explain
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
                        
                        # Format SHAP values text block for LLM prompt
                        shap_text = "\n".join([f"- {k}: {v:.4f}" for k, v in shap_explanation['attributions'].items()])
                        
                        # Run reasoning explanation
                        reasoning_chain = get_reasoning_chain(llm)
                        reasoning_result = reasoning_chain.invoke({
                            "student_id": student_id,
                            "predicted_class_name": predicted_class_name,
                            "predicted_prob": raw_prob,
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
                            "predicted_prob": raw_prob,
                            "confidence_score": confidence,
                            "xai_explanation": reasoning_result,
                            "qualitative_summary": summary_result
                        })
                        
                        # Cache reasoning & summary in Streamlit session state for review/XAI page integration
                        st.session_state["last_run_student"] = student_id
                        st.session_state["last_run_summary"] = summary_result
                        st.session_state["last_run_reasoning"] = reasoning_result
                        st.session_state["last_run_report"] = report_draft
                        
                        st.markdown("### Generated Intervention Draft")
                        st.markdown(report_draft)
                        
                        st.info("💡 This draft has been sent to the **Human-in-the-Loop Review Queue**. Go to the next tab to edit/finalize.")
    except Exception as e:
        st.error(f"Error during inference execution: {e}")
    finally:
        db.close()
