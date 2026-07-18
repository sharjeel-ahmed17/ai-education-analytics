import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from edu_copilot.db.session import SessionLocal
from edu_copilot.db.models import Student, StudentTabular
from edu_copilot.xai.feature_importance import get_permutation_importance
from edu_copilot.xai.shap_explainer import get_shap_explanations

st.set_page_config(layout="wide")

st.title("Explainable AI (XAI) Diagnostics")
st.markdown("Inspect local and global feature attribution weights for the primary neural network predictor.")

# Initialize DB connection
db = SessionLocal()
try:
    students = db.query(Student).all()
    background_tabular = db.query(StudentTabular).all()
finally:
    db.close()

if not students or not background_tabular:
    st.warning("No data found in the database. Load tabular records first.")
else:
    # Build background DataFrame for calculations
    bg_df = pd.DataFrame([{
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
    } for b in background_tabular])
    
    tab_global, tab_local, tab_gradcam = st.tabs([
        "📊 Global Feature Importance", 
        "🔍 Local SHAP Explanations", 
        "📷 CV Grad-CAM Stub (Out-of-Scope)"
    ])
    
    # ----------------- TAB 1: GLOBAL PERMUTATION IMPORTANCE -----------------
    with tab_global:
        st.subheader("Neural Network Permutation Feature Importance")
        st.markdown(
            "This chart shows the global contribution of each feature to the ANN model. "
            "It measures the drop in prediction performance (loss increase) when a feature is shuffled."
        )
        
        with st.spinner("Computing permutation importances..."):
            try:
                importances = get_permutation_importance(bg_df)
                
                # Plotly Chart
                df_imp = pd.DataFrame([
                    {"Feature": k.replace("_encoded", "").replace("_", " ").title(), "Importance": v}
                    for k, v in importances.items()
                ])
                
                fig = px.bar(
                    df_imp, 
                    x="Importance", 
                    y="Feature", 
                    orientation='h',
                    title="Global Feature Attributions (Normalized)",
                    labels={"Importance": "Attribution Impact", "Feature": "Input Factor"},
                    color="Importance",
                    color_continuous_scale=px.colors.sequential.Blues
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error calculating feature importances: {e}")
                
    # ----------------- TAB 2: LOCAL SHAP EXPLANATIONS -----------------
    with tab_local:
        st.subheader("Student-Specific SHAP Attributions")
        st.markdown(
            "Select an enrolled student to visualize how individual metrics pushed the neural network "
            "toward 'At Risk' (positive values in red) or pulled it toward 'On Track' (negative values in blue)."
        )
        
        options_map = {f"{s.name} ({s.student_id})": s.student_id for s in students}
        selected_label = st.selectbox("Select Student to Profile", list(options_map.keys()), key="shap_student_selector")
        target_student_id = options_map[selected_label]
        
        # Get target student tabular info
        student_tab = next((b for b in background_tabular if b.student_id == target_student_id), None)
        
        if student_tab:
            student_df = pd.DataFrame([{
                "student_id": student_tab.student_id,
                "gpa": student_tab.gpa,
                "attendance_rate": student_tab.attendance_rate,
                "study_hours_weekly": student_tab.study_hours_weekly,
                "parental_involvement": student_tab.parental_involvement,
                "extracurricular_activities": student_tab.extracurricular_activities,
                "sleep_hours": student_tab.sleep_hours,
                "previous_grade": student_tab.previous_grade,
                "family_income": student_tab.family_income,
                "internet_access": student_tab.internet_access
            }])
            
            with st.spinner("Computing SHAP values (KernelExplainer)..."):
                try:
                    shap_results = get_shap_explanations(student_df, bg_df, artifacts_dir="src/edu_copilot/models/artifacts")
                    
                    # Prepare plotting data
                    chart_data = []
                    for feature_name, attribution in shap_results['attributions'].items():
                        # Extract the actual value from tabular
                        raw_feat_name = feature_name.replace("_encoded", "")
                        raw_val = getattr(student_tab, raw_feat_name, None)
                        
                        # Formatting float vs boolean vs string values
                        if isinstance(raw_val, float):
                            val_str = f"{raw_val:.2f}"
                            if "rate" in raw_feat_name:
                                val_str = f"{raw_val:.1%}"
                        else:
                            val_str = str(raw_val)
                            
                        chart_data.append({
                            "Feature Display": f"{raw_feat_name.replace('_', ' ').title()} ({val_str})",
                            "SHAP Value": attribution,
                            "Direction": "Increases Risk" if attribution > 0 else "Decreases Risk"
                        })
                        
                    df_shap = pd.DataFrame(chart_data)
                    
                    # Custom red/blue palette for waterfall attribution styling
                    color_map = {"Increases Risk": "#EF4444", "Decreases Risk": "#3B82F6"}
                    
                    fig = px.bar(
                        df_shap,
                        x="SHAP Value",
                        y="Feature Display",
                        orientation='h',
                        color="Direction",
                        color_discrete_map=color_map,
                        title=f"SHAP Force Diagnostic for Student {target_student_id}",
                        labels={"SHAP Value": "Attribution Impact", "Feature Display": "Feature & Ingested Value"}
                    )
                    # Add base value annotation
                    fig.add_vline(x=0, line_dash="dash", line_color="#475569")
                    fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=450)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Log baseline probability
                    st.write(f"📈 **Base Expected Probability (Reference Center)**: {shap_results['base_value']:.2%}")
                    st.write(f"🎯 **Model Predicted Probability**: {shap_results['prediction_value']:.2%}")
                except Exception as e:
                    st.error(f"Error executing KernelExplainer calculations: {e}")
                    
    # ----------------- TAB 3: GRAD-CAM STUB DETAILS -----------------
    with tab_gradcam:
        st.subheader("Grad-CAM Computer Vision Extension Hook")
        st.markdown(
            "> [!NOTE]\n"
            "> Grad-CAM is a visual explanation technique for Convolutional Neural Networks (CNNs). "
            "Because image and audio modalities are explicitly out-of-scope for the primary predictor, "
            "this hook remains a code stub for future upgrades."
        )
        
        # Load stub contents to display
        stub_file = "src/edu_copilot/xai/gradcam_stub.py"
        if os.path.exists(stub_file):
            with open(stub_file, "r", encoding="utf-8") as f:
                code_content = f.read()
            st.code(code_content, language="python")
        else:
            st.error("Grad-CAM stub script missing.")
