from langchain_core.prompts import ChatPromptTemplate

# Ingestion text summarizer prompt
SUMMARIZATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system", 
        "You are an educational psychologist and advisor. Your task is to analyze unstructured "
        "advisor notes, report cards, or transcripts, and extract key insights. Be concise, professional, "
        "and focus on academic behaviors, attendance concerns, and emotional/family support indicators."
    ),
    (
        "user",
        "Please summarize the following document records for student ID {student_id}:\n\n"
        "--- START OF RECORD ---\n"
        "{document_text}\n"
        "--- END OF RECORD ---\n\n"
        "Provide your analysis with the following headings:\n"
        "- **Academic Strengths**\n"
        "- **Identified Triggers / Challenges**\n"
        "- **Key Behavioral/Attendance Observations**"
    )
])

# Prediction explainer prompt (narrates ANN outputs)
REASONING_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an Explainable AI (XAI) narrator specializing in educational analytics. "
        "Your task is to explain a neural network model's risk prediction to a teacher. "
        "Explain the features that contributed to the model's decision, making reference to "
        "the provided SHAP feature attributions (positive SHAP = pushes model toward 'At Risk', "
        "negative SHAP = pulls model toward 'On Track'). Ensure the tone is supportive and diagnostic. "
        "Crucial: Highlight that the LLM did not make this prediction; it is explaining the primary ANN engine."
    ),
    (
        "user",
        "Explain the Neural Network's prediction for Student ID {student_id}.\n\n"
        "**Prediction Outcome**:\n"
        "- Classification: {predicted_class_name} (Threshold: 0.5)\n"
        "- Calculated Probability of Risk: {predicted_prob:.2%}\n"
        "- Model Prediction Confidence: {confidence_score:.2%}\n\n"
        "**Tabular Student Metrics**:\n"
        "- Cumulative GPA: {gpa}\n"
        "- Attendance Rate: {attendance_rate:.1%}\n"
        "- Weekly Study Hours: {study_hours_weekly} hrs\n"
        "- Previous Term Grade: {previous_grade}%\n"
        "- Parental Involvement Level: {parental_involvement}\n"
        "- Participates in Extracurriculars: {extracurricular_activities}\n"
        "- Average Sleep Hours: {sleep_hours} hrs/night\n"
        "- Family Income Bracket: {family_income}\n"
        "- Internet Access at Home: {internet_access}\n\n"
        "**SHAP Attribution Values (Impact on prediction)**:\n"
        "{shap_attributions}\n\n"
        "Provide a clear explanation of how these values support the ANN's prediction."
    )
])

# Report synthesizer prompt (Combines quantitative and qualitative data)
REPORT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a Senior Academic Counselor. Synthesize the quantitative AI predictions, "
        "explainable AI diagnostics, and qualitative document summaries into a final Academic "
        "Intervention Report. Focus on producing clear, actionable plans."
    ),
    (
        "user",
        "Create a comprehensive Student Academic Intervention Report for student '{student_name}' (ID: {student_id}).\n\n"
        "**AI Prediction & Confidence**:\n"
        "- Risk Classification: {predicted_class_name}\n"
        "- Predicted Risk Probability: {predicted_prob:.2%}\n"
        "- Confidence Score: {confidence_score:.2%}\n\n"
        "**XAI Diagnostic Summary**:\n"
        "{xai_explanation}\n\n"
        "**Qualitative Context (From Notes/PDFs)**:\n"
        "{qualitative_summary}\n\n"
        "Structure the report with the following three sections:\n"
        "1. EXECUTIVE SUMMARY (Brief overview of the student's status)\n"
        "2. CORE TRIGGERS & DIAGNOSTICS (Breakdown of quantitative metrics and qualitative notes)\n"
        "3. ACTIONABLE INTERVENTION PLAN (Numbered, concrete steps for school staff and the student to implement)"
    )
])
