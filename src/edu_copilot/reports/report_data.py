from pydantic import BaseModel, Field
from typing import List, Optional

class ReportData(BaseModel):
    """
    Unified report data model used to compile both PDF and DOCX documents.
    """
    student_id: str = Field(..., description="Unique student ID")
    student_name: str = Field(..., description="Full student name")
    grade_level: str = Field(..., description="Current grade level")
    predicted_risk: str = Field(..., description="Predicted risk status (e.g. 'At Risk', 'On Track')")
    risk_probability: float = Field(..., description="Calculated model probability (0.0 to 1.0)")
    confidence_score: float = Field(..., description="Calculated prediction certainty (0.0 to 1.0)")
    
    # Quantitative parameters
    gpa: float = Field(..., description="Student cumulative GPA")
    attendance_rate: float = Field(..., description="Attendance percentage")
    study_hours: float = Field(..., description="Weekly study hours")
    parental_involvement: str = Field(..., description="Parental involvement tier")
    previous_grade: float = Field(..., description="Previous grade percentage")
    
    # New Multimodal scores
    tabular_score: Optional[float] = Field(None, description="Tabular model risk score")
    cnn_legibility: Optional[float] = Field(None, description="CNN Legibility score")
    cnn_correctness: Optional[float] = Field(None, description="CNN Correctness score")
    cnn_completeness: Optional[float] = Field(None, description="CNN Completeness score")
    timeseries_score: Optional[float] = Field(None, description="Time series model risk score")
    audio_score: Optional[float] = Field(None, description="Audio fluency score")
    fused_score: Optional[float] = Field(None, description="Fused recommendation score")
    
    # Path to explainability artifacts
    gradcam_image_path: Optional[str] = Field(None, description="Path to Grad-CAM heatmap image")
    timeseries_attention_path: Optional[str] = Field(None, description="Path to time series attention chart")
    audio_attention_path: Optional[str] = Field(None, description="Path to audio attention chart")
    
    # LLM reasoning blocks
    summary: str = Field(..., description="Executive summary of the student's status")
    reasoning: str = Field(..., description="Core quantitative/qualitative analysis text")
    recommendations: List[str] = Field(..., description="Actionable checklist items")
    
    # Human-in-the-loop review markers
    reviewed_status: str = Field(..., description="HITL Review outcome ('Approved', 'Modified', etc.)")
    reviewer_notes: Optional[str] = Field(None, description="Reviewer feedback or notes")
