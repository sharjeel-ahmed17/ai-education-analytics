from pydantic import BaseModel, Field
from typing import Optional, List

class TabularStudentData(BaseModel):
    """
    Pydantic schema for structured student performance records.
    """
    student_id: str = Field(..., description="Unique identifier for the student")
    gpa: float = Field(..., ge=0.0, le=4.0, description="Grade Point Average (0.0 - 4.0)")
    attendance_rate: float = Field(..., ge=0.0, le=1.0, description="Attendance rate (0.0 - 1.0)")
    study_hours_weekly: float = Field(..., ge=0.0, description="Hours spent studying weekly")
    parental_involvement: str = Field(..., description="Level of parental involvement: 'Low', 'Medium', or 'High'")
    extracurricular_activities: bool = Field(..., description="Participation in extracurriculars")
    sleep_hours: float = Field(..., ge=0.0, description="Average sleep hours per night")
    previous_grade: float = Field(..., ge=0.0, le=100.0, description="Previous course/term grade (0 - 100)")
    family_income: str = Field(..., description="Family income tier: 'Low', 'Medium', or 'High'")
    internet_access: bool = Field(..., description="Whether the student has internet access at home")

class NoteData(BaseModel):
    """
    Pydantic schema for unstructured notes/feedback about a student.
    """
    student_id: str = Field(..., description="Associated student ID")
    content: str = Field(..., description="Feedback content text")
    author: Optional[str] = Field(None, description="Author of the note (e.g. Teacher, Counselor)")
    created_at: Optional[str] = Field(None, description="ISO timestamp of note creation")

class PDFData(BaseModel):
    """
    Pydantic schema for parsed PDF document contents.
    """
    student_id: str = Field(..., description="Associated student ID")
    filename: str = Field(..., description="Original PDF file name")
    content: str = Field(..., description="Extracted text content from the PDF")
