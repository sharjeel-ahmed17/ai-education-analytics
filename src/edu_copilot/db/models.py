from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Student(Base):
    """
    Core student model. Holds master details of the student.
    """
    __tablename__ = 'students'
    
    student_id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    grade_level = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    tabular_records = relationship("StudentTabular", back_populates="student", cascade="all, delete-orphan")
    predictions = relationship("PredictionRecord", back_populates="student", cascade="all, delete-orphan")
    reviews = relationship("ReviewRecord", back_populates="student", cascade="all, delete-orphan")
    reports = relationship("ReportRecord", back_populates="student", cascade="all, delete-orphan")


class StudentTabular(Base):
    """
    Structured performance features associated with each student.
    """
    __tablename__ = 'student_tabular_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String(50), ForeignKey('students.student_id', ondelete='CASCADE'), nullable=False)
    gpa = Column(Float, nullable=False)
    attendance_rate = Column(Float, nullable=False)
    study_hours_weekly = Column(Float, nullable=False)
    parental_involvement = Column(String(20), nullable=False)
    extracurricular_activities = Column(Boolean, nullable=False)
    sleep_hours = Column(Float, nullable=False)
    previous_grade = Column(Float, nullable=False)
    family_income = Column(String(20), nullable=False)
    internet_access = Column(Boolean, nullable=False)
    
    student = relationship("Student", back_populates="tabular_records")


class PredictionRecord(Base):
    """
    Output logs from the primary ANN model.
    """
    __tablename__ = 'prediction_records'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String(50), ForeignKey('students.student_id', ondelete='CASCADE'), nullable=False)
    predicted_prob = Column(Float, nullable=False)
    confidence_score = Column(Float, nullable=False)
    predicted_class = Column(Integer, nullable=False) # 0 = On Track, 1 = At Risk
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Sub-scores and Fused-scores
    tabular_score = Column(Float, nullable=True)
    cnn_legibility = Column(Float, nullable=True)
    cnn_correctness = Column(Float, nullable=True)
    cnn_completeness = Column(Float, nullable=True)
    timeseries_score = Column(Float, nullable=True)
    audio_score = Column(Float, nullable=True)
    fused_score = Column(Float, nullable=True)
    
    student = relationship("Student", back_populates="predictions")
    reviews = relationship("ReviewRecord", back_populates="prediction", cascade="all, delete-orphan")
    reports = relationship("ReportRecord", back_populates="prediction", cascade="all, delete-orphan")


class ReviewRecord(Base):
    """
    Human-In-The-Loop (HITL) analyst review logs.
    """
    __tablename__ = 'review_records'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(Integer, ForeignKey('prediction_records.id', ondelete='SET NULL'), nullable=True)
    student_id = Column(String(50), ForeignKey('students.student_id', ondelete='CASCADE'), nullable=False)
    status = Column(String(20), default="Pending", nullable=False) # 'Pending', 'Approved', 'Rejected', 'Modified'
    modified_recommendation = Column(Text, nullable=True)
    reviewer_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    student = relationship("Student", back_populates="reviews")
    prediction = relationship("PredictionRecord", back_populates="reviews")
    reports = relationship("ReportRecord", back_populates="review")


class ReportRecord(Base):
    """
    Final generated summaries and intervention recommendations.
    """
    __tablename__ = 'report_records'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String(50), ForeignKey('students.student_id', ondelete='CASCADE'), nullable=False)
    prediction_id = Column(Integer, ForeignKey('prediction_records.id', ondelete='CASCADE'), nullable=False)
    review_id = Column(Integer, ForeignKey('review_records.id', ondelete='SET NULL'), nullable=True)
    summary = Column(Text, nullable=False)
    reasoning = Column(Text, nullable=False)
    recommendations = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    student = relationship("Student", back_populates="reports")
    prediction = relationship("PredictionRecord", back_populates="reports")
    review = relationship("ReviewRecord", back_populates="reports")
