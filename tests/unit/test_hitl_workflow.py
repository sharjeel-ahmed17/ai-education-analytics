import pytest
from sqlalchemy.orm import Session
from edu_copilot.db.models import Student, PredictionRecord, ReviewRecord
from edu_copilot.hitl.workflow import (
    create_pending_review, 
    approve_recommendation, 
    reject_recommendation, 
    modify_recommendation
)

def test_hitl_workflow_transitions(db_session: Session) -> None:
    """
    Verifies state transitions (Pending -> Approved, Rejected, Modified)
    in the database for the Human-in-the-Loop auditing logs.
    """
    # 1. Arrange: Setup student and prediction logs
    student = Student(
        student_id="T101",
        name="Tim Test",
        grade_level="11th Grade"
    )
    db_session.add(student)
    db_session.flush()
    
    prediction = PredictionRecord(
        student_id="T101",
        predicted_prob=0.75,
        confidence_score=0.50,
        predicted_class=1
    )
    db_session.add(prediction)
    db_session.flush()
    
    # 2. Act & Assert: Create Pending Record
    review = create_pending_review(db_session, student_id="T101", prediction_id=prediction.id)
    assert review.id is not None
    assert review.status == "Pending"
    
    # Test Approval
    approved_rec = approve_recommendation(db_session, review_id=review.id, notes="Valid alert")
    assert approved_rec.status == "Approved"
    assert approved_rec.reviewer_notes == "Valid alert"
    
    # Reset back to pending for next test
    review.status = "Pending"
    db_session.commit()
    
    # Test Rejection
    rejected_rec = reject_recommendation(db_session, review_id=review.id, notes="False positive")
    assert rejected_rec.status == "Rejected"
    assert rejected_rec.reviewer_notes == "False positive"
    
    # Reset back to pending for next test
    review.status = "Pending"
    db_session.commit()
    
    # Test Modification
    modified_rec = modify_recommendation(
        db_session, 
        review_id=review.id, 
        modified_text="Custom plan text", 
        notes="Minor alterations"
    )
    assert modified_rec.status == "Modified"
    assert modified_rec.modified_recommendation == "Custom plan text"
    assert modified_rec.reviewer_notes == "Minor alterations"
