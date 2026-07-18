from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional
from edu_copilot.db.models import ReviewRecord

def create_pending_review(
    db: Session, 
    student_id: str, 
    prediction_id: int
) -> ReviewRecord:
    """
    Creates a new pending review record in the database.
    """
    review = ReviewRecord(
        student_id=student_id,
        prediction_id=prediction_id,
        status="Pending"
    )
    db.add(review)
    db.commit()
    db.refresh(review)
    return review

def approve_recommendation(
    db: Session, 
    review_id: int, 
    notes: Optional[str] = None
) -> ReviewRecord:
    """
    Approves the AI prediction without changes.
    """
    review = db.query(ReviewRecord).filter(ReviewRecord.id == review_id).first()
    if not review:
        raise ValueError(f"Review record with ID {review_id} not found.")
        
    review.status = "Approved"
    review.reviewer_notes = notes
    review.created_at = datetime.utcnow()
    
    db.commit()
    db.refresh(review)
    return review

def reject_recommendation(
    db: Session, 
    review_id: int, 
    notes: Optional[str] = None
) -> ReviewRecord:
    """
    Rejects the AI prediction outright.
    """
    review = db.query(ReviewRecord).filter(ReviewRecord.id == review_id).first()
    if not review:
        raise ValueError(f"Review record with ID {review_id} not found.")
        
    review.status = "Rejected"
    review.reviewer_notes = notes
    review.created_at = datetime.utcnow()
    
    db.commit()
    db.refresh(review)
    return review

def modify_recommendation(
    db: Session, 
    review_id: int, 
    modified_text: str, 
    notes: Optional[str] = None
) -> ReviewRecord:
    """
    Modifies the recommended actions before finalizing.
    """
    review = db.query(ReviewRecord).filter(ReviewRecord.id == review_id).first()
    if not review:
        raise ValueError(f"Review record with ID {review_id} not found.")
        
    review.status = "Modified"
    review.modified_recommendation = modified_text
    review.reviewer_notes = notes
    review.created_at = datetime.utcnow()
    
    db.commit()
    db.refresh(review)
    return review
