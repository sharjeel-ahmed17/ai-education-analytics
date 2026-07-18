from sqlalchemy.orm import Session
from typing import List, Optional
from edu_copilot.db.models import ReviewRecord, Student, PredictionRecord

def get_pending_reviews(db: Session) -> List[ReviewRecord]:
    """
    Retrieves all human review records that are in 'Pending' status.
    
    Args:
        db (Session): Relational database session.
        
    Returns:
        List[ReviewRecord]: List of pending review records.
    """
    return (
        db.query(ReviewRecord)
        .filter(ReviewRecord.status == "Pending")
        .order_by(ReviewRecord.created_at.desc())
        .all()
    )

def get_review_by_id(db: Session, review_id: int) -> Optional[ReviewRecord]:
    """
    Retrieves a single review record by its primary key ID.
    """
    return db.query(ReviewRecord).filter(ReviewRecord.id == review_id).first()

def get_reviews_by_student(db: Session, student_id: str) -> List[ReviewRecord]:
    """
    Retrieves all review records associated with a student.
    """
    return (
        db.query(ReviewRecord)
        .filter(ReviewRecord.student_id == student_id)
        .order_by(ReviewRecord.created_at.desc())
        .all()
    )

def get_all_reviews(db: Session) -> List[ReviewRecord]:
    """
    Retrieves all reviews, ordered by creation time.
    """
    return (
        db.query(ReviewRecord)
        .order_by(ReviewRecord.created_at.desc())
        .all()
    )
