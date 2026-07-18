import os
from typing import Optional
from datetime import datetime
from edu_copilot.data.schemas import NoteData

def load_text_note(student_id: str, file_path: str, author: Optional[str] = None) -> NoteData:
    """
    Loads unstructured notes/feedback from a text file.
    
    Args:
        student_id (str): The ID of the student.
        file_path (str): Path to the text file.
        author (Optional[str]): Author of the note (e.g. Teacher, Advisor).
        
    Returns:
        NoteData: Loaded note object with content and metadata.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Text file not found at: {file_path}")
        
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    return NoteData(
        student_id=student_id,
        content=content.strip(),
        author=author or "Educator",
        created_at=datetime.utcnow().isoformat()
    )
