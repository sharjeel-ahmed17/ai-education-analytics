import os
from pypdf import PdfReader
from edu_copilot.data.schemas import PDFData

def load_pdf(student_id: str, file_path: str) -> PDFData:
    """
    Parses a PDF file (e.g. report card, exam sheet) and extracts text using pypdf.
    
    Args:
        student_id (str): The ID of the student associated with the document.
        file_path (str): The filesystem path to the PDF.
        
    Returns:
        PDFData: Extracted text and metadata.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}")
        
    reader = PdfReader(file_path)
    text_content = []
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            text_content.append(text)
            
    full_text = "\n\n".join(text_content)
    
    return PDFData(
        student_id=student_id,
        filename=os.path.basename(file_path),
        content=full_text.strip()
    )
