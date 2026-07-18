from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def split_text_into_chunks(
    text: str, 
    student_id: str, 
    source_type: str, 
    chunk_size: int = 500, 
    chunk_overlap: int = 50
) -> List[Document]:
    """
    Splits extracted text from teacher notes or PDFs into chunks for vector indexing.
    
    Args:
        text (str): The raw text to chunk.
        student_id (str): Associated student ID for metadata scoping.
        source_type (str): Type of source document (e.g. 'note', 'pdf').
        chunk_size (int): Character size of each chunk.
        chunk_overlap (int): Overlap size between adjacent chunks.
        
    Returns:
        List[Document]: List of LangChain Document objects with student metadata.
    """
    if not text:
        return []
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Generate splits
    chunks = splitter.split_text(text)
    
    documents = [
        Document(
            page_content=chunk,
            metadata={
                "student_id": student_id,
                "source_type": source_type,
                "chunk_index": i
            }
        )
        for i, chunk in enumerate(chunks)
    ]
    
    return documents
