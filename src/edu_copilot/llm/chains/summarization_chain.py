from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from edu_copilot.llm.prompts.report_prompts import SUMMARIZATION_PROMPT

def get_summarization_chain(llm) -> Runnable:
    """
    Creates a LangChain runnable to summarize teacher notes and parsed PDFs.
    
    Args:
        llm: A LangChain ChatModel instance.
        
    Returns:
        Runnable: Summarization pipeline.
    """
    return SUMMARIZATION_PROMPT | llm | StrOutputParser()
