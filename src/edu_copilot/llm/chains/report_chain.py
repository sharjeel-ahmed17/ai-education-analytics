from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from edu_copilot.llm.prompts.report_prompts import REPORT_PROMPT

def get_report_chain(llm) -> Runnable:
    """
    Creates a LangChain runnable to synthesize the final student intervention report.
    
    Args:
        llm: A LangChain ChatModel instance.
        
    Returns:
        Runnable: Report synthesis pipeline.
    """
    return REPORT_PROMPT | llm | StrOutputParser()
