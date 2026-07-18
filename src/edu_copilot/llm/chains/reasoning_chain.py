from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from edu_copilot.llm.prompts.report_prompts import REASONING_PROMPT

def get_reasoning_chain(llm) -> Runnable:
    """
    Creates a LangChain runnable to generate explanations based on tabular prediction
    metrics and SHAP feature attributions.
    
    Args:
        llm: A LangChain ChatModel instance.
        
    Returns:
        Runnable: Reasoning/Explanation pipeline.
    """
    return REASONING_PROMPT | llm | StrOutputParser()
