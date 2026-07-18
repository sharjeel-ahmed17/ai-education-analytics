import os
from typing import List, Any, Optional
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from edu_copilot.config import settings

class MockChatModel(SimpleChatModel):
    """
    Simulated chat model to allow end-to-end execution without external API keys.
    Inspects prompts and outputs contextual academic reports, summaries, or explanations.
    """
    
    @property
    def _llm_type(self) -> str:
        return "mock_chat_model"

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> str:
        # Concatenate message texts for inspect-based conditional replies
        prompt_context = "\n".join([
            m.content if isinstance(m.content, str) else str(m.content) 
            for m in messages
        ]).lower()
        
        # Summarize request detection
        if "summarize" in prompt_context or "document" in prompt_context:
            return (
                "### Document Summarization Analysis\n"
                "- **Key Strengths**: The student demonstrates high active participation during peer-to-peer exercises and expresses interest in creative tasks.\n"
                "- **Identified Pitfalls**: Academic results show a negative trajectory in mathematics and chemistry. The student has skipped multiple lab sessions.\n"
                "- **Behavioral Indicators**: Teachers note class distraction in afternoon slots and a pattern of missing assignment deadlines."
            )
            
        # Explanations/Reasoning request detection
        elif "explain" in prompt_context or "risk" in prompt_context or "shap" in prompt_context:
            return (
                "### AI Explanation & Student Risk Reasoning\n"
                "The student's predicted academic risk (At Risk status) is triggered primarily by the intersection of three factors:\n"
                "1. **Low Attendance Rate**: Shuffled permutation test indicates a critical boundary breach, dropping engagement.\n"
                "2. **Underlying GPA**: A baseline GPA below 2.5 reduces the resilience against late-term grade drops.\n"
                "3. **Weekly Study Routine**: Low study hours (less than 4 hours weekly) prevent recovery in quantitative subjects.\n\n"
                "Qualitative evidence from counselor notes corroborates these points, stating that the student feels overwhelmed and is skipping laboratory work."
            )
            
        # Report generation request detection
        else:
            return (
                "### Student Intervention & Academic Report\n\n"
                "**Student Overview**\n"
                "The student has shown high vulnerability to academic failure or dropouts. Immediate guidance intervention is advised.\n\n"
                "**Predictive Insights**\n"
                "- **Model Risk Calculation**: 90% Probability of At-Risk Status\n"
                "- **XAI Attributions**: Attendance and GPA are the core risk drivers. The relative SHAP contribution is high.\n\n"
                "**Actionable Intervention Items**\n"
                "- **Action 1**: Mandatory peer tutoring sessions for chemistry twice a week.\n"
                "- **Action 2**: Create a daily academic dashboard to check off lab submissions and assignments.\n"
                "- **Action 3**: Weekly checking session with advisor to keep track of attendance milestones."
            )

def get_llm(provider: str = "openai") -> SimpleChatModel:
    """
    Returns the appropriate LLM chat client based on env settings.
    Falls back to MockChatModel if credentials are not configured.
    """
    if provider == "groq" and settings.groq_api_key:
        print("Using Groq Chat Client...")
        return ChatGroq(
            groq_api_key=settings.groq_api_key, 
            model="llama-3.1-70b-versatile"
        )
    elif settings.openai_api_key:
        print("Using OpenAI Chat Client...")
        return ChatOpenAI(
            openai_api_key=settings.openai_api_key, 
            model="gpt-4o-mini"
        )
    else:
        print("No LLM provider keys detected. Falling back to MockChatModel...")
        return MockChatModel()
