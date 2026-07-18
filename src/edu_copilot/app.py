import os
import sys
import shutil
import streamlit as st

# Dynamic Page Routing Generator
# Streamlit searches for a directory named 'pages' in the same folder as the script.
# This generator maps files from the required 'src/edu_copilot/ui/pages/' structure 
# into temporary wrappers inside 'src/edu_copilot/pages/' at startup.
current_dir = os.path.dirname(os.path.abspath(__file__))
pages_dir = os.path.join(current_dir, "pages")
ui_pages_dir = os.path.join(current_dir, "ui", "pages")

if os.path.exists(ui_pages_dir):
    os.makedirs(pages_dir, exist_ok=True)
    for filename in os.listdir(ui_pages_dir):
        if filename.endswith(".py") and not filename.startswith("__"):
            src_file = os.path.join(ui_pages_dir, filename)
            dst_file = os.path.join(pages_dir, filename)
            # Create a execution redirect wrapper script
            with open(dst_file, "w", encoding="utf-8") as f:
                f.write(
                    f'import os\n'
                    f'page_path = r"{src_file}"\n'
                    f'with open(page_path, encoding="utf-8") as f:\n'
                    f'    exec(f.read(), globals())\n'
                )

# Ensure project root is in the Python path
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))

from edu_copilot.config import settings
from edu_copilot.db.session import SessionLocal
from edu_copilot.db.models import Student, PredictionRecord, ReviewRecord, ReportRecord

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AI Co-Pilot - Education Analytics",
    page_icon="🎓",
    layout="wide"
)

# Custom premium styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #F8FAFC;
        padding: 1.25rem;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #3B82F6;
        text-align: center;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0.2rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        font-weight: 500;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allowed_ok=True)

st.markdown('<div class="main-title">AI Co-Pilot for Education Analytics</div>', unsafe_allowed_ok=True)
st.markdown('<div class="sub-title">A production-grade decision support platform combining deep predictive modeling with Human-in-the-Loop controls.</div>', unsafe_allowed_ok=True)

# Fetch current stats from DB
db = SessionLocal()
try:
    total_students = db.query(Student).count()
    total_preds = db.query(PredictionRecord).count()
    pending_reviews = db.query(ReviewRecord).filter(ReviewRecord.status == "Pending").count()
    approved_reports = db.query(ReportRecord).count()
except Exception as e:
    st.error(f"Failed to query database statistics: {e}")
    total_students, total_preds, pending_reviews, approved_reports = 0, 0, 0, 0
finally:
    db.close()

# Display KPI Metric Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f'<div class="metric-box">'
        f'<div class="metric-label">Enrolled Students</div>'
        f'<div class="metric-value">{total_students}</div>'
        f'</div>', 
        unsafe_allowed_ok=True
    )
with col2:
    st.markdown(
        f'<div class="metric-box">'
        f'<div class="metric-label">Neural Net Runs</div>'
        f'<div class="metric-value">{total_preds}</div>'
        f'</div>', 
        unsafe_allowed_ok=True
    )
with col3:
    st.markdown(
        f'<div class="metric-box">'
        f'<div class="metric-label">Pending Reviews</div>'
        f'<div class="metric-value">{pending_reviews}</div>'
        f'</div>', 
        unsafe_allowed_ok=True
    )
with col4:
    st.markdown(
        f'<div class="metric-box">'
        f'<div class="metric-label">Finalized Reports</div>'
        f'<div class="metric-value">{approved_reports}</div>'
        f'</div>', 
        unsafe_allowed_ok=True
    )

st.markdown("---")

# Main Page Information
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("Interactive Workspace Modules")
    st.markdown("""
    Select a module from the sidebar to interact with the system:
    
    *   **1. Upload Data**: Ingest structured student spreadsheets (CSV), unstructured counselor notes (TXT), or term report cards (PDF).
    *   **2. Run Predictions**: Perform binary risk inference using the exported **ONNX representation** of our trained Keras model.
    *   **3. HITL Review Queue**: Audit pending warnings. Modify AI recommendations, add professional notes, and save actions.
    *   **4. Explainability (XAI)**: View individual feature attributions via **SHAP explanations** and review global feature importances.
    *   **5. Export Reports**: Synthesize diagnostic outputs into branded **PDF** and **Word (DOCX)** documents.
    *   **6. Business Case**: Outline of the problem, monetization strategy, and go-to-market plan.
    """)

with col_right:
    st.subheader("System Configuration")
    
    # LLM Provider status indicators
    if settings.openai_api_key:
        st.success("OpenAI Integration: Active (GPT-4o-mini)")
    else:
        st.info("OpenAI Integration: Inactive (Using Mock LLM generator)")
        
    if settings.groq_api_key:
        st.success("Groq Integration: Active (Llama-3.1)")
    else:
        st.info("Groq Integration: Inactive (Using Mock LLM generator)")
        
    # Embeddings provider
    if settings.cohere_api_key:
        st.success("Cohere Integration: Active (Embeddings)")
    elif settings.openai_api_key:
        st.success("OpenAI Integration: Active (Embeddings)")
    else:
        st.info("Embeddings: Inactive (Using Mock local encoder)")
        
    # Database
    db_uri = settings.database_url or "sqlite:///edu_copilot.db"
    if db_uri.startswith("sqlite"):
        st.warning("Storage: Using local SQLite database (edu_copilot.db)")
    else:
        st.success("Storage: Connected to external PostgreSQL database")
