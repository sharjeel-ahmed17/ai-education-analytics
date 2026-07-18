import os
import tempfile
import pandas as pd
import streamlit as st
from edu_copilot.db.session import SessionLocal
from edu_copilot.db.models import Student, StudentTabular
from edu_copilot.data.loaders.tabular_loader import load_tabular_csv
from edu_copilot.data.loaders.pdf_loader import load_pdf
from edu_copilot.data.loaders.text_loader import load_text_note
from edu_copilot.data.preprocessing.text_preprocessing import split_text_into_chunks
from edu_copilot.vectorstore.qdrant_client import QdrantVectorStore
from edu_copilot.llm.embeddings import get_embeddings_model

st.set_page_config(layout="wide")

st.title("Data Ingestion & Modalities")
st.markdown("Upload quantitative spreadsheets and qualitative records to fuel the student early-warning metrics.")

# ----------------- SECTION 1: TABULAR CSV -----------------
st.header("1. Quantitative Student Spreadsheets (CSV)")
st.markdown("Upload a CSV file containing student demographics and academic markers.")

uploaded_csv = st.file_uploader("Select tabular CSV file", type=["csv"], key="uploader_csv")

if uploaded_csv:
    # Save stream to temporary file for loader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_csv.getvalue())
        tmp_path = tmp.name
        
    try:
        records = load_tabular_csv(tmp_path)
        db = SessionLocal()
        new_students_count = 0
        
        for record in records:
            # 1. Ensure master student record exists
            student = db.query(Student).filter(Student.student_id == record.student_id).first()
            if not student:
                # Default a readable name if not explicitly specified
                student = Student(
                    student_id=record.student_id,
                    name=f"Student {record.student_id}",
                    grade_level="10th Grade"
                )
                db.add(student)
                db.flush()
                
            # 2. Update or insert tabular indicators
            existing_tabular = db.query(StudentTabular).filter(
                StudentTabular.student_id == record.student_id
            ).first()
            
            if not existing_tabular:
                tabular_record = StudentTabular(
                    student_id=record.student_id,
                    gpa=record.gpa,
                    attendance_rate=record.attendance_rate,
                    study_hours_weekly=record.study_hours_weekly,
                    parental_involvement=record.parental_involvement,
                    extracurricular_activities=record.extracurricular_activities,
                    sleep_hours=record.sleep_hours,
                    previous_grade=record.previous_grade,
                    family_income=record.family_income,
                    internet_access=record.internet_access
                )
                db.add(tabular_record)
                new_students_count += 1
                
        db.commit()
        st.success(f"Processing complete! Ingested {len(records)} records. Added {new_students_count} new students to database.")
    except Exception as e:
        st.error(f"Failed to ingest CSV data: {e}")
    finally:
        db.close()
        try:
            os.remove(tmp_path)
        except Exception:
            pass

st.markdown("---")

# ----------------- SECTION 2: QUALITATIVE INGESTION -----------------
st.header("2. Qualitative Document Records (TXT & PDF)")
st.markdown("Select an enrolled student to attach qualitative teacher observations or transcript PDFs.")

db = SessionLocal()
try:
    student_records = db.query(Student).all()
finally:
    db.close()

if not student_records:
    st.info("No active student records found in database. Ingest tabular data first.")
else:
    # Build dropdown selections
    options_map = {f"{s.name} ({s.student_id})": s.student_id for s in student_records}
    selected_label = st.selectbox("Associate with Student Profile", list(options_map.keys()))
    target_student_id = options_map[selected_label]
    
    col_notes, col_pdfs = st.columns(2)
    
    # Text notes path
    with col_notes:
        st.subheader("Teacher / Counselor Notes (.TXT)")
        uploaded_notes = st.file_uploader("Upload adviser note", type=["txt"], key="uploader_notes")
        
        if uploaded_notes:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
                tmp.write(uploaded_notes.getvalue())
                tmp_path = tmp.name
                
            try:
                # Load content
                note_obj = load_text_note(target_student_id, tmp_path)
                # Split content into vector store documents
                chunks = split_text_into_chunks(note_obj.content, target_student_id, source_type="note")
                
                # Index in Qdrant
                embeddings = get_embeddings_model()
                vstore = QdrantVectorStore(embeddings)
                vstore.add_documents(chunks)
                
                st.success(f"Adviser notes ingested. Chunked into {len(chunks)} sections in vector store.")
            except Exception as e:
                st.error(f"Inference error indexing note: {e}")
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                    
    # PDF path
    with col_pdfs:
        st.subheader("Academic Transcript / Report card (.PDF)")
        uploaded_pdf = st.file_uploader("Upload school report card", type=["pdf"], key="uploader_pdf")
        
        if uploaded_pdf:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_pdf.getvalue())
                tmp_path = tmp.name
                
            try:
                # Parse PDF text
                pdf_obj = load_pdf(target_student_id, tmp_path)
                # Split content
                chunks = split_text_into_chunks(pdf_obj.content, target_student_id, source_type="pdf")
                
                # Index in Qdrant
                embeddings = get_embeddings_model()
                vstore = QdrantVectorStore(embeddings)
                vstore.add_documents(chunks)
                
                st.success(f"PDF transcript ingested. Chunked into {len(chunks)} sections in vector store.")
            except Exception as e:
                st.error(f"Inference error indexing PDF: {e}")
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

st.markdown("---")

# ----------------- SECTION 3: CURRENT DATABASE -----------------
st.subheader("Database Overview")
db = SessionLocal()
try:
    current_students = db.query(Student).all()
    if current_students:
        display_data = []
        for s in current_students:
            tabular = db.query(StudentTabular).filter(StudentTabular.student_id == s.student_id).first()
            display_data.append({
                "Student ID": s.student_id,
                "Name": s.name,
                "Grade Level": s.grade_level,
                "Has Tabular Stats": "Yes" if tabular else "No",
                "Registration Date": s.created_at.strftime("%Y-%m-%d %H:%M:%S")
            })
        st.dataframe(pd.DataFrame(display_data), use_container_width=True)
    else:
        st.info("The database is currently empty.")
finally:
    db.close()
