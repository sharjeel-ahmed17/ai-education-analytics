import os
import sys
import pandas as pd

# Ensure src directory is on the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from edu_copilot.db.models import Base, Student, StudentTabular
from edu_copilot.db.session import engine, SessionLocal

def reset_database() -> None:
    """
    Deletes the existing database (if any) and creates tables based on current schema.
    Then populates students from data/sample/student_records.csv.
    """
    db_file = "edu_copilot.db"
    if os.path.exists(db_file):
        print(f"Removing existing database file '{db_file}'...")
        try:
            os.remove(db_file)
        except Exception as e:
            print(f"Warning: Could not remove database file: {e}")

    print("Dropping existing tables if any...")
    try:
        Base.metadata.drop_all(bind=engine)
    except Exception as e:
        print(f"Warning during drop_all: {e}")

    print("Initializing database schema...")
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")

    # Load and ingest default students
    csv_path = "data/sample/student_records.csv"
    if os.path.exists(csv_path):
        print(f"Ingesting default student data from '{csv_path}'...")
        df = pd.read_csv(csv_path)
        db = SessionLocal()
        try:
            for _, row in df.iterrows():
                student_id = str(row['student_id'])
                # Create Student profile
                student = Student(
                    student_id=student_id,
                    name=str(row['name']),
                    grade_level=str(row['grade_level'])
                )
                db.add(student)
                db.flush()

                # Create Student Tabular details
                tabular = StudentTabular(
                    student_id=student_id,
                    gpa=float(row['gpa']),
                    attendance_rate=float(row['attendance_rate']),
                    study_hours_weekly=float(row['study_hours_weekly']),
                    parental_involvement=str(row['parental_involvement']),
                    extracurricular_activities=bool(row['extracurricular_activities']),
                    sleep_hours=float(row['sleep_hours']),
                    previous_grade=float(row['previous_grade']),
                    family_income=str(row['family_income']),
                    internet_access=bool(row['internet_access'])
                )
                db.add(tabular)
            db.commit()
            print(f"Ingested {len(df)} student profiles and tabular records.")
        except Exception as e:
            db.rollback()
            print(f"Failed to ingest student profiles: {e}")
        finally:
            db.close()
    else:
        print(f"Warning: Default student records CSV not found at '{csv_path}'. Database is empty.")

if __name__ == "__main__":
    reset_database()
