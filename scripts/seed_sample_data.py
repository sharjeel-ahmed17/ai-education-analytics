import os
import sys
import random
import pandas as pd
from reportlab.pdfgen import canvas

# Ensure src directory is on python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from edu_copilot.db.session import SessionLocal, engine
from edu_copilot.db.models import Base, Student, StudentTabular

def create_sample_files(sample_dir: str) -> None:
    """
    Creates dummy PDF and TXT files for ingestion testing.
    """
    os.makedirs(sample_dir, exist_ok=True)
    
    # 1. Create a dummy PDF report card
    pdf_path = os.path.join(sample_dir, "report_card_101.pdf")
    print(f"Creating sample PDF at {pdf_path}...")
    c = canvas.Canvas(pdf_path)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "Oakwood Academy Academic Report Card")
    c.setFont("Helvetica", 12)
    c.drawString(100, 720, "Student Name: John Doe")
    c.drawString(100, 700, "Student ID: S101")
    c.drawString(100, 680, "Term: Spring 2026")
    
    c.drawString(100, 640, "Grades:")
    c.drawString(120, 620, "Mathematics: C- (62%)")
    c.drawString(120, 600, "English Literature: B (84%)")
    c.drawString(120, 580, "Chemistry: D (58%)")
    c.drawString(120, 560, "History: C (71%)")
    
    c.drawString(100, 520, "Teacher Comments:")
    c.drawString(100, 500, "John struggles with math homework completion and exam preparation.")
    c.drawString(100, 485, "He has skipped 4 chemistry labs. Attendance must improve.")
    c.save()

    # 2. Create teacher feedback notes
    note_path = os.path.join(sample_dir, "teacher_notes_101.txt")
    print(f"Creating sample notes at {note_path}...")
    with open(note_path, "w", encoding="utf-8") as f:
        f.write("Teacher Note - April 12, 2026\n")
        f.write("Student: John Doe (S101)\n")
        f.write("Author: Mrs. Gable (Adviser)\n\n")
        f.write("John shows high potential in creative writing and group activities. ")
        f.write("However, his recent performance in sciences is concerning. ")
        f.write("He appears distracted in afternoon classes and has missed several assignments. ")
        f.write("Spoke with him yesterday: he reports feeling overwhelmed by chemistry. ")
        f.write("Recommended peer tutoring. Needs close academic monitoring.")


def generate_synthetic_data(num_students: int = 100) -> pd.DataFrame:
    """
    Generates a DataFrame of synthetic student performance indicators.
    """
    first_names = ["Emma", "Liam", "Olivia", "Noah", "Ava", "Oliver", "Sophia", "Elijah", "Isabella", "James"]
    last_names = ["Smith", "Jones", "Taylor", "Brown", "Williams", "Wilson", "Johnson", "Davies", "Patel", "Wright"]
    
    records = []
    
    for i in range(num_students):
        student_id = f"S{101 + i}"
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        grade_level = random.choice(["9th Grade", "10th Grade", "11th Grade", "12th Grade"])
        
        # Correlated generation to make model training realistic
        # Base probability of being at risk
        gpa = round(random.uniform(1.8, 4.0), 2)
        attendance = round(random.uniform(0.75, 1.0), 2)
        
        # If student has low attendance or gpa, study hours are likely lower
        if gpa < 2.5 or attendance < 0.85:
            study_hours = round(random.uniform(1.0, 6.0), 1)
            previous_grade = round(random.uniform(50.0, 75.0), 1)
            parental = random.choices(["Low", "Medium", "High"], weights=[0.6, 0.3, 0.1])[0]
        else:
            study_hours = round(random.uniform(5.0, 20.0), 1)
            previous_grade = round(random.uniform(70.0, 99.0), 1)
            parental = random.choices(["Low", "Medium", "High"], weights=[0.1, 0.4, 0.5])[0]
            
        sleep_hours = round(random.uniform(5.0, 9.0), 1)
        extracurricular = random.choice([True, False])
        income = random.choices(["Low", "Medium", "High"], weights=[0.3, 0.5, 0.2])[0]
        internet = random.choices([True, False], weights=[0.85, 0.15])[0]
        
        # Define logic-based risk target for model training
        # At risk if:
        # GPA < 2.3 OR attendance < 0.80 OR (GPA < 2.7 AND study hours < 3.0)
        at_risk = 0
        if gpa < 2.3 or attendance < 0.82 or (gpa < 2.7 and study_hours < 4.0):
            at_risk = 1
            
        records.append({
            "student_id": student_id,
            "name": name,
            "grade_level": grade_level,
            "gpa": gpa,
            "attendance_rate": attendance,
            "study_hours_weekly": study_hours,
            "parental_involvement": parental,
            "extracurricular_activities": extracurricular,
            "sleep_hours": sleep_hours,
            "previous_grade": previous_grade,
            "family_income": income,
            "internet_access": internet,
            "at_risk": at_risk
        })
        
    return pd.DataFrame(records)


def seed_database(df: pd.DataFrame) -> None:
    """
    Seeds the relational DB with students and their tabular performance records.
    """
    db = SessionLocal()
    print("Seeding database records...")
    
    try:
        # Clear existing records first to avoid unique constraints in repeat seeds
        db.query(StudentTabular).delete()
        db.query(Student).delete()
        db.commit()
        
        # Seed 15 representative students into DB
        seed_subset = df.head(15).copy()
        
        # Force John Doe as S101 to match dummy files
        seed_subset.loc[0, "student_id"] = "S101"
        seed_subset.loc[0, "name"] = "John Doe"
        seed_subset.loc[0, "gpa"] = 2.2
        seed_subset.loc[0, "attendance_rate"] = 0.80
        seed_subset.loc[0, "study_hours_weekly"] = 3.0
        seed_subset.loc[0, "previous_grade"] = 61.5
        seed_subset.loc[0, "parental_involvement"] = "Low"
        seed_subset.loc[0, "at_risk"] = 1
        
        for _, row in seed_subset.iterrows():
            # Add student
            student = Student(
                student_id=row['student_id'],
                name=row['name'],
                grade_level=row['grade_level']
            )
            db.add(student)
            db.flush() # populate relationships properly
            
            # Add tabular performance records
            tabular = StudentTabular(
                student_id=row['student_id'],
                gpa=row['gpa'],
                attendance_rate=row['attendance_rate'],
                study_hours_weekly=row['study_hours_weekly'],
                parental_involvement=row['parental_involvement'],
                extracurricular_activities=bool(row['extracurricular_activities']),
                sleep_hours=row['sleep_hours'],
                previous_grade=row['previous_grade'],
                family_income=row['family_income'],
                internet_access=bool(row['internet_access'])
            )
            db.add(tabular)
            
        db.commit()
        print(f"Successfully seeded {len(seed_subset)} students and their metrics into the database.")
    except Exception as e:
        db.rollback()
        print(f"Error during seeding: {e}")
    finally:
        db.close()


def main() -> None:
    sample_dir = os.path.join("data", "sample")
    os.makedirs(sample_dir, exist_ok=True)
    
    # 1. Create file artifacts
    create_sample_files(sample_dir)
    
    # 2. Generate training CSV
    df = generate_synthetic_data(100)
    csv_path = os.path.join(sample_dir, "student_records.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved synthetic training CSV with 100 rows to {csv_path}")
    
    # 3. Seed to DB
    seed_database(df)
    
    print("All sample seeding operations completed successfully.")

if __name__ == "__main__":
    main()
