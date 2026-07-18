import os
import sys

# Ensure src directory is on the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from edu_copilot.db.models import Base
from edu_copilot.db.session import engine

def setup_database() -> None:
    """
    Initializes database tables based on SQLAlchemy metadata.
    """
    print("Initializing database connection...")
    try:
        # Create all tables defined in models.py
        Base.metadata.create_all(bind=engine)
        print("Successfully created/verified all database tables.")
    except Exception as e:
        print(f"Error occurred during database initialization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    setup_database()
