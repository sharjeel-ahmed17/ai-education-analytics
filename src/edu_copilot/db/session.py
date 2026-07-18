import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
from edu_copilot.config import settings

# Load connection URI from config, default to local SQLite database
db_uri = settings.database_url or "sqlite:///edu_copilot.db"

# Adjust engine parameters based on database type (SQLite requires check_same_thread=False)
connect_args = {}
if db_uri.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(db_uri, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Generator[Session, None, None]:
    """
    SQLAlchemy database session generator.
    Yields:
        Session: Database session instance.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
