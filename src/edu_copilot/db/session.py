import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
from edu_copilot.config import settings

# Load connection URI from config, default to local SQLite database
db_uri = settings.database_url or "sqlite:///edu_copilot.db"

# Adjust engine parameters based on database type
connect_args = {}
if db_uri.startswith("sqlite"):
    connect_args["check_same_thread"] = False
elif db_uri.startswith("postgresql://"):
    # Map default postgresql:// to use psycopg (v3) instead of psycopg2
    db_uri = db_uri.replace("postgresql://", "postgresql+psycopg://", 1)

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
