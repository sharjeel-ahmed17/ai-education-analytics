import os
import sys
import pytest
import tempfile
import shutil
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from edu_copilot.db.models import Base

# Ensure src directory is in Python path for test scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

@pytest.fixture(scope="session")
def db_engine():
    """
    Initializes an in-memory SQLite database for the test suite duration.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def db_session(db_engine):
    """
    Provides a transactional database session that rolls back changes after each test.
    """
    connection = db_engine.connect()
    transaction = connection.begin()
    
    Session = sessionmaker(bind=connection)
    session = Session()
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture
def temp_dir():
    """
    Provides a temporary directory that is automatically cleaned up after the test.
    """
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)
