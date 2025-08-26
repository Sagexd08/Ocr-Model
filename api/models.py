from sqlalchemy import Column, Integer, String, JSON, ForeignKey
from .database import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, unique=True, index=True)
    content = Column(JSON)

class CorrectedData(Base):
    __tablename__ = "corrected_data"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    corrected_content = Column(JSON)
