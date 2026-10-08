from sqlalchemy import func, text, Column, Table, ForeignKey, DateTime, Uuid
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column,  relationship
from pgvector.sqlalchemy import Vector
from datetime import datetime
import uuid

class Base(DeclarativeBase):
    pass


class Patient(Base):
    __tablename__ = 'patients'
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(unique=True)
    full_name: Mapped[str] = mapped_column()
    consent_given: Mapped[bool] = mapped_column(server_default=text('false'))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.current_timestamp())
    enrollments: Mapped[list['Enrollment']] = relationship(back_populates='patient')
    
    
class Condition(Base):
    __tablename__ = 'conditions'
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column()
    description: Mapped[str] = mapped_column()
    is_active: Mapped[bool] = mapped_column(server_default=text('true'))
    enrollments: Mapped[list['Enrollment']] = relationship(back_populates='condition')
    content_templates: Mapped[list['ContentTemplate']] = relationship(back_populates='condition', order_by='ContentTemplate.day_offset')
    
    
class Enrollment(Base):
    __tablename__ = 'enrollments'
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('patients.id'))
    condition_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('conditions.id')) 
    enrolled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.current_timestamp())
    status: Mapped[str] = mapped_column()
    patient: Mapped['Patient'] = relationship(back_populates='enrollments')
    condition: Mapped['Condition'] = relationship(back_populates='enrollments')
    sends: Mapped[list['Send']] = relationship(back_populates='enrollment')
    
    
class ContentTemplate(Base):
    __tablename__ = 'content_templates'
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    condition_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('conditions.id'))
    day_offset: Mapped[int] = mapped_column()
    subject: Mapped[str] = mapped_column()
    body: Mapped[str] = mapped_column()
    approved_by_physician: Mapped[bool] = mapped_column(server_default=text('false'))
    sends: Mapped[list['Send']] = relationship(back_populates='content_template')
    chunks: Mapped[list['Chunk']] = relationship(back_populates='content_templates', secondary='content_template_chunks')
    condition: Mapped['Condition'] = relationship(back_populates='content_templates')
    
    
class Send(Base):
    __tablename__ = 'sends'
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    enrollment_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('enrollments.id')) 
    content_template_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('content_templates.id')) 
    sent_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.current_timestamp())
    status: Mapped[str] = mapped_column()
    content_template: Mapped['ContentTemplate'] = relationship(back_populates='sends')
    enrollment: Mapped['Enrollment'] = relationship(back_populates='sends')
    engagements: Mapped[list['Engagement']] = relationship(back_populates='send')
   
    
class Engagement(Base):
    __tablename__ = 'engagement'
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    send_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('sends.id')) 
    opened_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    clicked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    send: Mapped['Send'] = relationship(back_populates='engagements')
    
    
class Book(Base):
    __tablename__ = 'books'
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(unique=True)
    author: Mapped[str] = mapped_column()
    chunks: Mapped[list['Chunk']] = relationship(back_populates='book')
    
    
class Chunk(Base):
    __tablename__ = 'chunks'
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    book_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('books.id')) 
    content: Mapped[str] = mapped_column()
    page_number: Mapped[int] = mapped_column()
    embedding: Mapped[list[float] | None] = mapped_column(Vector(384), nullable=True)
    book: Mapped['Book'] = relationship(back_populates='chunks')
    content_templates: Mapped[list['ContentTemplate']] = relationship(back_populates='chunks', secondary='content_template_chunks')
    topics: Mapped[list['Topic']] = relationship(back_populates='chunks', secondary='chunk_topics')


content_template_chunks = Table(
    'content_template_chunks',
    Base.metadata,
    Column('content_template_id', Uuid, ForeignKey('content_templates.id'), primary_key=True), 
    Column('chunk_id', Uuid, ForeignKey('chunks.id'), primary_key=True) 
)


class Topic(Base):
   __tablename__ = 'topics'
   id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4) 
   name: Mapped[str] = mapped_column()
   chunks: Mapped[list['Chunk']] = relationship(back_populates='topics', secondary='chunk_topics')
   
   
chunk_topics = Table(
    'chunk_topics',
    Base.metadata,
    Column('chunk_id', Uuid, ForeignKey('chunks.id'), primary_key=True), 
    Column('topic_id', Uuid, ForeignKey('topics.id'), primary_key=True) 
)
    
