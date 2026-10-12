from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import db_settings

engine = create_engine(db_settings.database_url_gen)
SessionLocal = sessionmaker(engine)