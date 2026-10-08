from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config import db_settings

engine = create_engine(
    db_settings.database_url_gen,
    echo=False,           # pon True solo cuando quieras ver el SQL
    pool_pre_ping=True,   # detecta conexiones muertas (útil al migrar a Supabase)
)

SessionLocal = sessionmaker(bind=engine)


def get_session() -> Iterator[Session]:
    """Una sesión por unidad de trabajo. En la Semana 3, FastAPI la usará con Depends()."""
    with SessionLocal() as session:
        yield session