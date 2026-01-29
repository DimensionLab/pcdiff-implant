"""
SQLAlchemy database engine, session factory, and Base class.

Uses SQLite for single-user local deployment. The ORM layer is designed
to allow future migration to PostgreSQL if multi-user support is needed.
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from web_viewer.backend.config import settings

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},  # SQLite requires this
    echo=settings.sql_echo,
)


# Enable WAL mode and foreign keys for SQLite
@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency that yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables. Called on application startup."""
    from web_viewer.backend.models import (  # noqa: F401 – force model registration
        audit_log,
        color_profile,
        fit_metrics_result,
        point_cloud,
        project,
        scan,
    )

    Base.metadata.create_all(bind=engine)
