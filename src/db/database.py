from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from config import settings

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,   # drops stale connections before using them
    pool_size=5,
    max_overflow=10,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


# FastAPI dependency — yields a DB session and closes it after the request
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
