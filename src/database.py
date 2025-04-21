# Plik: src/database.py
import os
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, Text, ForeignKeyConstraint, DateTime
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.exc import SQLAlchemyError
import json
import logging
import time
import datetime

# --- Konfiguracja Połączenia z PostgreSQL ---
DB_USER = os.getenv("POSTGRES_USER", "wsp_user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "wsp_password")
DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "wsp_db")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')


# --- Modele Tabel (Schema) ---
class Event(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String, index=True, nullable=False, unique=True)
    description = Column(String, nullable=True)

    groups = relationship("Group", back_populates="event", cascade="all, delete-orphan")
    relationships = relationship("Relationship",
                               back_populates="event",
                               cascade="all, delete-orphan")
    results = relationship("AssignmentResult", back_populates="event", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Event(id={self.id}, name='{self.name}')>"

class Group(Base):
    __tablename__ = "groups"
    id = Column(Integer, primary_key=True, autoincrement=False)
    event_id = Column(Integer, ForeignKey("events.id", ondelete="CASCADE"), primary_key=True)
    size = Column(Integer, nullable=False)
    guest_names_json = Column(Text, nullable=True)

    event = relationship("Event", back_populates="groups")

    relationships1 = relationship("Relationship",
                                foreign_keys="[Relationship.event_id, Relationship.group1_id]",
                                back_populates="group1",
                                cascade="all, delete-orphan",
                                overlaps="relationships")
    relationships2 = relationship("Relationship",
                                foreign_keys="[Relationship.event_id, Relationship.group2_id]",
                                back_populates="group2",
                                cascade="all, delete-orphan",
                                overlaps="relationships,relationships1")

    @property
    def guest_names(self):
        if self.guest_names_json:
            try:
                return json.loads(self.guest_names_json)
            except json.JSONDecodeError:
                logging.error(f"Błąd dekodowania JSON dla grupy ID={self.id}, EventID={self.event_id}")
                return ["Błąd JSON"]
        return []

    def __repr__(self):
        return f"<Group(id={self.id}, event_id={self.event_id}, size={self.size})>"

class Relationship(Base):
    __tablename__ = "relationships"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    # --- ZMIANA TUTAJ: Dodano bezpośredni ForeignKey do events.id ---
    event_id = Column(Integer, ForeignKey("events.id", ondelete="CASCADE"), nullable=False, index=True)
    group1_id = Column(Integer, nullable=False)
    group2_id = Column(Integer, nullable=False)
    rel_type = Column(String, nullable=False)
    weight = Column(Float, nullable=False)

    event = relationship("Event", back_populates="relationships", overlaps="relationships1,relationships2")
    group1 = relationship("Group", foreign_keys=[event_id, group1_id], overlaps="event,relationships,relationships2")
    group2 = relationship("Group", foreign_keys=[event_id, group2_id], overlaps="event,group1,relationships,relationships1")

    __table_args__ = (
        ForeignKeyConstraint(['event_id', 'group1_id'], ['groups.event_id', 'groups.id'], ondelete="CASCADE"),
        ForeignKeyConstraint(['event_id', 'group2_id'], ['groups.event_id', 'groups.id'], ondelete="CASCADE"),
        {},
    )

    def __repr__(self):
        return f"<Relationship(id={self.id}, event={self.event_id}, g1={self.group1_id}, g2={self.group2_id}, type='{self.rel_type}')>"

class AssignmentResult(Base):
    __tablename__ = "assignment_results"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    event_id = Column(Integer, ForeignKey("events.id", ondelete="CASCADE"), nullable=False, index=True)
    run_timestamp = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    algorithm = Column(String, nullable=False)
    parameters_json = Column(Text, nullable=True)
    score = Column(Float, nullable=True)
    conflicts = Column(Integer, nullable=True)
    assignment_json = Column(Text, nullable=True)
    status = Column(String, nullable=False, default="success")

    event = relationship("Event", back_populates="results")

    def __repr__(self):
        return f"<AssignmentResult(id={self.id}, event_id={self.event_id}, algo='{self.algorithm}', score={self.score}, conflicts={self.conflicts})>"


# --- Funkcje Pomocnicze Bazy Danych ---
def wait_for_db(max_retries=20, delay=5):
    logging.info("Sprawdzanie dostępności bazy danych...")
    retries = 0
    while retries < max_retries:
        try:
            connection = engine.connect()
            connection.close()
            logging.info("Połączenie z bazą danych udane.")
            return True
        except SQLAlchemyError as e:
            retries += 1
            logging.warning(f"Nie można połączyć się z bazą danych (próba {retries}/{max_retries}). Błąd: {e}. Ponawiam za {delay}s...")
            time.sleep(delay)
    logging.error(f"Nie udało się połączyć z bazą danych po {max_retries} próbach.")
    return False

def create_db_and_tables():
    logging.info("Próba utworzenia tabel bazy danych...")
    if wait_for_db():
        try:
            Base.metadata.create_all(bind=engine)
            logging.info("Tabele bazy danych utworzone pomyślnie (lub już istniały).")
        except SQLAlchemyError as e:
            logging.exception(f"Błąd podczas tworzenia tabel: {e}")
        except Exception as e:
             logging.exception(f"Nieoczekiwany błąd podczas tworzenia tabel: {e}")
    else:
        logging.error("Nie można utworzyć tabel, ponieważ baza danych jest niedostępna.")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

if __name__ == "__main__":
    print("Uruchamianie skryptu database.py w celu utworzenia tabel...")
    logging.getLogger().setLevel(logging.INFO)
    create_db_and_tables()
    print("Zakończono próbę tworzenia tabel.")

# koniec pliku: src/database.py