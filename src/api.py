# Plik: src/api.py

from fastapi import FastAPI, HTTPException, Body, Depends
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Optional, List
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
import datetime
import logging
import configparser

try:
    from .solver import run_solve_process, load_config
    from .database import SessionLocal, Event, Group, Relationship, AssignmentResult, get_db, create_db_and_tables
    from .seed_db import seed_groups_and_relationships
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    logging.error(f"Błąd importu w api.py: {e}. Używanie atrap funkcji i modeli.")
    IMPORTS_SUCCESSFUL = False
    def run_solve_process(params): return {"error": "Moduł solvera niezaładowany.", "status": "failure"}, None, None
    def load_config(): return None
    def get_db(): yield None
    def create_db_and_tables(): pass
    # Zmieniono nazwę atrapy
    def seed_groups_and_relationships(db, event_id, num_guests, config): pass
    if 'BaseModel' not in locals(): from typing import TypeVar; BaseModel = TypeVar('BaseModel')
    if 'Field' not in locals(): Field = lambda *args, **kwargs: None
    if 'field_validator' not in locals(): field_validator = lambda *args, **kwargs: (lambda f: f)
    class Event: pass
    class AssignmentResult: pass
    class Session: pass
# Potrzebujemy configparser nawet jeśli importy zawiodą, do bloku EventCreate
import configparser

# Inicjalizacja FastAPI
app = FastAPI(
    title="WSP Solver API",
    description="API do rozwiązywania Problemu Rozmieszczania Gości Weselnych (WSP) z zarządzaniem wydarzeniami i wynikami.",
    version="0.3.2" # Zwiększamy wersję po poprawkach seedowania
)

# --- Modele danych Pydantic ---
# (Modele SolverInput, EventCreate, EventOutput, AssignmentResultOutput, SolverOutput)
class SolverInput(BaseModel):
    event_id: Optional[int] = Field(None, description="ID wydarzenia z bazy danych do przetworzenia (jeśli podane, num_guests jest ignorowane).")
    num_guests: Optional[int] = Field(None, ge=10, le=500, description="Liczba gości do WYGENEROWANIA (używane tylko jeśli event_id NIE jest podane).")
    algorithm: Optional[str] = Field(None, description="Algorytm ('Greedy', 'DSatur', 'Tabu Search') - nadpisuje config.")
    table_capacity: Optional[int] = Field(None, ge=2, description="Pojemność stołu - nadpisuje config.")
    balance_weight: Optional[float] = Field(None, ge=0.0, description="Waga balansu - nadpisuje config.")
    ts_max_iterations: Optional[int] = Field(None, ge=10, description="Max iteracji TS - nadpisuje config.")
    ts_tabu_tenure: Optional[int] = Field(None, ge=1, description="Kadencja Tabu TS - nadpisuje config.")
    generate_graph_viz: bool = Field(False, description="Czy wygenerować i zapisać statyczną wizualizację grafu grup?")

    @field_validator('event_id')
    def check_event_or_guests_id(cls, v, info):
        if v is None and info.data.get('num_guests') is None: raise ValueError("Należy podać 'event_id' lub 'num_guests'.")
        return v
    @field_validator('num_guests')
    def check_event_or_guests_num(cls, v, info):
        if v is None and info.data.get('event_id') is None: raise ValueError("Należy podać 'event_id' lub 'num_guests'.")
        return v
    class Config: json_schema_extra = { "examples": [ {"summary": "Rozwiąż z DB (ID=1)", "value": { "event_id": 1, "algorithm": "Tabu Search"}}, {"summary": "Generuj (80 gości)", "value": { "num_guests": 80, "algorithm": "DSatur", "table_capacity": 10 }}, {"summary": "Rozwiąż z DB i wygeneruj graf", "value": { "event_id": 2, "algorithm": "Greedy", "generate_graph_viz": True}} ]}

class EventCreate(BaseModel):
    name: str = Field(..., min_length=3, max_length=100, description="Nazwa nowego wydarzenia.")
    description: Optional[str] = Field(None, max_length=255, description="Opcjonalny opis wydarzenia.")
    num_guests_to_seed: Optional[int] = Field(None, ge=10, le=500, description="Jeśli podane, stworzy wydarzenie i od razu wypełni je wygenerowanymi danymi dla tej liczby gości.")

class EventOutput(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    class Config: from_attributes = True

class AssignmentResultOutput(BaseModel):
     id: int; event_id: int; run_timestamp: datetime.datetime; algorithm: str
     parameters_json: Optional[str] = None; score: Optional[float] = None
     conflicts: Optional[int] = None; assignment_json: Optional[str] = None
     status: str
     class Config: from_attributes = True

class SolverOutput(BaseModel):
    event_id: Optional[int] = None; event_name: Optional[str] = None
    num_guests_processed: Optional[int] = None; algorithm: Optional[str] = None
    num_groups: Optional[int] = None; num_tables_used: Optional[int] = None
    table_capacity: Optional[int] = None; assignment: Optional[Dict[str, int]] = None
    score: Optional[float] = None; conflicts: Optional[int] = None
    status: str; group_graph_path: Optional[str] = None; error: Optional[str] = None

# --- Punkty końcowe API ---

@app.on_event("startup")
def on_startup():
    """Wykonuje się przy starcie API."""
    logging.info("Uruchamianie API...")
    if IMPORTS_SUCCESSFUL:
        try: create_db_and_tables()
        except Exception as e: logging.error(f"Nie udało się zainicjalizować tabel przy starcie API: {e}")
    else: logging.error("Pominięto tworzenie tabel - importy modułów zależnych nie powiodły się.")

@app.get("/", summary="Root", tags=["General"], include_in_schema=False)
async def read_root():
    return {"message": "Witaj w API Rozmieszczania Gości Weselnych! Dokumentacja dostępna pod /docs"}

# --- Endpoints dla Wydarzeń (Events) ---

@app.post("/events", response_model=EventOutput, status_code=201, summary="Stwórz nowe wydarzenie", tags=["Events"])
async def create_new_event(event_data: EventCreate, db: Session = Depends(get_db)):
    """
    Tworzy nowe wydarzenie w bazie danych. Sprawdza unikalność nazwy.
    Jeśli podano `num_guests_to_seed`, po utworzeniu wydarzenia,
    wypełnia je wygenerowanymi danymi (grupami i relacjami).
    """
    if not IMPORTS_SUCCESSFUL or db is None:
         raise HTTPException(status_code=503, detail="Serwis niedostępny (problem z zależnościami lub bazą danych)")

    # 1. Sprawdź istnienie po nazwie PRZED próbą utworzenia
    existing_event = db.query(Event).filter(Event.name == event_data.name).first()
    if existing_event:
        raise HTTPException(status_code=400, detail=f"Wydarzenie o nazwie '{event_data.name}' już istnieje.")

    # 2. Utwórz wydarzenie
    db_event = Event(name=event_data.name, description=event_data.description)
    db.add(db_event)
    try:
        db.commit()
        db.refresh(db_event)
        logging.info(f"Utworzono wydarzenie: ID={db_event.id}, Nazwa='{db_event.name}'")

        if event_data.num_guests_to_seed:
            config = load_config()
            if not config:
                logging.warning(f"Nie można wczytać config.ini dla seedowania wydarzenia ID: {db_event.id}. Wydarzenie utworzone bez danych.")
                return db_event

            logging.info(f"Uruchamianie seed_groups_and_relationships dla event_id={db_event.id}")
            try:
                seed_groups_and_relationships(
                    db=db, # Przekaż bieżącą sesję
                    event_id=db_event.id,
                    num_guests=event_data.num_guests_to_seed,
                    config=config
                )
                db.commit()
                logging.info(f"Pomyślnie wypełniono danymi wydarzenie ID: {db_event.id}")
            except Exception as seed_e:
                db.rollback()
                logging.exception(f"Błąd podczas seedowania grup/relacji dla wydarzenia {db_event.id}: {seed_e}")
                logging.warning(f"Wydarzenie ID={db_event.id} utworzone, ale automatyczne wypełnienie danymi nie powiodło się.")

        return db_event

    except IntegrityError as ie:
         db.rollback()
         logging.error(f"Błąd unikalności podczas próby zapisu wydarzenia '{event_data.name}': {ie}")
         raise HTTPException(status_code=400, detail=f"Wydarzenie o nazwie '{event_data.name}' już istnieje (IntegrityError).")
    except SQLAlchemyError as e:
        db.rollback()
        logging.exception(f"Błąd SQLAlchemy podczas tworzenia wydarzenia '{event_data.name}': {e}")
        raise HTTPException(status_code=500, detail="Błąd serwera podczas zapisu wydarzenia do bazy danych.")
    except Exception as e:
        db.rollback()
        logging.exception(f"Nieoczekiwany błąd podczas tworzenia wydarzenia '{event_data.name}': {e}")
        raise HTTPException(status_code=500, detail="Nieoczekiwany błąd serwera.")


@app.get("/events", response_model=List[EventOutput], summary="Listuj wydarzenia", tags=["Events"])
async def list_events(db: Session = Depends(get_db)):
    """Pobiera listę wszystkich wydarzeń posortowaną według nazwy."""
    if not IMPORTS_SUCCESSFUL or db is None: raise HTTPException(status_code=503, detail="Serwis niedostępny")
    try:
        events = db.query(Event).order_by(Event.name).all()
        return events
    except Exception as e:
        logging.exception(f"Błąd podczas listowania wydarzeń: {e}")
        raise HTTPException(status_code=500, detail="Błąd serwera podczas odczytu wydarzeń.")

@app.get("/events/{event_id}", response_model=EventOutput, summary="Pobierz szczegóły wydarzenia", tags=["Events"])
async def get_event_details(event_id: int, db: Session = Depends(get_db)):
    """Pobiera szczegóły konkretnego wydarzenia na podstawie jego ID."""
    if not IMPORTS_SUCCESSFUL or db is None: raise HTTPException(status_code=503, detail="Serwis niedostępny")
    try:
        event = db.query(Event).filter(Event.id == event_id).first()
        if not event: raise HTTPException(status_code=404, detail=f"Wydarzenie o ID={event_id} nie znalezione.")
        return event
    except HTTPException: raise
    except Exception as e:
        logging.exception(f"Błąd podczas pobierania wydarzenia {event_id}: {e}")
        raise HTTPException(status_code=500, detail="Błąd serwera podczas pobierania wydarzenia.")

@app.delete("/events/{event_id}", status_code=204, summary="Usuń wydarzenie", tags=["Events"])
async def delete_event_by_id(event_id: int, db: Session = Depends(get_db)):
    """Usuwa wydarzenie i wszystkie powiązane z nim dane (grupy, relacje, wyniki) dzięki kaskadzie w bazie."""
    if not IMPORTS_SUCCESSFUL or db is None: raise HTTPException(status_code=503, detail="Serwis niedostępny")
    try:
        event = db.query(Event).filter(Event.id == event_id).first()
        if not event: raise HTTPException(status_code=404, detail=f"Wydarzenie o ID={event_id} nie znalezione.")
        db.delete(event); db.commit()
        logging.info(f"Usunięto wydarzenie ID={event_id}")
        return
    except HTTPException: raise
    except SQLAlchemyError as e:
        db.rollback(); logging.exception(f"Błąd SQLAlchemy podczas usuwania wydarzenia {event_id}: {e}")
        raise HTTPException(status_code=500, detail="Błąd serwera podczas usuwania wydarzenia z bazy.")
    except Exception as e:
        db.rollback(); logging.exception(f"Nieoczekiwany błąd podczas usuwania wydarzenia {event_id}: {e}")
        raise HTTPException(status_code=500, detail="Nieoczekiwany błąd serwera.")

@app.post("/solve", response_model=SolverOutput, summary="Rozwiąż problem WSP", tags=["Solver"])
async def solve_seating_problem(input_data: SolverInput = Body(...)):
    """
    Uruchamia proces rozwiązywania WSP.
    Wybiera dane na podstawie `event_id` (priorytet) lub generuje dla `num_guests`.
    Wynik (jeśli dotyczy istniejącego wydarzenia) jest zapisywany w bazie danych.
    """
    if not IMPORTS_SUCCESSFUL: raise HTTPException(status_code=503, detail="Serwis niedostępny (problem z modułem solvera)")
    if input_data.event_id is None and input_data.num_guests is None: raise HTTPException(status_code=422, detail="Należy podać 'event_id' lub 'num_guests'.")
    if input_data.event_id is not None and input_data.num_guests is not None:
         logging.warning("Podano jednocześnie event_id i num_guests. Ignoruję num_guests.")
         input_data.num_guests = None
    params_dict = input_data.model_dump(exclude_unset=True)
    logging.info(f"Uruchamianie solvera z parametrami: {params_dict}")
    results_summary, group_graph, assignment = run_solve_process(params_dict)
    if results_summary and results_summary.get("error"):
        error_message = results_summary["error"]; logging.error(f"Solver zwrócił błąd: {error_message}")
        status_code = 500 if "wewnętrzny błąd serwera" in error_message else 400
        raise HTTPException(status_code=status_code, detail=error_message)
    try:
        if assignment: results_summary["assignment"] = {str(k): v for k, v in assignment.items()}
        else:
            results_summary["assignment"] = None
            if results_summary.get("status") == "success": logging.warning("Solver zakończył się sukcesem, ale nie zwrócił przypisania (assignment=None).")
        final_output = SolverOutput(**results_summary)
        return final_output
    except Exception as pydantic_error:
        logging.exception(f"Błąd walidacji odpowiedzi Pydantic (SolverOutput): {pydantic_error}. Dane wejściowe: {results_summary}")
        raise HTTPException(status_code=500, detail="Błąd formatowania odpowiedzi serwera.")

@app.get("/events/{event_id}/results", response_model=List[AssignmentResultOutput], summary="Listuj wyniki dla wydarzenia", tags=["Results"])
async def list_results_for_event(event_id: int, db: Session = Depends(get_db)):
    """Pobiera historię wyników (uruchomień solvera) dla konkretnego wydarzenia, posortowaną od najnowszego."""
    if not IMPORTS_SUCCESSFUL or db is None: raise HTTPException(status_code=503, detail="Serwis niedostępny")
    try:
        event = db.query(Event).filter(Event.id == event_id).first()
        if not event: raise HTTPException(status_code=404, detail=f"Wydarzenie o ID={event_id} nie znalezione.")
        results = db.query(AssignmentResult).filter(AssignmentResult.event_id == event_id).order_by(AssignmentResult.run_timestamp.desc()).all()
        return results
    except HTTPException: raise
    except Exception as e:
        logging.exception(f"Błąd podczas listowania wyników dla wydarzenia {event_id}: {e}")
        raise HTTPException(status_code=500, detail="Błąd serwera podczas pobierania wyników.")

@app.get("/config", summary="Pokaż konfigurację", tags=["General"])
async def get_config_endpoint():
    """Zwraca zawartość wczytanego pliku konfiguracyjnego `config.ini`."""
    if not IMPORTS_SUCCESSFUL: raise HTTPException(status_code=503, detail="Serwis niedostępny (problem z modułem solvera)")
    config = load_config()
    if config is None: raise HTTPException(status_code=500, detail="Nie można załadować pliku konfiguracyjnego config.ini.")
    config_dict = {section: dict(config.items(section)) for section in config.sections()}
    return config_dict

# Koniec pliku api.py
