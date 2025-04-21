# -*- coding: utf-8 -*-
# Plik: src/seed_db.py

import logging
import pandas as pd
import json
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

# Używamy importów relatywnych
try:
    from .solver import generate_data, load_config
    from .database import engine, SessionLocal, Event, Group, Relationship, create_db_and_tables
    SEED_IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"BŁĄD: Nie można zaimportować potrzebnych modułów w seed_db.py: {e}")
    SEED_IMPORTS_SUCCESSFUL = False
    def generate_data(*args, **kwargs): return pd.DataFrame(), [], 0
    def load_config(): return None
    class Base: pass
    class Event(Base): pass
    class Group(Base): pass
    class Relationship(Base): pass
    def SessionLocal(): return None
    def create_db_and_tables(): pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
log = logging.getLogger(__name__)

def seed_groups_and_relationships(db: Session, event_id: int, num_guests: int, config):
    """Generuje grupy i relacje dla ISTNIEJĄCEGO wydarzenia i przygotowuje je do zapisu."""
    log.info(f"Rozpoczynanie dodawania grup/relacji dla event_id={event_id} ({num_guests} gości)")

    existing_groups = db.query(Group).filter(Group.event_id == event_id).first()
    if existing_groups:
        log.warning(f"Event ID={event_id} już posiada grupy. Pomijanie dodawania grup/relacji.")
        return

    if not config:
        log.error("Brak obiektu konfiguracji. Nie można wygenerować danych.")
        raise ValueError("Brak konfiguracji dla generatora danych.") # Rzuć błąd

    try:
        COUPLE_RATIO = config.getfloat('DataGeneration', 'couple_ratio', fallback=0.3)
        FAMILY_RATIO = config.getfloat('DataGeneration', 'family_ratio', fallback=0.1)
        CONFLICT_PROB = config.getfloat('DataGeneration', 'conflict_prob', fallback=0.1)
        PREFER_LIKE_PROB = config.getfloat('DataGeneration', 'prefer_like_prob', fallback=0.1)
        PREFER_DISLIKE_PROB = config.getfloat('DataGeneration', 'prefer_dislike_prob', fallback=0.1)
        WEIGHT_PREFER_WITH = config.getint('SeatingParameters', 'weight_prefer_with', fallback=-5)
        WEIGHT_PREFER_NOT_WITH = config.getint('SeatingParameters', 'weight_prefer_not_with', fallback=3)
        WEIGHT_CONFLICT = config.getint('SeatingParameters', 'weight_conflict', fallback=1000)
    except Exception as e:
        log.error(f"Błąd odczytu konfiguracji dla generatora danych: {e}")
        raise ValueError(f"Błąd konfiguracji: {e}")


    guest_df, relationships_list, num_groups_generated = generate_data(
        num_guests=num_guests, couple_ratio=COUPLE_RATIO, family_ratio=FAMILY_RATIO,
        conflict_prob=CONFLICT_PROB, prefer_like_prob=PREFER_LIKE_PROB, prefer_dislike_prob=PREFER_DISLIKE_PROB,
        WEIGHT_PREFER_WITH=WEIGHT_PREFER_WITH, WEIGHT_PREFER_NOT_WITH=WEIGHT_PREFER_NOT_WITH, WEIGHT_CONFLICT=WEIGHT_CONFLICT
    )
    if guest_df is None or guest_df.empty:
         log.error(f"generate_data nie zwróciło danych dla event_id={event_id}.")
         raise ValueError("generate_data zwróciło puste dane.")


    groups_to_add = []
    grouped = guest_df.groupby('GroupID')
    for group_id_gen, group_data in grouped:
        group_id_db = int(group_id_gen)
        group_size = len(group_data)
        guest_names = list(group_data['GuestName'])

        group = Group(id=group_id_db, event_id=event_id, size=group_size, guest_names_json=json.dumps(guest_names))
        groups_to_add.append(group)

    if not groups_to_add:
        log.warning(f"Nie wygenerowano żadnych grup do dodania dla event_id={event_id}")
        return

    db.add_all(groups_to_add)
    log.info(f"Przygotowano {len(groups_to_add)} grup do dodania dla event_id={event_id}.")


    guest_to_group_map = guest_df.set_index('GuestID')['GroupID'].to_dict()
    relationships_to_add = []
    processed_pairs = set()

    for rel in relationships_list:
        gid1, gid2 = rel['GuestID1'], rel['GuestID2']
        group1_id_gen = guest_to_group_map.get(gid1)
        group2_id_gen = guest_to_group_map.get(gid2)

        if group1_id_gen is not None and group2_id_gen is not None and group1_id_gen != group2_id_gen:
            group1_id_db = int(group1_id_gen)
            group2_id_db = int(group2_id_gen)
            pair = tuple(sorted((group1_id_db, group2_id_db)))

            if pair not in processed_pairs:
                relationship = Relationship(
                    event_id=event_id,
                    group1_id=pair[0],
                    group2_id=pair[1],
                    rel_type=rel['Type'],
                    weight=rel['Weight']
                )
                relationships_to_add.append(relationship)
                processed_pairs.add(pair)

    db.add_all(relationships_to_add)
    log.info(f"Przygotowano {len(relationships_to_add)} relacji do dodania dla event_id={event_id}.")
    log.info(f"Pomyślnie przygotowano grupy i relacje dla event_id={event_id} (oczekują na commit).")

if __name__ == "__main__":
    print("Uruchamianie skryptu seed_db.py...")

    if not SEED_IMPORTS_SUCCESSFUL:
        print("Krytyczny błąd: Nie udało się zaimportować wymaganych modułów. Zakończono.")
        exit(1)

    try:
        create_db_and_tables()
    except Exception as e:
        print(f"Błąd podczas tworzenia tabel: {e}")
        exit(1)

    main_config = load_config()
    if main_config is None:
        print("BŁĄD: Nie można wczytać config.ini.")
        exit(1)

    db_session: Session = SessionLocal()
    if db_session is None:
        print("BŁĄD: Nie można utworzyć sesji bazy danych.")
        exit(1)

    events_to_seed = [
        {"name": "Wesele Testowe (60)", "num_guests": 60},
        {"name": "Bankiet Firmowy (100)", "num_guests": 100},
    ]

    try:
        for event_data in events_to_seed:
            event_name = event_data["name"]
            num_guests = event_data["num_guests"]
            print(f"\nPrzetwarzanie wydarzenia: '{event_name}'")

            target_event = None
            try:
                target_event = db_session.query(Event).filter(Event.name == event_name).first()

                if not target_event:
                    print(f"  Tworzenie wydarzenia '{event_name}'...")
                    target_event = Event(name=event_name, description=f"Automatycznie wygenerowane przez seed_db.py ({num_guests} gości).")
                    db_session.add(target_event)
                    db_session.commit()
                    db_session.refresh(target_event)
                    print(f"  Utworzono wydarzenie ID: {target_event.id}")
                else:
                    print(f"  Wydarzenie '{event_name}' (ID: {target_event.id}) już istnieje.")

                event_id_to_seed = target_event.id
                print(f"  Próba dodania grup/relacji dla event_id={event_id_to_seed}...")

                seed_groups_and_relationships(db=db_session, event_id=event_id_to_seed, num_guests=num_guests, config=main_config)
                db_session.commit()
                print(f"  Zakończono próbę dodawania grup/relacji dla event_id={event_id_to_seed}.")

            except IntegrityError:
                db_session.rollback()
                print(f"  Błąd unikalności (IntegrityError) podczas przetwarzania '{event_name}'. Pomijanie.")
                target_event = db_session.query(Event).filter(Event.name == event_name).first()
                if target_event:
                     print(f"  Wydarzenie '{event_name}' (ID: {target_event.id}) istnieje.")
                continue
            except ValueError as ve:
                 print(f"  Błąd wartości podczas seedowania '{event_name}': {ve}")
                 db_session.rollback()
            except Exception as e_inner:
                 print(f"  Błąd podczas przetwarzania wydarzenia '{event_name}': {e_inner}")
                 db_session.rollback()

        print("\nZakończono działanie seed_db.py.")

    except Exception as main_e:
        print(f"\nKrytyczny błąd w pętli głównej seed_db.py: {main_e}")
        db_session.rollback()
    finally:
        if db_session:
            db_session.close()

# koniec pliku: src/seed_db.py