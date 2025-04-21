# -*- coding: utf-8 -*-
# Plik: src/dashboard.py

import streamlit as st
import pandas as pd
import json
import math
import os
import plotly.graph_objects as go
import logging
import numpy as np
import requests
import configparser 

try:
    from scipy import stats
except ImportError:
    st.error("Biblioteka SciPy nie jest zainstalowana. Uruchom 'pip install -r requirements.txt'.")
    stats = None

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')=

# Adres bazowy API (działającego w tym samym kontenerze)
API_BASE_URL = "http://localhost:8000"

# Import logiki solvera i bazy danych z użyciem importów absolutnych (bo PYTHONPATH=/app)
try:
    from src.solver import run_solve_process, load_config, visualize_group_graph
    from src.database import SessionLocal, Event 
    import networkx as nx
    try:
        import matplotlib
        MATPLOTLIB_AVAILABLE = True
    except ImportError:
        MATPLOTLIB_AVAILABLE = False
    DASHBOARD_IMPORTS_SUCCESSFUL = True
except ImportError as e:
    log.error(f"Dashboard: Nie można zaimportować modułów z 'src.solver' lub 'src.database': {e}. Używanie atrap.")
    DASHBOARD_IMPORTS_SUCCESSFUL = False
    # Definicje atrap funkcji i klas potrzebnych w dashboardzie
    def run_solve_process(params): return {"error": "Moduł solvera niezaładowany.", "status": "failure"}, None, None
    def load_config(): return None
    def visualize_group_graph(*args, **kwargs): log.error("Atrapa visualize_group_graph wywołana."); return None
    def SessionLocal(): log.error("Atrapa SessionLocal wywołana."); return None
    class Event: pass
    nx = None; MATPLOTLIB_AVAILABLE = False
except Exception as e:
    log.error(f"Dashboard: Błąd podczas importu lub inicjalizacji: {e}")
    DASHBOARD_IMPORTS_SUCCESSFUL = False
    def run_solve_process(params): return {"error": f"Błąd inicjalizacji: {e}", "status": "failure"}, None, None
    def load_config(): return None
    def visualize_group_graph(*args, **kwargs): log.error("Atrapa visualize_group_graph wywołana."); return None
    def SessionLocal(): log.error("Atrapa SessionLocal wywołana."); return None
    class Event: pass
    nx = None; MATPLOTLIB_AVAILABLE = False

# --- Funkcje pomocnicze ---

# Funkcja do pobierania wydarzeń z API
@st.cache_data(ttl=60)
def get_events_from_api():
    """Pobiera listę wydarzeń z API."""
    if not DASHBOARD_IMPORTS_SUCCESSFUL: return [] # Nie próbuj, jeśli importy zawiodły
    try:
        response = requests.get(f"{API_BASE_URL}/events")
        response.raise_for_status()
        log.info("Pobrano listę wydarzeń z API.")
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Błąd pobierania listy wydarzeń z API: {e}")
        log.error(f"Błąd API /events: {e}")
        return []

# Funkcja tworząca wykres Plotly dla planu stołów
def create_plotly_seating_plan(assignment, graph, table_capacity):
    """Tworzy interaktywny wykres planu stołów za pomocą Plotly."""
    if go is None: st.error("Biblioteka Plotly nie jest dostępna."); return None
    if not assignment: log.warning("create_plotly_seating_plan: Brak przypisania."); fig = go.Figure(); fig.update_layout(title_text="Brak danych przypisania"); return fig
    if not graph or nx is None: log.warning("create_plotly_seating_plan: Brak grafu lub biblioteki NetworkX."); fig = go.Figure(); fig.update_layout(title_text="Brak danych grafu"); return fig

    tables_content = {}; max_table_id = 0
    all_group_ids_in_assignment = set(assignment.keys()); graph_nodes = set(graph.nodes())
    missing_groups = all_group_ids_in_assignment - graph_nodes
    if missing_groups: log.warning(f"(Wiz Plotly) Grupy z przypisania nie znalezione w grafie: {missing_groups}")

    for group_id, table_id in assignment.items():
        if table_id is not None and isinstance(table_id, int) and table_id > 0:
            if group_id in graph_nodes:
                if table_id not in tables_content: tables_content[table_id] = {'groups': [], 'load': 0}
                try:
                    group_info = graph.nodes[group_id]; group_size = group_info.get('size', 1)
                    guest_names_list = group_info.get('guest_names', ['?']); max_guests_in_hover = 5
                    guest_names_str = ", ".join(guest_names_list[:max_guests_in_hover])
                    if len(guest_names_list) > max_guests_in_hover: guest_names_str += ", ..."
                    tables_content[table_id]['groups'].append({'id': group_id, 'size': group_size, 'names_str': guest_names_str})
                    tables_content[table_id]['load'] += group_size; max_table_id = max(max_table_id, table_id)
                except KeyError as e: log.warning(f"(Wiz Plotly) Błąd dostępu do danych węzła {group_id}: {e}")

    num_tables_used = len(tables_content)
    if num_tables_used == 0: log.warning("create_plotly_seating_plan: Brak stołów."); fig = go.Figure(); fig.update_layout(title_text="Brak przypisanych stołów"); return fig

    ncols = math.ceil(math.sqrt(num_tables_used)); nrows = math.ceil(num_tables_used / ncols)
    cell_width = 300; cell_height = 300; padding = 30; radius_factor = 0.35
    fig = go.Figure(); table_ids_sorted = sorted(tables_content.keys()); current_table_index = 0

    for r in range(nrows):
        for c in range(ncols):
            if current_table_index >= num_tables_used: break
            table_id = table_ids_sorted[current_table_index]; table_data = tables_content.get(table_id)
            if not table_data: continue
            table_load = table_data['load']; center_x = (c + 0.5) * cell_width; center_y = (r + 0.5) * cell_height
            radius = min(cell_width, cell_height) * radius_factor
            fill_color = 'lightcoral' if table_load > table_capacity else 'lightblue'; line_color = 'red' if table_load > table_capacity else 'black'
            fig.add_shape(type="circle", xref="x", yref="y", x0=center_x - radius, y0=center_y - radius, x1=center_x + radius, y1=center_y + radius, line_color=line_color, fillcolor=fill_color, opacity=0.7, layer='below')
            fig.add_annotation(x=center_x, y=center_y + radius + padding*0.5, text=f"Stół {table_id}<br>({table_load}/{table_capacity})", showarrow=False, font=dict(size=12, color="black"), align="center")
            hover_texts_table = [f"<b>Stół {table_id} ({table_load}/{table_capacity})</b>"]
            for group in table_data['groups']:
                 group_label = f"Gr{group['id']}({group['size']})"; hover_texts_table.append(f" - {group_label}: {group['names_str']}")
            fig.add_trace(go.Scatter(x=[center_x], y=[center_y], mode='markers', marker=dict(size=radius*2, color='rgba(0,0,0,0)'), hoverinfo='text', text="<br>".join(hover_texts_table), showlegend=False))
            current_table_index += 1
        if current_table_index >= num_tables_used: break

    total_height = nrows * cell_height + 2*padding + 50; total_width = ncols * cell_width + 2*padding + 50
    fig.update_layout(title=f"Interaktywny Plan Stołów (Najedź na stół po szczegóły)", xaxis=dict(visible=False, range=[-padding, ncols * cell_width + padding]), yaxis=dict(visible=False, range=[-padding, nrows * cell_height + padding], scaleanchor="x", scaleratio=1), plot_bgcolor='rgba(255,255,255,1)', height=max(600, total_height), width=max(600, total_width), hovermode='closest', margin=dict(l=10, r=10, t=50, b=10))
    return fig

# Funkcja do wczytywania i cache'owania wyników eksperymentów
@st.cache_data
def load_experiment_results(filename):
    """Wczytuje wyniki z pliku CSV."""
    if os.path.exists(filename):
        try: return pd.read_csv(filename)
        except pd.errors.EmptyDataError: st.warning(f"Plik wyników {filename} jest pusty."); return None
        except Exception as e: st.error(f"Błąd wczytywania {filename}: {e}"); return None
    else: return None

# --- Konfiguracja Strony Streamlit ---
st.set_page_config(layout="wide", page_title="WSP Dashboard")
st.title("Dashboard - Rozmieszczanie Gości Weselnych")

if not DASHBOARD_IMPORTS_SUCCESSFUL:
    st.error("Nie udało się załadować kluczowych modułów (solver/database). Funkcjonalność dashboardu jest ograniczona. Sprawdź logi kontenera.")
    st.stop() # Zatrzymaj wykonywanie reszty skryptu, jeśli importy zawiodły

# --- Panel Boczny - Ustawienia ---
st.sidebar.header("Tryb Uruchomienia")
run_mode = st.sidebar.radio("Wybierz źródło danych:", ["Generuj nowe dane", "Użyj wydarzenia z bazy"], key="run_mode")

st.sidebar.header("Parametry Uruchomienia")
config = load_config()
default_guests = 50; default_capacity = 10; default_balance = 0.5;
default_ts_iter = 1000; default_ts_tenure = 10; event_id = None; event_list = {}
selected_event_label = None # Zainicjuj poza blokiem if

if config:
    try:
        default_guests = config.getint('DataGeneration', 'default_num_guests', fallback=50)
    except (configparser.NoSectionError, configparser.NoOptionError, AttributeError):
        default_guests = 50
    default_capacity = config.getint('SeatingParameters', 'table_capacity', fallback=10) if config else 10
    default_balance = config.getfloat('SeatingParameters', 'balance_weight', fallback=0.5) if config else 0.5
    default_ts_iter = config.getint('TabuSearch', 'max_iterations', fallback=1000) if config else 1000
    default_ts_tenure = config.getint('TabuSearch', 'tabu_tenure', fallback=10) if config else 10
else:
    st.sidebar.warning("Nie można załadować config.ini. Używanie wartości domyślnych.")

# Dynamiczne UI w zależności od trybu
if run_mode == "Użyj wydarzenia z bazy":
    num_guests = None
    all_events_data = get_events_from_api()
    event_list = {f"{e['name']} (ID: {e['id']})": e['id'] for e in all_events_data} if all_events_data else {}

    if event_list:
        if 'selected_event_label' not in st.session_state:
             st.session_state.selected_event_label = list(event_list.keys())[0] # Wybierz pierwszy domyślnie

        # Aktualizuj selected_event_label na podstawie selectboxa
        st.session_state.selected_event_label = st.sidebar.selectbox(
            "Wybierz Wydarzenie",
            options=list(event_list.keys()),
            index = list(event_list.keys()).index(st.session_state.selected_event_label) if st.session_state.selected_event_label in event_list else 0, # Ustaw domyślny index
            key='event_selector_dropdown' # Nowy klucz dla selectboxa
        )
        selected_event_label = st.session_state.selected_event_label
        event_id = event_list.get(selected_event_label) # Pobierz ID
    else:
        st.sidebar.warning("Brak wydarzeń w bazie danych.")
        event_id = None
        selected_event_label = None
else: # Tryb "Generuj nowe dane"
    num_guests = st.sidebar.slider("Liczba Gości do wygenerowania", min_value=10, max_value=300, value=default_guests, step=5)
    event_id = None
    selected_event_label = None
    st.sidebar.caption("Pozostałe parametry generacji (ratio, prob) zostaną wzięte z `config.ini`.")


# Wspólne parametry algorytmów
st.sidebar.subheader("Parametry Algorytmu i Stołów")
table_capacity = st.sidebar.slider("Pojemność Stołu", min_value=2, max_value=20, value=default_capacity, step=1)
algorithm = st.sidebar.selectbox("Wybierz Algorytm", ["Greedy", "DSatur", "Tabu Search"], index=2)
balance_weight = st.sidebar.slider("Waga Balansu", min_value=0.0, max_value=5.0, value=default_balance, step=0.1)

ts_max_iterations = default_ts_iter; ts_tabu_tenure = default_ts_tenure
if algorithm == "Tabu Search":
    st.sidebar.subheader("Parametry Tabu Search")
    ts_max_iterations = st.sidebar.number_input("Max Iteracji TS", min_value=10, max_value=10000, value=default_ts_iter, step=100)
    ts_tabu_tenure = st.sidebar.number_input("Kadencja Tabu TS", min_value=1, max_value=50, value=default_ts_tenure, step=1)

st.sidebar.subheader("Opcje Wizualizacji")
generate_graph_viz = st.sidebar.checkbox("Generuj wizualizację grafu grup (plik PNG)", value=False, disabled=not MATPLOTLIB_AVAILABLE)
if not MATPLOTLIB_AVAILABLE: st.sidebar.caption("Matplotlib niedostępny, wizualizacja grafu niemożliwa.")

run_button = st.sidebar.button("Uruchom Solver")

# --- Zarządzanie Wydarzeniami w Sidebarze ---
st.sidebar.divider()
st.sidebar.header("Zarządzanie Wydarzeniami")

# Formularz tworzenia nowego wydarzenia
st.sidebar.subheader("Stwórz Nowe Wydarzenie")
with st.sidebar.form("new_event_form", clear_on_submit=True):
    new_event_name = st.text_input("Nazwa nowego wydarzenia", key="new_event_name_input")
    new_event_desc = st.text_area("Opis (opcjonalnie)", key="new_event_desc_input")
    seed_guests = st.checkbox("Wypełnij przykładowymi danymi?", key="seed_check")
    num_seed_guests = st.number_input(
        "Liczba gości do wygenerowania",
        min_value=10, max_value=500, value=50, step=5,
        disabled=not seed_guests,
        key="num_seed_guests_input"
    )
    create_submitted = st.form_submit_button("Stwórz Wydarzenie")
    if create_submitted:
        if not new_event_name:
            st.warning("Nazwa wydarzenia jest wymagana.")
        else:
            payload = {
                "name": new_event_name,
                "description": new_event_desc if new_event_desc else None,
                "num_guests_to_seed": int(num_seed_guests) if seed_guests else None
            }
            log.info(f"Wysyłanie żądania POST /events z payload: {payload}")
            try:
                response = requests.post(f"{API_BASE_URL}/events", json=payload)
                if response.status_code == 201:
                    st.success(f"Utworzono wydarzenie '{new_event_name}' (ID: {response.json()['id']}).")
                    get_events_from_api.clear() # Wyczyść cache listy wydarzeń
                    # Ustaw nowo utworzone wydarzenie jako aktywne, jeśli to możliwe
                    new_event_label = f"{new_event_name} (ID: {response.json()['id']})"
                    st.session_state.selected_event_label = new_event_label
                    st.rerun() # Odśwież stronę, aby zaktualizować listę
                else:
                    try: detail = response.json().get("detail", response.text)
                    except json.JSONDecodeError: detail = response.text
                    st.error(f"Błąd tworzenia wydarzenia ({response.status_code}): {detail}")
                    log.error(f"API /events POST error {response.status_code}: {detail}")
            except requests.exceptions.RequestException as e:
                st.error(f"Błąd połączenia z API podczas tworzenia wydarzenia: {e}")
                log.error(f"Connection error on POST /events: {e}")

# Sekcja usuwania wybranego wydarzenia
st.sidebar.subheader("Usuń Wybrane Wydarzenie")
if run_mode == "Użyj wydarzenia z bazy" and event_id is not None and selected_event_label is not None:
    st.sidebar.write(f"Wybrane: **{selected_event_label}**")
    confirm_key = f"delete_confirm_{event_id}"
    button_key = f"delete_btn_{event_id}"

    confirm_delete = st.sidebar.checkbox("Potwierdzam chęć usunięcia tego wydarzenia", key=confirm_key)
    delete_button = st.sidebar.button("Usuń Wydarzenie", disabled=not confirm_delete, key=button_key)

    if delete_button and confirm_delete:
        log.info(f"Wysyłanie żądania DELETE /events/{event_id}")
        try:
            response = requests.delete(f"{API_BASE_URL}/events/{event_id}")
            if response.status_code == 204:
                st.success(f"Usunięto wydarzenie ID: {event_id}.")
                get_events_from_api.clear() # Wyczyść cache
                # Zresetuj wybór w sesji po usunięciu
                if 'selected_event_label' in st.session_state:
                     del st.session_state['selected_event_label']
                st.rerun() # Odśwież stronę
            elif response.status_code == 404:
                 st.error(f"Błąd usuwania: Wydarzenie ID: {event_id} nie znalezione (być może już usunięte?).")
                 log.error(f"API /events/{event_id} DELETE error 404")
            else:
                try: detail = response.json().get("detail", response.text)
                except json.JSONDecodeError: detail = response.text
                st.error(f"Błąd usuwania wydarzenia ({response.status_code}): {detail}")
                log.error(f"API /events/{event_id} DELETE error {response.status_code}: {detail}")
        except requests.exceptions.RequestException as e:
            st.error(f"Błąd połączenia z API podczas usuwania wydarzenia: {e}")
            log.error(f"Connection error on DELETE /events/{event_id}: {e}")

elif run_mode == "Użyj wydarzenia z bazy":
     st.sidebar.info("Wybierz wydarzenie z listy, aby móc je usunąć.")
else:
     st.sidebar.info("Zarządzanie wydarzeniami jest dostępne tylko w trybie 'Użyj wydarzenia z bazy'.")


# --- Główny Obszar Wyświetlania Wyników ---
st.header("Wyniki Uruchomienia")

if 'last_results_summary' not in st.session_state: st.session_state.last_results_summary = None
if 'last_group_graph' not in st.session_state: st.session_state.last_group_graph = None
if 'last_assignment' not in st.session_state: st.session_state.last_assignment = None
if 'last_params' not in st.session_state: st.session_state.last_params = None

if run_button:
    if DASHBOARD_IMPORTS_SUCCESSFUL:
        params = {
            "algorithm": algorithm, "table_capacity": table_capacity,
            "balance_weight": balance_weight, "ts_max_iterations": ts_max_iterations,
            "ts_tabu_tenure": ts_tabu_tenure, "generate_graph_viz": generate_graph_viz
        }
        info_text = None
        if run_mode == "Generuj nowe dane" and num_guests is not None:
            params["num_guests"] = num_guests
            info_text = f"Uruchamianie solvera (generowanie dla {num_guests} gości) z algorytmem {algorithm}..."
        elif run_mode == "Użyj wydarzenia z bazy" and event_id is not None:
            params["event_id"] = event_id
            info_text = f"Uruchamianie solvera (dla wydarzenia ID: {event_id}, '{selected_event_label}') z algorytmem {algorithm}..."
        else:
            st.error("Wybierz poprawne parametry (liczbę gości lub wydarzenie z bazy).")
            info_text = None

        if info_text:
            st.session_state.last_params = params
            st.info(info_text)
            with st.spinner("Przetwarzanie..."):
                results_summary, group_graph, assignment = run_solve_process(params)
                st.session_state.last_results_summary = results_summary
                st.session_state.last_group_graph = group_graph
                st.session_state.last_assignment = assignment
                # Wymuś odświeżenie, aby pokazać nowe wyniki
                st.rerun()
    else:
        st.error("Nie można uruchomić solvera - problem z importem modułów.")

# Wyświetl wyniki
current_results = st.session_state.get('last_results_summary')
current_graph = st.session_state.get('last_group_graph')
current_assignment = st.session_state.get('last_assignment')
current_params = st.session_state.get('last_params')

if current_results:
    if current_results.get("status") == "success" and current_assignment is not None:
        event_display_name = current_results.get('event_name') or f'Generowane {current_results.get("num_guests_processed", "?")} gości'
        st.success(f"Wyniki dla: {event_display_name}, Algorytm: {current_results.get('algorithm')}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Wynik (Score)", f"{current_results.get('score'):.2f}" if current_results.get('score') is not None else "N/A")
        col2.metric("Konflikty", current_results.get('conflicts', 'N/A'))
        col3.metric("Użyte Stoły", current_results.get('num_tables_used', 'N/A'))

        st.subheader("Przypisanie Grup do Stołów (JSON):")
        try:
            assignment_str_keys = {str(k): v for k, v in current_assignment.items()}
            st.code(json.dumps(assignment_str_keys, indent=2), language='json')
        except Exception as json_e:
            st.error(f"Błąd formatowania przypisania jako JSON: {json_e}")
            st.text(str(current_assignment))

        st.subheader("Interaktywny Plan Stołów")
        if current_graph is not None:
            plotly_fig = create_plotly_seating_plan(current_assignment, current_graph, current_results.get("table_capacity", 10))
            if plotly_fig:
                st.plotly_chart(plotly_fig, use_container_width=False)
                try:
                    algo_name_safe = "".join(c if c.isalnum() else "_" for c in current_results.get('algorithm', ''))
                    event_suffix = f"ev{current_results.get('event_id')}" if current_results.get('event_id') else f"{current_results.get('num_guests_processed', 'gen')}g"
                    html_filename = f"seating_plan_{algo_name_safe}_{event_suffix}.html"
                    seating_plan_html = plotly_fig.to_html(include_plotlyjs='cdn')
                    st.download_button(label="Pobierz plan stołów (HTML)", data=seating_plan_html, file_name=html_filename, mime="text/html")
                except Exception as html_err:
                    st.warning(f"Błąd generowania przycisku pobierania HTML: {html_err}")
            else:
                st.warning("Nie udało się wygenerować wykresu Plotly planu stołów.")
        else:
            st.warning("Brak danych grafu (group_graph) do wygenerowania planu stołów.")

        st.subheader("Wizualizacja Grafu Grup (Statyczna)")
        group_graph_path = current_results.get("group_graph_path")
        if group_graph_path:
            expected_path_in_container = os.path.abspath(os.path.join("/app", group_graph_path))
            log.info(f"Sprawdzanie ścieżki do grafu: {expected_path_in_container}")
            if os.path.exists(expected_path_in_container):
                try:
                    st.image(expected_path_in_container, caption=f"Graf Grup ({expected_path_in_container})")
                    with open(expected_path_in_container, "rb") as fp:
                        st.download_button(label="Pobierz graf (PNG)", data=fp, file_name=os.path.basename(expected_path_in_container), mime="image/png")
                except Exception as img_e:
                    st.error(f"Błąd wyświetlania/pobierania grafu: {img_e}")
                    log.exception(f"Błąd podczas próby odczytu obrazu: {expected_path_in_container}")
            else:
                st.warning(f"Plik grafu nie istnieje w oczekiwanej lokalizacji: {expected_path_in_container}")
                st.info(f"Ścieżka zwrócona przez solver: {group_graph_path}")
                results_dir_path = "/app/results"
                if os.path.exists(results_dir_path):
                     st.info(f"Zawartość katalogu {results_dir_path}: {os.listdir(results_dir_path)}")
                else:
                     st.info(f"Katalog {results_dir_path} nie istnieje.")

        elif current_params and current_params.get('generate_graph_viz'):
            st.warning("Zażądano wizualizacji grafu, ale nie została wygenerowana lub ścieżka nie została zwrócona.")
        else:
            st.info("Nie zażądano generowania wizualizacji grafu grup.")

    elif current_results and current_results.get("error"):
        st.error(f"Błąd ostatniego uruchomienia: {current_results['error']}")
    elif current_results and current_assignment is None and current_results.get("status") == "success":
         st.warning("Solver zakończył działanie poprawnie, ale nie zwrócił żadnego przypisania.")
         st.json(current_results)

elif not run_button:
     if DASHBOARD_IMPORTS_SUCCESSFUL:
         st.info("Ustaw parametry w panelu bocznym i kliknij 'Uruchom Solver' lub zarządzaj wydarzeniami.")


# --- SEKCJA ANALIZY SKALOWALNOŚCI ZE STATYSTYKĄ ---
st.divider()
st.header("Analiza Skalowalności i Statystyczna")
RESULTS_FILENAME = "results/scalability_results.csv"
RESULTS_FILEPATH = os.path.join("/app", RESULTS_FILENAME) # Ścieżka w kontenerze

st.info(f"Ta sekcja analizuje wyniki z pliku `{RESULTS_FILEPATH}`. "
        f"Uruchom `docker compose exec wsp_app python src/run_experiments.py`, aby go wygenerować/zaktualizować.")

if st.button("Wczytaj/Odśwież wyniki skalowalności"):
    load_experiment_results.clear() # Wyczyść cache Streamlit dla tej funkcji

results_df = load_experiment_results(RESULTS_FILEPATH)

if results_df is not None:
    st.subheader("Surowe Wyniki Eksperymentów (fragment)")
    st.dataframe(results_df.head())

    required_cols = ['score', 'conflicts', 'num_guests', 'algorithm', 'exec_time_sec']
    if not all(col in results_df.columns for col in required_cols):
         st.error(f"Brak wymaganych kolumn w pliku CSV: {required_cols}. Aktualne kolumny: {results_df.columns.tolist()}")
    else:
        valid_results_df = results_df.dropna(subset=['score', 'conflicts']).copy()
        if 'conflicts' in valid_results_df.columns:
            valid_results_df['conflicts'] = valid_results_df['conflicts'].astype(int)

        if not valid_results_df.empty:
            st.subheader("Zagregowane Wyniki z 95% Przedziałami Ufności")
            aggregated_list = []
            if stats is not None:
                try:
                    for name, group in valid_results_df.groupby(['num_guests', 'algorithm'], sort=True):
                        n = len(group);
                        mean_score = group['score'].mean(); std_score = group['score'].std()
                        mean_time = group['exec_time_sec'].mean(); std_time = group['exec_time_sec'].std()
                        ci_score = (np.nan, np.nan); ci_time = (np.nan, np.nan)
                        if n > 1 and not np.isnan(std_score) and std_score > 0:
                            se_score = stats.sem(group['score'], nan_policy='omit');
                            if not np.isnan(se_score) and se_score > 0 : ci_score = stats.t.interval(0.95, df=n-1, loc=mean_score, scale=se_score)
                        if n > 1 and not np.isnan(std_time) and std_time > 0:
                            se_time = stats.sem(group['exec_time_sec'], nan_policy='omit');
                            if not np.isnan(se_time) and se_time > 0: ci_time = stats.t.interval(0.95, df=n-1, loc=mean_time, scale=se_time)
                        aggregated_list.append({'num_guests': name[0], 'algorithm': name[1],'avg_score': mean_score, 'std_score': std_score,'score_ci_lower': ci_score[0], 'score_ci_upper': ci_score[1],'avg_time': mean_time, 'std_time': std_time,'time_ci_lower': ci_time[0], 'time_ci_upper': ci_time[1],'run_count': n})

                    if aggregated_list:
                         aggregated_results = pd.DataFrame(aggregated_list)
                         agg_display = aggregated_results.copy()
                         agg_display['Score 95% CI'] = "[" + agg_display['score_ci_lower'].round(2).astype(str) + ", " + agg_display['score_ci_upper'].round(2).astype(str) + "]"
                         agg_display['Time 95% CI (s)'] = "[" + agg_display['time_ci_lower'].round(4).astype(str) + ", " + agg_display['time_ci_upper'].round(4).astype(str) + "]"
                         st.dataframe(agg_display[['num_guests', 'algorithm', 'avg_score', 'Score 95% CI','avg_time', 'Time 95% CI (s)', 'run_count']].round({'avg_score': 2, 'avg_time': 4}))

                         st.subheader("Wykresy Skalowalności z 95% CI")
                         algorithms = aggregated_results['algorithm'].unique()
                         st.markdown("##### Jakość Rozwiązania (Średni Wynik z 95% CI)")
                         fig_score_ci = go.Figure()
                         for algo in algorithms:
                             algo_data = aggregated_results[aggregated_results['algorithm'] == algo].sort_values('num_guests')
                             fig_score_ci.add_trace(go.Scatter(x=algo_data['num_guests'], y=algo_data['avg_score'], mode='lines+markers', name=algo))
                             plot_data_ci = algo_data.dropna(subset=['score_ci_lower', 'score_ci_upper'])
                             if not plot_data_ci.empty: fig_score_ci.add_trace(go.Scatter(x=np.concatenate([plot_data_ci['num_guests'], plot_data_ci['num_guests'][::-1]]), y=np.concatenate([plot_data_ci['score_ci_upper'], plot_data_ci['score_ci_lower'][::-1]]), fill='toself', fillcolor=fig_score_ci.data[-1].line.color, opacity=0.2, line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False, name=f'{algo} 95% CI'))
                         fig_score_ci.update_layout(xaxis_title='Liczba Gości', yaxis_title='Średni Wynik (Niższy = Lepszy)', title="Skalowalność - Jakość Rozwiązania (z 95% CI)", legend_title="Algorytm")
                         st.plotly_chart(fig_score_ci, use_container_width=True)

                         st.markdown("##### Czas Wykonania (Średni z 95% CI)")
                         fig_time_ci = go.Figure()
                         for algo in algorithms:
                             algo_data = aggregated_results[aggregated_results['algorithm'] == algo].sort_values('num_guests')
                             fig_time_ci.add_trace(go.Scatter(x=algo_data['num_guests'], y=algo_data['avg_time'], mode='lines+markers', name=algo))
                             plot_data_ci = algo_data.dropna(subset=['time_ci_lower', 'time_ci_upper']); plot_data_ci = plot_data_ci[plot_data_ci['time_ci_lower'] > 0]
                             if not plot_data_ci.empty: fig_time_ci.add_trace(go.Scatter(x=np.concatenate([plot_data_ci['num_guests'], plot_data_ci['num_guests'][::-1]]), y=np.concatenate([plot_data_ci['time_ci_upper'], plot_data_ci['time_ci_lower'][::-1]]), fill='toself', fillcolor=fig_time_ci.data[-1].line.color, opacity=0.2, line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False, name=f'{algo} 95% CI'))
                         fig_time_ci.update_layout(xaxis_title='Liczba Gości', yaxis_title='Średni Czas (s) - Skala Log', yaxis_type="log", title="Skalowalność - Czas Wykonania (z 95% CI)", legend_title="Algorytm")
                         st.plotly_chart(fig_time_ci, use_container_width=True)

                         st.subheader("Testy Istotności Statystycznej (Test t Welcha, p-value)")
                         st.caption("Porównanie wyników (score) Tabu Search z innymi algorytmami dla każdej liczby gości.")
                         significance_results = []; algorithms_to_compare = ['Greedy', 'DSatur']
                         for num_g in sorted(valid_results_df['num_guests'].unique()):
                             guest_data = valid_results_df[valid_results_df['num_guests'] == num_g]
                             scores_ts = guest_data[guest_data['algorithm'] == 'Tabu Search']['score'].dropna()
                             row_data = {"num_guests": num_g}
                             for algo_comp in algorithms_to_compare:
                                 scores_other = guest_data[guest_data['algorithm'] == algo_comp]['score'].dropna(); p_value = np.nan
                                 if len(scores_ts) >= 2 and len(scores_other) >= 2:
                                     try: t_stat, p_value = stats.ttest_ind(scores_ts, scores_other, equal_var=False, nan_policy='omit')
                                     except Exception as test_err: log.warning(f"Błąd testu t dla {num_g} gości ({algo_comp}): {test_err}"); p_value = np.nan
                                 row_data[f"p-value (TS vs {algo_comp})"] = p_value
                             significance_results.append(row_data)

                         if significance_results:
                             significance_df = pd.DataFrame(significance_results)
                             # Zaktualizowano stylizację zgodnie z ostrzeżeniem - użycie .map zamiast .applymap
                             def highlight_significant(p):
                                 if pd.isna(p): return ''
                                 color = 'green' if p < 0.05 else 'red'
                                 weight = 'bold' if p < 0.05 else 'normal'
                                 return f'color: {color}; font-weight: {weight};'
                             st.dataframe(significance_df.style.format({"p-value (TS vs Greedy)": "{:.4f}", "p-value (TS vs DSatur)": "{:.4f}"}).map(highlight_significant, subset=["p-value (TS vs Greedy)", "p-value (TS vs DSatur)"]))
                             st.caption("Wartości p < 0.05 (zielone, pogrubione) generalnie wskazują na istotną statystycznie różnicę (na korzyść TS, jeśli średnia TS jest niższa).")
                         else: st.info("Nie można było przeprowadzić testów statystycznych (za mało danych?).")
                    else: st.info("Brak zagregowanych wyników do wyświetlenia.")
                except Exception as agg_plot_e:
                    st.error(f"Błąd podczas agregacji, tworzenia wykresów lub testów statystycznych: {agg_plot_e}")
                    log.exception("Błąd w sekcji analizy skalowalności")
            else: # Brak danych w scipy.stats
                st.error("Nie można przeprowadzić analizy statystycznej - brak biblioteki SciPy.")
        else: # Pusty valid_results_df
            st.warning("Brak poprawnych wyników (po usunięciu NaN) do analizy skalowalności.")
else: # results_df is None
    st.warning(f"Nie znaleziono pliku wyników: {RESULTS_FILEPATH}")
    st.info(f"Bieżący katalog roboczy kontenera: {os.getcwd()}")

# koniec pliku: src/dashboard.py
