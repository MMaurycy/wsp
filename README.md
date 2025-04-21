# WSP Solver: Optymalizator Rozmieszczenia Gości Weselnych (i nie tylko!)

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
## Wprowadzenie

WSP Solver to zaawansowana aplikacja webowa zaprojektowana do rozwiązywania **Problemu Rozmieszczania Gości Weselnych** (Wedding Seating Problem - WSP), złożonego problemu optymalizacji kombinatorycznej. Aplikacja nie tylko implementuje różne algorytmy heurystyczne i metaheurystyczne do znalezienia optymalnego lub bliskiego optymalnemu rozmieszczenia gości przy stołach, ale również oferuje rozbudowane API RESTful, interaktywny dashboard do zarządzania i wizualizacji oraz trwałą persystencję danych w bazie PostgreSQL. Projekt jest w pełni skonteneryzowany przy użyciu Docker i Docker Compose.

Problem WSP polega na takim przypisaniu grup gości do stołów o ograniczonej pojemności, aby zminimalizować sumaryczny "koszt" wynikający z konfliktów między gośćmi oraz niespełnionych preferencji (np. chęci siedzenia blisko kogoś lub unikania kogoś), jednocześnie starając się zbalansować obciążenie stołów.

## Kluczowe Funkcje

* **Zaawansowany Solver:** Implementacja logiki rozwiązującej WSP z algorytmami:
    * **Greedy:** Prosta heurystyka zachłanna.
    * **DSatur:** Heurystyka kolorowania grafów adaptowana do problemu WSP.
    * **Tabu Search:** Metaheurystyka przeszukiwania lokalnego z mechanizmem pamięci (lista tabu) do unikania cykli i poprawy eksploracji przestrzeni rozwiązań.
* **Elastyczny Model Oceny:** Konfigurowalna funkcja celu (`calculate_seating_score`) uwzględniająca:
    * Kary za konflikty między grupami przy tym samym stole.
    * Nagrody/kary za spełnienie/niespełnienie preferencji (`PreferSitWith`, `PreferNotSitWith`).
    * Karę za niezbalansowane obciążenie stołów (opcjonalna waga `balance_weight`).
* **Generator Danych Testowych:** Możliwość generowania syntetycznych danych wejściowych (list gości, grup, relacji) na podstawie zadanych parametrów (liczba gości, proporcje par/rodzin, prawdopodobieństwa relacji).
* **Interaktywne API RESTful (FastAPI):**
    * Pełne zarządzanie **Wydarzeniami** (scenariuszami WSP) poprzez operacje CRUD (`POST`, `GET`, `DELETE`).
    * Endpoint `/solve` do uruchamiania procesu rozwiązywania dla danych z bazy (`event_id`) lub generowanych (`num_guests`), z możliwością konfiguracji parametrów algorytmu.
    * Endpoint `/events/{event_id}/results` do pobierania historii zapisanych wyników dla danego wydarzenia.
    * Automatycznie generowana dokumentacja interaktywna (Swagger UI) pod `/docs`.
* **Interaktywny Dashboard (Streamlit):**
    * Wybór trybu pracy: generowanie danych vs. użycie wydarzenia z bazy.
    * Pobieranie i wybór wydarzeń z bazy danych poprzez API.
    * **Zarządzanie wydarzeniami** bezpośrednio z UI (tworzenie z opcją seedowania, usuwanie).
    * Konfiguracja parametrów solvera i wizualizacji.
    * Wyświetlanie wyników pojedynczego uruchomienia (metryki, surowe przypisanie JSON).
    * **Wizualizacje:** Interaktywny plan stołów (Plotly) oraz opcjonalnie statyczny graf relacji między grupami (Matplotlib).
    * Możliwość pobrania wizualizacji (HTML, PNG).
    * **Analiza Skalowalności:** Sekcja prezentująca zagregowane wyniki eksperymentów z pliku CSV, w tym interaktywne wykresy (Plotly) i testy istotności statystycznej (SciPy).
    * Przycisk do uruchamiania skryptu eksperymentów w tle (`subprocess.Popen`).
* **Persystencja Danych (PostgreSQL + SQLAlchemy):**
    * Dane wydarzeń, grup, relacji oraz wyników uruchomień solvera są przechowywane w bazie danych PostgreSQL.
    * Użycie ORM SQLAlchemy do definicji modeli i interakcji z bazą.
    * Skrypt `seed_db.py` do inicjalnego wypełniania bazy przykładowymi danymi.
* **Pełna Konteneryzacja (Docker + Docker Compose):**
    * Aplikacja (API + Dashboard) i baza danych działają jako osobne, połączone kontenery Docker.
    * Łatwe uruchomienie całego środowiska za pomocą `docker compose up`.
    * Zarządzanie zależnościami przez `requirements.txt`.
    * Konfiguracja przez `config.ini` i zmienne środowiskowe.
* **(Przygotowane do Aktywacji) Integracja z MLflow:** Kod w `solver.py` zawiera logikę do śledzenia parametrów, metryk i artefaktów za pomocą MLflow (wymaga aktywacji i uruchomienia `mlflow ui`).

## Architektura

Aplikacja została zaprojektowana z myślą o separacji komponentów:

1.  **Backend API (FastAPI):** Odpowiada za logikę biznesową, walidację danych wejściowych, komunikację z bazą danych oraz uruchamianie logiki solvera. Wystawia interfejs RESTful.
2.  **Solver (`src/solver.py`):** Moduł Pythona zawierający implementacje algorytmów WSP, funkcję oceny oraz logikę przetwarzania danych (z bazy lub generatora) i zapisu wyników. Komunikuje się z bazą danych i (opcjonalnie) z MLflow.
3.  **Baza Danych (PostgreSQL):** Przechowuje dane o wydarzeniach, grupach, relacjach i historii wyników. Działa jako osobny serwis Docker.
4.  **Frontend Dashboard (Streamlit):** Interfejs użytkownika do interakcji z aplikacją. Komunikuje się z Backend API w celu pobierania danych (np. listy wydarzeń) i zlecania zadań (tworzenie/usuwanie wydarzeń, uruchamianie solvera). Wyświetla wyniki i wizualizacje.
5.  **Konteneryzacja (Docker):** Całość jest zarządzana przez Docker Compose, co ułatwia uruchomienie i zapewnia spójność środowiska. API i Dashboard działają w jednym kontenerze (`wsp_app`), komunikując się z kontenerem bazy danych (`db`).
6.  
## Technologie

* **Backend:** Python 3.10+, FastAPI, Uvicorn
* **Frontend:** Streamlit
* **Baza Danych:** PostgreSQL 15
* **ORM:** SQLAlchemy
* **Sterownik DB:** psycopg2-binary
* **Konteneryzacja:** Docker, Docker Compose
* **Przetwarzanie Danych/Grafy:** Pandas, NetworkX, NumPy
* **Wizualizacje:** Plotly (dashboard), Matplotlib (solver)
* **Testowanie Statystyczne:** SciPy
* **Śledzenie Eksperymentów:** MLflow (zintegrowane, gotowe do aktywacji)
* **Inne:** Requests (komunikacja dashboard-API), ConfigParser

## Instalacja i Uruchomienie

**Wymagania wstępne:**

* Git
* Docker ([Instrukcja instalacji](https://docs.docker.com/engine/install/))
* Docker Compose ([Instrukcja instalacji](https://docs.docker.com/compose/install/))

**Kroki:**

1.  **Sklonuj repozytorium:**
    ```bash
    git clone [https://sjp.pl/repozytorium](https://sjp.pl/repozytorium)
    cd [Nazwa Katalogu Repozytorium]
    ```

2.  **Konfiguracja (opcjonalnie):**
    * Parametry domyślne znajdują się w `config.ini`. Możesz je dostosować.
    * Dane logowania do bazy danych są ustawiane w `docker-compose.yml` w sekcji `environment` serwisu `db`. Domyślne to `wsp_user`/`wsp_password`.

3.  **Zbuduj i uruchom kontenery Docker:**
    ```bash
    docker compose up -d --build
    ```
    * `-d`: Uruchom w tle (detached mode).
    * `--build`: Wymuś przebudowanie obrazów przy pierwszej lub po zmianach w `Dockerfile`/kodzie.

4.  **Poczekaj na uruchomienie:** Baza danych potrzebuje chwili na inicjalizację (healthcheck w `docker-compose.yml` powinien o to zadbać). Możesz monitorować logi: `docker compose logs -f`.

5.  **Wypełnij bazę danych przykładowymi danymi (opcjonalnie, ale zalecane):**
    ```bash
    docker compose exec wsp_app python src/seed_db.py
    ```
    * Spowoduje to utworzenie kilku przykładowych wydarzeń (jeśli jeszcze nie istnieją) i wypełnienie ich wygenerowanymi grupami/relacjami.

## Użytkowanie

1.  **Dashboard:** Otwórz przeglądarkę i przejdź pod adres:
    * `http://localhost:8501`

2.  **API (Dokumentacja Swagger UI):** Otwórz przeglądarkę i przejdź pod adres:
    * `http://localhost:8000/docs`
    * Tutaj możesz interaktywnie testować wszystkie dostępne endpointy API.

**Podstawowy przepływ pracy w Dashboardzie:**

* Wybierz tryb "Użyj wydarzenia z bazy".
* Wybierz wydarzenie z listy rozwijanej (np. jedno z dodanych przez `seed_db.py`).
* Dostosuj parametry algorytmu (np. wybierz "Tabu Search").
* Kliknij "Uruchom Solver".
* Obserwuj wyniki (metryki, JSON, plan stołów).
* Spróbuj stworzyć własne wydarzenie za pomocą formularza w panelu bocznym (zaznaczając "Wypełnij przykładowymi danymi").
* Wybierz nowo utworzone wydarzenie i uruchom dla niego solver.
* Usuń testowe wydarzenie.

## Eksperymenty Skalowalności

* Skrypt `src/run_experiments.py` służy do przeprowadzania serii testów dla różnych rozmiarów problemu i algorytmów.
* Możesz go uruchomić ręcznie w kontenerze:
    ```bash
    docker compose exec wsp_app python src/run_experiments.py
    ```
* Alternatywnie, użyj przycisku "Uruchom Eksperymenty Skalowalności w tle" w sekcji "Analiza Skalowalności" w dashboardzie. Proces zostanie uruchomiony w tle w kontenerze.
* Wyniki są zapisywane w `results/scalability_results.csv`.
* Dashboard w sekcji "Analiza Skalowalności" wczytuje ten plik i prezentuje wyniki (użyj przycisku "Wczytaj/Odśwież...", aby zobaczyć najnowsze dane po zakończeniu eksperymentów).

## Integracja z MLflow (Opcjonalnie)

Kod w `src/solver.py` jest przygotowany do logowania przebiegów do MLflow. Aby skorzystać z tej funkcji:

1.  Upewnij się, że `mlflow` jest w `requirements.txt` i kontener jest przebudowany.
2.  Dodaj mapowanie portu dla MLflow UI w `docker-compose.yml` (jeśli jeszcze go nie ma):
    ```yaml
    services:
      wsp_app:
        ports:
          # ... inne porty ...
          - "5001:5001" # Port dla MLflow UI
    ```
3.  Uruchom MLflow UI (po wykonaniu kilku przebiegów solvera):
    ```bash
    docker compose exec wsp_app mlflow ui --host 0.0.0.0 --port 5001 --backend-store-uri file:///app/mlruns
    ```
4.  Otwórz `http://localhost:5001` w przeglądarce.

## Licencja

Ten projekt jest udostępniany na licencji MIT - zobacz plik [LICENSE](LICENSE) po szczegóły.

## Autor

* Marcin Przybylski - https://github.com/MMaurycy
