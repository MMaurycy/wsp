# WSP Solver: Optymalizator Rozmieszczenia Gości Weselnych (i nie tylko!)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit) ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat-square&logo=postgresql) ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker) ![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-D71F00?style=flat-square) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy) ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly) ![NetworkX](https://img.shields.io/badge/NetworkX-6a0dad?style=flat-square) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square&logo=matplotlib) ![SciPy](https://img.shields.io/badge/SciPy-8994D1?style=flat-square&logo=scipy) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

## Wprowadzenie

WSP Solver to zaawansowana aplikacja webowa zaprojektowana do rozwiązywania **Problemu Rozmieszczania Gości Weselnych** (Wedding Seating Problem - WSP), złożonego problemu optymalizacji kombinatorycznej. Aplikacja nie tylko implementuje różne algorytmy heurystyczne i metaheurystyczne do znalezienia optymalnego lub bliskiego optymalnemu rozmieszczenia gości przy stołach, ale również oferuje rozbudowane API RESTful, interaktywny dashboard do zarządzania i wizualizacji oraz trwałą persystencję danych w bazie PostgreSQL. Projekt jest w pełni skonteneryzowany przy użyciu Docker i Docker Compose.

Problem WSP polega na takim przypisaniu grup gości do stołów o ograniczonej pojemności, aby zminimalizować sumaryczny "koszt" wynikający z konfliktów między gośćmi oraz niespełnionych preferencji (np. chęci siedzenia blisko kogoś lub unikania kogoś), jednocześnie starając się zbalansować obciążenie stołów.

## Kluczowe Funkcje

* **Zaawansowany Solver:** Implementacja logiki rozwiązującej WSP z algorytmami:
    * Greedy
    * DSatur
    * Tabu Search
* **Elastyczny Model Oceny:** Konfigurowalna funkcja celu (`calculate_seating_score`) uwzględniająca konflikty, preferencje i balans stołów.
* **Generator Danych Testowych:** Możliwość generowania syntetycznych danych wejściowych.
* **Interaktywne API RESTful (FastAPI):**
    * Pełne zarządzanie Wydarzeniami (scenariuszami WSP) poprzez operacje CRUD (`POST`, `GET`, `DELETE`).
    * Endpoint `/solve` do uruchamiania procesu rozwiązywania dla danych z bazy (`event_id`) lub generowanych (`num_guests`).
    * Endpoint `/events/{event_id}/results` do pobierania historii zapisanych wyników.
    * Automatyczna dokumentacja Swagger UI (`/docs`).
* **Interaktywny Dashboard (Streamlit):**
    * Wybór trybu pracy (generowanie vs. baza danych).
    * Wybór wydarzenia z bazy danych poprzez API.
    * **Zarządzanie wydarzeniami** bezpośrednio z UI (tworzenie z opcją seedowania, usuwanie).
    * Konfiguracja parametrów solvera i wizualizacji.
    * Wyświetlanie wyników (metryki, JSON, interaktywny plan stołów Plotly, statyczny graf grup PNG).
    * Możliwość pobrania wizualizacji.
    * Sekcja **Analizy Skalowalności** z wykresami i statystykami.
    * Przycisk do uruchamiania eksperymentów w tle.
* **Persystencja Danych (PostgreSQL + SQLAlchemy):** Przechowywanie danych wydarzeń, grup, relacji i historii wyników.
* **Pełna Konteneryzacja (Docker + Docker Compose):** Łatwe uruchomienie całego środowiska.

## Architektura

Aplikacja została zaprojektowana z myślą o separacji komponentów:

1.  **Backend API (FastAPI):** Logika biznesowa, walidacja, komunikacja z DB, uruchamianie solvera.
2.  **Solver (`src/solver.py`):** Implementacje algorytmów, funkcja oceny, przetwarzanie danych, zapis wyników.
3.  **Baza Danych (PostgreSQL):** Przechowywanie danych. Osobny serwis Docker.
4.  **Frontend Dashboard (Streamlit):** Interfejs użytkownika. Komunikuje się z API.
5.  **Konteneryzacja (Docker):** API i Dashboard w kontenerze `wsp_app`, baza danych w kontenerze `db`. Uruchamiane przez Docker Compose.

## Technologie

* **Backend:** Python 3.10+, FastAPI, Uvicorn
* **Frontend:** Streamlit
* **Baza Danych:** PostgreSQL 15
* **ORM:** SQLAlchemy
* **Sterownik DB:** psycopg2-binary
* **Konteneryzacja:** Docker, Docker Compose
* **Przetwarzanie Danych/Grafy:** Pandas, NetworkX, NumPy
* **Wizualizacje:** Plotly, Matplotlib
* **Testowanie Statystyczne:** SciPy
* **Inne:** Requests, ConfigParser

## Algorytmy i Ocena

* **Zaimplementowane algorytmy:**
    * `Greedy`: Prosta, szybka heurystyka.
    * `DSatur`: Heurystyka kolorowania grafu.
    * `Tabu Search`: Metaheurystyka przeszukiwania lokalnego z pamięcią tabu.
* **Funkcja Oceny (`calculate_seating_score`):** Ocenia jakość rozwiązania, biorąc pod uwagę:
    * Konflikty między grupami przy tym samym stole (wysoka kara).
    * Preferencje (siedzenie razem - nagroda, siedzenie osobno - kara).
    * Równomierność rozmieszczenia gości przy stołach (kara za wariancję obciążenia).

## Schemat Bazy Danych

Główne tabele w bazie danych PostgreSQL:

* `events`: Przechowuje informacje o wydarzeniach (ID, nazwa, opis).
* `groups`: Przechowuje informacje o grupach gości powiązanych z danym wydarzeniem (ID grupy w ramach wydarzenia, ID wydarzenia, rozmiar grupy, lista nazwisk gości jako JSON).
* `relationships`: Przechowuje informacje o relacjach (konflikt, preferencje) między parami grup w ramach wydarzenia (ID wydarzenia, ID grupy 1, ID grupy 2, typ relacji, waga).
* `assignment_results`: Przechowuje historię wyników uruchomień solvera dla poszczególnych wydarzeń (ID wydarzenia, timestamp, algorytm, parametry, wynik, konflikty, przypisanie jako JSON, status).

## API

Pełna interaktywna dokumentacja API (Swagger UI) jest dostępna po uruchomieniu aplikacji pod adresem:

* `http://localhost:8000/docs`

Główne endpointy:

* `GET /events`: Listuje wszystkie wydarzenia.
* `POST /events`: Tworzy nowe wydarzenie (opcjonalnie z seedowaniem danych).
* `GET /events/{event_id}`: Pobiera szczegóły wydarzenia.
* `DELETE /events/{event_id}`: Usuwa wydarzenie.
* `POST /solve`: Uruchamia solver dla danego wydarzenia lub generowanych danych.
* `GET /events/{event_id}/results`: Listuje historię wyników dla wydarzenia.
* `GET /config`: Zwraca aktualną konfigurację z `config.ini`.

## Dashboard

Interaktywny dashboard dostępny pod adresem:

* `http://localhost:8501`

Umożliwia:

* Wybór trybu pracy (generowanie vs. baza danych).
* Wyświetlanie i wybieranie istniejących wydarzeń.
* **Tworzenie nowych wydarzeń** (z opcją wypełnienia danymi).
* **Usuwanie wybranych wydarzeń.**
* Konfigurację parametrów algorytmów i wizualizacji.
* Uruchamianie solvera dla wybranej konfiguracji.
* Przeglądanie wyników: metryki, JSON z przypisaniem.
* Wyświetlanie **interaktywnego planu stołów** (Plotly).
* Wyświetlanie (opcjonalnie) **statycznego grafu grup** (Matplotlib).
* Pobieranie wizualizacji (HTML, PNG).
* Przeglądanie **analizy skalowalności** z wykresami i statystykami.
* Uruchamianie **eksperymentów skalowalności** w tle.

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
    * Domyślne parametry działania znajdują się w `config.ini`.
    * Dane logowania do bazy danych (użytkownik, hasło, nazwa bazy) są ustawione w `docker-compose.yml`.

3.  **Zbuduj i uruchom kontenery Docker:**
    ```bash
    docker compose up -d --build
    ```

4.  **Wypełnij bazę danych przykładowymi danymi (zalecane przy pierwszym uruchomieniu):**
    * Poczekaj chwilę, aż kontener bazy danych (`db`) będzie w stanie "healthy" (sprawdź `docker compose ps`).
    * Uruchom skrypt seedujący:
        ```bash
        docker compose exec wsp_app python src/seed_db.py
        ```

## Użytkowanie

1.  **Dashboard:** Otwórz `http://localhost:8501` w przeglądarce.
2.  **API Docs:** Otwórz `http://localhost:8000/docs` w przeglądarce.

## Eksperymenty Skalowalności

* Możesz uruchomić predefiniowane eksperymenty za pomocą skryptu `src/run_experiments.py`:
    ```bash
    docker compose exec wsp_app python src/run_experiments.py
    ```
* Lub użyj przycisku "Uruchom Eksperymenty Skalowalności w tle" w Dashboardzie.
* Wyniki zostaną zapisane w `results/scalability_results.csv` i będą widoczne w sekcji analizy w Dashboardzie po odświeżeniu.

## Licencja

Ten projekt jest udostępniany na licencji MIT - zobacz plik `LICENSE` po szczegóły.

## Autor

*  Marcin Maurycy Przybylski - https://github.com/MMaurycy
