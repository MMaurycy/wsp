# -*- coding: utf-8 -*-
# Plik: src/run_experiments.py

import pandas as pd
import time
import logging
import configparser
import ast
import os
import numpy as np
import math
try:
    from solver import (load_config, generate_data, build_group_graph,
                        calculate_seating_score, greedy_seating,
                        color_graph_dsatur, tabu_search_seating)
except ImportError:
    print("BŁĄD KRYTYCZNY: Nie można zaimportować modułów z solver.py.")
    print("Upewnij się, że uruchamiasz ten skrypt z katalogu głównego projektu (np. /home/marcin/wsp/)")
    print("lub że katalog 'src' jest w ścieżce Pythona.")
    exit()
except Exception as e:
    print(f"BŁĄD KRYTYCZNY podczas importu z solver.py: {e}")
    exit()


log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler = logging.StreamHandler()
log_handler.setFormatter(log_formatter)
logger = logging.getLogger('experiment_runner')
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)
logging.getLogger('solver').setLevel(logging.WARNING)

def run_scalability_experiments():
    """
    Przeprowadza serię eksperymentów skalowalności dla różnych liczb gości i algorytmów.
    Zapisuje szczegółowe wyniki do pliku CSV.
    """
    logger.info("===== Rozpoczynanie Eksperymentów Skalowalności =====")
    start_total_time = time.time()

    config = load_config()
    if config is None:
        logger.error("Nie udało się załadować konfiguracji (config.ini). Przerywanie eksperymentów.")
        return

    try:
        guest_counts_str = config.get('Experiment', 'guest_counts', fallback='[50, 100]')
        try:
             GUEST_COUNTS_TO_TEST = ast.literal_eval(guest_counts_str)
             if not isinstance(GUEST_COUNTS_TO_TEST, list): raise ValueError("guest_counts nie jest listą")
             logger.info(f"Odczytano guest_counts: {GUEST_COUNTS_TO_TEST}")
        except Exception as e:
            logger.error(f"Błąd parsowania guest_counts ('{guest_counts_str}') z config.ini: {e}. Używam [50, 100].")
            GUEST_COUNTS_TO_TEST = [50, 100]

        NUM_RUNS = config.getint('Experiment', 'num_runs', fallback=5)
        RESULTS_FILENAME = config.get('Experiment', 'results_file', fallback='results/scalability_results.csv')
        RESULTS_DIR = os.path.dirname(RESULTS_FILENAME) or '.'
        if RESULTS_DIR and not os.path.exists(RESULTS_DIR):
             try:
                 os.makedirs(RESULTS_DIR)
                 logger.info(f"Utworzono katalog na wyniki: {RESULTS_DIR}")
             except OSError as e:
                 logger.error(f"Nie można utworzyć katalogu '{RESULTS_DIR}': {e}. Wyniki mogą nie zostać zapisane.")
                 RESULTS_FILENAME = os.path.basename(RESULTS_FILENAME)

        algorithms_to_run = ["Greedy", "DSatur", "Tabu Search"]

        COUPLE_RATIO = config.getfloat('DataGeneration', 'couple_ratio', fallback=0.3)
        FAMILY_RATIO = config.getfloat('DataGeneration', 'family_ratio', fallback=0.1)
        CONFLICT_PROB = config.getfloat('DataGeneration', 'conflict_prob', fallback=0.1)
        PREFER_LIKE_PROB = config.getfloat('DataGeneration', 'prefer_like_prob', fallback=0.1)
        PREFER_DISLIKE_PROB = config.getfloat('DataGeneration', 'prefer_dislike_prob', fallback=0.1)
        TABLE_CAPACITY = config.getint('SeatingParameters', 'table_capacity', fallback=10)
        BALANCE_WEIGHT = config.getfloat('SeatingParameters', 'balance_weight', fallback=0.5)
        WEIGHT_PREFER_WITH = config.getint('SeatingParameters', 'weight_prefer_with', fallback=-5)
        WEIGHT_PREFER_NOT_WITH = config.getint('SeatingParameters', 'weight_prefer_not_with', fallback=3)
        WEIGHT_CONFLICT = config.getint('SeatingParameters', 'weight_conflict', fallback=1000)
        TABLE_ESTIMATION_FACTOR = config.getfloat('SeatingParameters', 'table_estimation_factor', fallback=2.5)
        MAX_ITERATIONS = config.getint('TabuSearch', 'max_iterations', fallback=1000)
        TABU_TENURE = config.getint('TabuSearch', 'tabu_tenure', fallback=10)
        NO_IMPROVEMENT_STOP = config.getint('TabuSearch', 'no_improvement_stop', fallback=100)

        logger.info(f"Testowane liczby gości: {GUEST_COUNTS_TO_TEST}")
        logger.info(f"Liczba przebiegów na ustawienie: {NUM_RUNS}")
        logger.info(f"Testowane algorytmy: {algorithms_to_run}")
        logger.info(f"Plik wyników: {RESULTS_FILENAME}")

    except Exception as e:
        logger.exception(f"Błąd podczas wczytywania parametrów eksperymentu: {e}")
        return

    all_run_results = []

    # Główna pętla eksperymentu
    for num_guests in GUEST_COUNTS_TO_TEST:
        logger.info(f"\n----- Przetwarzanie dla {num_guests} gości -----")
        for run_num in range(1, NUM_RUNS + 1):
            logger.info(f"  --- Przebieg {run_num}/{NUM_RUNS} ---")
            run_start_time_total = time.time()

            guest_df, relationships, num_groups = generate_data(
                num_guests=num_guests, couple_ratio=COUPLE_RATIO, family_ratio=FAMILY_RATIO,
                conflict_prob=CONFLICT_PROB, prefer_like_prob=PREFER_LIKE_PROB, prefer_dislike_prob=PREFER_DISLIKE_PROB,
                WEIGHT_PREFER_WITH=WEIGHT_PREFER_WITH, WEIGHT_PREFER_NOT_WITH=WEIGHT_PREFER_NOT_WITH, WEIGHT_CONFLICT=WEIGHT_CONFLICT
            )
            if guest_df is None:
                logger.error(f"    Nie udało się wygenerować danych dla {num_guests} gości, przebieg {run_num}. Pomijanie.")
                continue

            group_graph = build_group_graph(guest_df, relationships, WEIGHT_PREFER_WITH, WEIGHT_PREFER_NOT_WITH, WEIGHT_CONFLICT)
            if group_graph is None:
                logger.error(f"    Nie udało się zbudować grafu dla {num_guests} gości, przebieg {run_num}. Pomijanie.")
                continue
            if group_graph.number_of_nodes() == 0: # Sprawdź, czy graf nie jest pusty
                logger.warning(f"    Wygenerowany graf dla {num_guests} gości (przebieg {run_num}) nie ma węzłów. Pomijanie.")
                continue


            num_tables_try = max(math.ceil(num_groups / (TABLE_CAPACITY / TABLE_ESTIMATION_FACTOR)), math.ceil(num_guests / TABLE_CAPACITY), 2);

            # Przetestuj każdy algorytm
            for algo_name in algorithms_to_run:
                logger.info(f"    Uruchamianie algorytmu: {algo_name}...")
                assignment = None
                score = float('inf')
                conflicts = -1
                exec_time = -1.0
                status = "failure"

                start_algo_time = time.time()
                try:
                    if algo_name == 'Greedy':
                        assignment = greedy_seating(group_graph, num_tables_try, TABLE_CAPACITY, WEIGHT_CONFLICT)
                    elif algo_name == 'DSatur':
                        assignment = color_graph_dsatur(group_graph, num_tables_try, TABLE_CAPACITY)
                    elif algo_name == 'Tabu Search':
                        initial_assignment_ts = color_graph_dsatur(group_graph, num_tables_try, TABLE_CAPACITY)
                        if initial_assignment_ts:
                             init_score, init_conflicts = calculate_seating_score(group_graph, initial_assignment_ts, num_tables_try, BALANCE_WEIGHT, WEIGHT_CONFLICT)
                             if init_conflicts != -1:
                                assignment, _ = tabu_search_seating(
                                     graph=group_graph, num_tables=num_tables_try, table_capacity=TABLE_CAPACITY,
                                     initial_assignment=initial_assignment_ts, max_iterations=MAX_ITERATIONS,
                                     tabu_tenure=TABU_TENURE, balance_weight=BALANCE_WEIGHT,
                                     WEIGHT_CONFLICT=WEIGHT_CONFLICT, NO_IMPROVEMENT_STOP=NO_IMPROVEMENT_STOP
                                 )
                                if init_conflicts > 0:
                                     logger.warning(f"      TS startował z DSatur, który miał {init_conflicts} konfliktów.")
                             else:
                                 logger.error("      TS: Błąd oceny rozwiązania startowego DSatur.")
                                 assignment = None
                        else:
                             logger.error("      TS: Nie udało się uzyskać rozwiązania startowego DSatur.")
                             assignment = None
                except Exception as algo_e:
                     logger.exception(f"      Wystąpił błąd podczas działania algorytmu {algo_name}: {algo_e}")
                     assignment = None

                end_algo_time = time.time()
                exec_time = end_algo_time - start_algo_time

                if assignment is not None:
                    score, conflicts = calculate_seating_score(graph=group_graph, assignment=assignment, num_tables=num_tables_try, balance_weight=BALANCE_WEIGHT, WEIGHT_CONFLICT=WEIGHT_CONFLICT)
                    if conflicts != -1: status = "success"
                else:
                     score = float('inf')
                     conflicts = -1

                run_result = {
                    "num_guests": num_guests,
                    "run_number": run_num,
                    "algorithm": algo_name,
                    "num_groups": num_groups,
                    "num_tables_estimated": num_tables_try,
                    "score": score if score != float('inf') else np.nan,
                    "conflicts": conflicts if conflicts != -1 else np.nan,
                    "exec_time_sec": exec_time,
                    "status": status
                }
                all_run_results.append(run_result)
                logger.info(f"      Wynik {algo_name}: Score={score:.2f}, Conflicts={conflicts}, Time={exec_time:.4f}s, Status={status}")

            run_end_time_total = time.time()
            logger.info(f"  --- Koniec przebiegu {run_num} (czas: {run_end_time_total - run_start_time_total:.2f}s) ---")

    if all_run_results:
        results_df = pd.DataFrame(all_run_results)
        try:
            results_df.to_csv(RESULTS_FILENAME, index=False, encoding='utf-8')
            logger.info(f"Zapisano szczegółowe wyniki eksperymentów do: {RESULTS_FILENAME}")
        except Exception as e:
            logger.exception(f"Nie udało się zapisać wyników do pliku CSV ({RESULTS_FILENAME}): {e}")
    else:
        logger.warning("Nie zebrano żadnych wyników z eksperymentów.")

    end_total_time = time.time()
    logger.info(f"===== Zakończono Eksperymenty Skalowalności (całkowity czas: {(end_total_time - start_total_time)/60:.2f} min) =====")

if __name__ == "__main__":
    run_scalability_experiments()

# koniec pliku: src/run_experiments.py