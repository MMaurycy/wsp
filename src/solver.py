# -*- coding: utf-8 -*-
# Plik: src/solver.py

import pandas as pd
import networkx as nx
import os
import time
import random
import math
import copy
import numpy as np
import configparser
import logging 
import json
import datetime

# ===>>> Importy do wykresów <<<===
import matplotlib
try:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.cm as cm
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    logging.error(f"Nie można zaimportować Matplotlib lub jego komponentów: {e}. Wizualizacje będą niedostępne.")
    MATPLOTLIB_AVAILABLE = False
    plt = None
    cm = None
    patches = None
except Exception as e:
    logging.error(f"Nie można skonfigurować Matplotlib (np. backend 'Agg'): {e}")
    MATPLOTLIB_AVAILABLE = False
    plt = None
    cm = None
    patches = None

# ===>>> Importy elementów bazy danych <<<===
try:
    from .database import SessionLocal, Event, Group, Relationship, AssignmentResult
except ImportError:
    logging.error("Nie można zaimportować modeli bazy danych z database.py.")
    def SessionLocal(): logging.error("SessionLocal z database.py nie jest dostępne!"); return None
    class Base: pass
    class Event(Base): pass
    class Group(Base): pass
    class Relationship(Base): pass
    class AssignmentResult(Base): pass
except Exception as e:
    logging.error(f"Nieoczekiwany błąd podczas importu z database.py: {e}")
    def SessionLocal(): logging.error("SessionLocal z database.py nie jest dostępne!"); return None
    class Base: pass
    class Event(Base): pass
    class Group(Base): pass
    class Relationship(Base): pass
    class AssignmentResult(Base): pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# --- Funkcja wczytująca konfigurację ---
def load_config(config_path='config.ini'):
    """Wczytuje konfigurację z pliku .ini"""
    config = configparser.ConfigParser()
    if not os.path.exists(config_path):
        try:
            script_dir = os.path.dirname(__file__)
            config_path_alt = os.path.join(script_dir, '..', 'config.ini')
        except NameError:
            config_path_alt = '../config.ini'

        if os.path.exists(config_path_alt):
            config_path = config_path_alt
        else:
            logging.error(f"Nie znaleziono pliku konfiguracyjnego: {config_path} ani {config_path_alt}")
            return None

    try:
        config.read(config_path, encoding='utf-8')
        logging.info(f"Wczytano konfigurację z: {config_path}")
        if not config.sections():
            logging.error(f"Plik konfiguracyjny {config_path} jest pusty lub nieprawidłowy.")
            return None
        return config
    except UnicodeDecodeError as ude:
        logging.error(f"Błąd dekodowania UTF-8 w pliku {config_path}: {ude}. Sprawdź kodowanie pliku!")
        return None
    except Exception as e:
        logging.exception(f"Błąd podczas wczytywania konfiguracji z {config_path}: {e}")
        return None


# --- Generator Danych ---
def generate_data(num_guests=50, couple_ratio=0.3, family_ratio=0.1,
                  conflict_prob=0.08, prefer_like_prob=0.1, prefer_dislike_prob=0.1,
                  WEIGHT_PREFER_WITH=-5, WEIGHT_PREFER_NOT_WITH=3, WEIGHT_CONFLICT=1000):
    """Generuje dane gości oraz listę relacji (konflikty i preferencje)."""
    logging.info(f"Generowanie danych dla {num_guests} gości...")
    guests = []; relationships = []; guest_id_counter = 0; group_id_counter = 0;
    num_couples = int(num_guests * couple_ratio / 2);
    num_families = int(num_guests * family_ratio / 3);

    for _ in range(num_couples):
        group_id_counter += 1
        id1, id2 = guest_id_counter + 1, guest_id_counter + 2
        guests.extend([
            {'GuestID': id1, 'GuestName': f'Gość_{id1}', 'GroupID': group_id_counter},
            {'GuestID': id2, 'GuestName': f'Gość_{id2}', 'GroupID': group_id_counter}
        ])
        guest_id_counter += 2

    for _ in range(num_families):
        group_id_counter += 1
        id1, id2, id3 = guest_id_counter + 1, guest_id_counter + 2, guest_id_counter + 3
        guests.extend([
            {'GuestID': id1, 'GuestName': f'Gość_{id1}', 'GroupID': group_id_counter},
            {'GuestID': id2, 'GuestName': f'Gość_{id2}', 'GroupID': group_id_counter},
            {'GuestID': id3, 'GuestName': f'Gość_{id3}', 'GroupID': group_id_counter}
        ])
        guest_id_counter += 3

    num_singles = num_guests - len(guests);
    if num_singles < 0:
        logging.warning(f"Ujemna liczba singli ({num_singles}) po wygenerowaniu par i rodzin. Ustawiono na 0.")
        num_singles = 0

    for _ in range(num_singles):
        group_id_counter += 1
        id1 = guest_id_counter + 1
        guests.append({'GuestID': id1, 'GuestName': f'Gość_{id1}', 'GroupID': group_id_counter})
        guest_id_counter += 1

    if len(guests) != num_guests:
        logging.warning(f"Wygenerowano {len(guests)} gości, mimo że żądano {num_guests}.")

    if not guests:
        logging.warning("Lista gości jest pusta po generacji.")
        return pd.DataFrame(columns=['GuestID', 'GuestName', 'GroupID']), [], 0

    guest_df = pd.DataFrame(guests)
    num_groups = group_id_counter

    guest_ids = list(guest_df['GuestID']);
    rel_count = {'Conflict': 0, 'PreferSitWith': 0, 'PreferNotSitWith': 0};
    guest_to_group_map_internal = guest_df.set_index('GuestID')['GroupID'].to_dict()

    for i in range(len(guest_ids)):
        for j in range(i + 1, len(guest_ids)):
            gid1, gid2 = guest_ids[i], guest_ids[j]
            group1 = guest_to_group_map_internal.get(gid1)
            group2 = guest_to_group_map_internal.get(gid2)

            if group1 is None or group2 is None or group1 == group2: continue

            rand_val = random.random()
            if rand_val < conflict_prob:
                relationships.append({'GuestID1': gid1, 'GuestID2': gid2, 'Type': 'Conflict', 'Weight': WEIGHT_CONFLICT})
                rel_count['Conflict'] += 1
            elif rand_val < conflict_prob + prefer_like_prob:
                relationships.append({'GuestID1': gid1, 'GuestID2': gid2, 'Type': 'PreferSitWith', 'Weight': WEIGHT_PREFER_WITH})
                rel_count['PreferSitWith'] += 1
            elif rand_val < conflict_prob + prefer_like_prob + prefer_dislike_prob:
                relationships.append({'GuestID1': gid1, 'GuestID2': gid2, 'Type': 'PreferNotSitWith', 'Weight': WEIGHT_PREFER_NOT_WITH})
                rel_count['PreferNotSitWith'] += 1

    logging.info(f"Wygenerowano {len(guest_df)} gości w {num_groups} grupach.")
    logging.info(f"Wygenerowano relacje: Konflikty={rel_count['Conflict']}, PreferSitWith={rel_count['PreferSitWith']}, PreferNotSitWith={rel_count['PreferNotSitWith']}")
    return guest_df, relationships, num_groups


# --- Budowanie Grafu ---
def build_group_graph(guest_df, relationships, WEIGHT_PREFER_WITH, WEIGHT_PREFER_NOT_WITH, WEIGHT_CONFLICT):
    """Buduje graf ważony na podstawie DataFrame gości i listy relacji."""
    if guest_df is None or guest_df.empty:
        logging.error("build_group_graph: Brak danych wejściowych (guest_df).")
        return None
    logging.info("Rozpoczynanie budowy grafu (z generatora)...")
    G = nx.Graph();
    try:
        grouped = guest_df.groupby('GroupID')
        for group_id, group_data in grouped:
            members = list(group_data['GuestID'])
            size = len(members)
            guest_names = list(group_data['GuestName'])
            G.add_node(group_id, members=members, size=size, guest_names=guest_names)

        guest_to_group_map = guest_df.set_index('GuestID')['GroupID'].to_dict();
        edge_data = {}

        if relationships:
            logging.info("Przetwarzanie relacji (z generatora)...")
            for rel in relationships:
                gid1, gid2 = rel['GuestID1'], rel['GuestID2']
                group1_id = guest_to_group_map.get(gid1)
                group2_id = guest_to_group_map.get(gid2)

                if group1_id is not None and group2_id is not None and group1_id != group2_id:
                    pair = tuple(sorted((group1_id, group2_id)));

                    if pair not in edge_data:
                        edge_data[pair] = {'weight': 0, 'conflict': False, 'type': 'Neutral'}

                    rel_type = rel.get('Type')

                    if rel_type == 'Conflict':
                        edge_data[pair]['conflict'] = True
                        edge_data[pair]['type'] = 'Conflict'
                        edge_data[pair]['weight'] = WEIGHT_CONFLICT
                    elif not edge_data[pair]['conflict']:
                        if rel_type == 'PreferSitWith':
                            if edge_data[pair]['type'] != 'PreferNotSitWith':
                                edge_data[pair]['type'] = 'PreferSitWith'
                            edge_data[pair]['weight'] += WEIGHT_PREFER_WITH
                        elif rel_type == 'PreferNotSitWith':
                            if edge_data[pair]['type'] != 'PreferSitWith':
                                edge_data[pair]['type'] = 'PreferNotSitWith'
                            edge_data[pair]['weight'] += WEIGHT_PREFER_NOT_WITH

            added_edges = 0
            for pair, data in edge_data.items():
                if data['conflict'] or data['weight'] != 0:
                    u, v = pair
                    final_weight = WEIGHT_CONFLICT if data['conflict'] else data['weight']
                    G.add_edge(u, v, weight=final_weight, conflict=data['conflict'], type=data['type'])
                    added_edges += 1
            logging.info(f"Dodano {added_edges} krawędzi (z generatora).")
        else:
            logging.info("Brak relacji do przetworzenia (z generatora).")

    except Exception as e:
        logging.exception(f"Wystąpił błąd podczas budowania grafu (z generatora): {e}")
        return None

    logging.info(f"Zakończono budowę grafu (z generatora). Węzły: {G.number_of_nodes()}, Krawędzie: {G.number_of_edges()}.")
    return G


# --- Algorytm Kolorowania DSatur ---
def color_graph_dsatur(graph, num_tables, table_capacity):
    """Koloruje graf za pomocą heurystyki DSatur uwzględniając pojemność stołów."""
    if graph is None or graph.number_of_nodes() == 0: logging.error("DSatur: Brak grafu lub graf jest pusty."); return None
    if num_tables <= 0: logging.error("DSatur: Liczba stołów musi być dodatnia."); return None
    if table_capacity <= 0: logging.error("DSatur: Pojemność stołu musi być dodatnia."); return None
    logging.info(f"Rozpoczynanie kolorowania DSatur (Stoły: {num_tables}, Pojemność: {table_capacity})...")

    start_time = time.time();
    nodes = list(graph.nodes());
    colors = {node: 0 for node in nodes};
    uncolored_nodes = set(nodes);
    original_degrees = {node: graph.degree(node) for node in nodes};
    table_load = {i: 0 for i in range(1, num_tables + 1)}

    while uncolored_nodes:
        selected_node = -1; max_saturation = -1; max_degree = -1;
        eligible_nodes = list(uncolored_nodes)

        for node in eligible_nodes:
            current_saturation = 0
            forbidden_colors_for_node = set()
            for neighbor in graph.neighbors(node):
                 if colors[neighbor] != 0:
                    edge_data = graph.get_edge_data(node, neighbor)
                    if edge_data and edge_data.get('conflict', False):
                        forbidden_colors_for_node.add(colors[neighbor])
            current_saturation = len(forbidden_colors_for_node)
            current_original_degree = original_degrees[node]

            if current_saturation > max_saturation:
                max_saturation = current_saturation; max_degree = current_original_degree; selected_node = node
            elif current_saturation == max_saturation:
                if current_original_degree > max_degree:
                    max_degree = current_original_degree; selected_node = node

        if selected_node == -1:
            if eligible_nodes: selected_node = random.choice(eligible_nodes)
            else: logging.critical("DSatur: Brak węzłów do pokolorowania?"); return None

        assigned_color = 0;
        try: node_size = graph.nodes[selected_node]['size']
        except KeyError: node_size = 1

        possible_colors = list(range(1, num_tables + 1)); random.shuffle(possible_colors)

        for color in possible_colors:
            has_conflicting_neighbor_with_this_color = False
            for neighbor in graph.neighbors(selected_node):
                edge_data = graph.get_edge_data(selected_node, neighbor)
                if edge_data and edge_data.get('conflict', False) and colors[neighbor] == color:
                    has_conflicting_neighbor_with_this_color = True; break
            if has_conflicting_neighbor_with_this_color: continue

            if table_load[color] + node_size <= table_capacity:
                assigned_color = color; break

        if assigned_color == 0:
            try: guest_names_str = str(graph.nodes[selected_node]['guest_names'])
            except KeyError: guest_names_str = "(brak nazw)"
            logging.error(f"DSatur: Nie można znaleźć dostępnego koloru/stołu (<= {num_tables}, pojemność {table_capacity}) dla węzła {selected_node} (grupa {guest_names_str}, rozmiar {node_size}).")
            actual_forbidden = set()
            for neighbor in graph.neighbors(selected_node):
                 edge_data = graph.get_edge_data(selected_node, neighbor)
                 if edge_data and edge_data.get('conflict', False) and colors[neighbor] != 0:
                     actual_forbidden.add(colors[neighbor])
            logging.error(f"     Zabronione kolory (przez sąsiadów z konfliktem): {sorted(list(actual_forbidden))}")
            logging.error(f"     Obecne obciążenie stołów: { {k: v for k, v in table_load.items() if v > 0} }")
            return None

        colors[selected_node] = assigned_color; table_load[assigned_color] += node_size; uncolored_nodes.remove(selected_node)

    end_time = time.time();
    logging.info(f"Zakończono kolorowanie DSatur w {end_time - start_time:.4f} sek.");
    return colors


# --- Funkcja Oceny Jakości ---
def calculate_seating_score(graph, assignment, num_tables, balance_weight=0.5, WEIGHT_CONFLICT=1000):
    """Oblicza wynik jakościowy danego rozmieszczenia."""
    if assignment is None: return float('inf'), -1
    if graph is None: logging.error("Ocena: Brak grafu."); return float('inf'), -1

    interaction_score = 0; conflict_penalty_sum = 0; num_violated_conflicts = 0;
    table_load = {i: 0 for i in range(1, num_tables + 1)}; tables = {}; total_guests = 0; valid_assignment_items = 0

    if not isinstance(assignment, dict): logging.error(f"Ocena: Oczekiwano słownika, otrzymano {type(assignment)}"); return float('inf'), -1

    for group_id in graph.nodes():
        table_id = assignment.get(group_id);
        try: node_size = graph.nodes[group_id]['size']
        except KeyError: node_size = 1

        if table_id is None or not isinstance(table_id, int) or table_id <= 0 or table_id > num_tables:
            logging.warning(f"Ocena: Grupa {group_id} ma nieprawidłowy lub brakujący stół ({table_id}). Nakładanie kary.")
            interaction_score += WEIGHT_CONFLICT * node_size
            num_violated_conflicts += 1
            continue

        valid_assignment_items += 1;
        if table_id not in tables: tables[table_id] = []
        tables[table_id].append(group_id);
        table_load[table_id] += node_size;
        total_guests += node_size

    if valid_assignment_items != len(graph.nodes()):
        logging.warning(f"Ocena: Nie wszystkie grupy ({len(graph.nodes())}) zostały poprawnie uwzględnione w ocenie ({valid_assignment_items})!")

    for table_id, groups_at_table in tables.items():
        if table_id <= 0 or table_id > num_tables: continue
        for i in range(len(groups_at_table)):
            for j in range(i + 1, len(groups_at_table)):
                u, v = groups_at_table[i], groups_at_table[j]
                if graph.has_edge(u, v):
                    edge_data = graph.get_edge_data(u, v);
                    is_conflict = edge_data.get('conflict', False)
                    weight = edge_data.get('weight', 0)

                    if is_conflict:
                        conflict_penalty_sum += WEIGHT_CONFLICT
                        num_violated_conflicts += 1
                    else:
                        interaction_score += weight

    balance_penalty = 0;
    actual_loads = [load for load in table_load.values() if load > 0]
    num_used_tables = len(actual_loads)
    if num_used_tables > 1 and total_guests > 0:
        average_load = total_guests / num_used_tables
        variance = sum((load - average_load) ** 2 for load in actual_loads) / num_used_tables
        balance_penalty = math.sqrt(variance) * balance_weight
    elif num_used_tables <= 1 :
        balance_penalty = 0

    final_score = interaction_score + conflict_penalty_sum + balance_penalty;
    return final_score, num_violated_conflicts

# --- Algorytm Zachłanny (Greedy) ---
def greedy_seating(graph, num_tables, table_capacity, WEIGHT_CONFLICT=1000):
    """Prosty algorytm zachłanny do rozmieszczania grup."""
    if graph is None or graph.number_of_nodes() == 0: logging.error("Greedy: Brak grafu lub graf jest pusty."); return None
    logging.info(f"Rozpoczynanie algorytmu zachłannego (Stoły: {num_tables}, Pojemność: {table_capacity})...")
    start_time = time.time();
    try:
        nodes = sorted(graph.nodes(), key=lambda node: graph.nodes[node]['size'], reverse=True)
    except KeyError:
        logging.error("Greedy: Brak atrybutu 'size' w węzłach grafu. Używam losowej kolejności.")
        nodes = list(graph.nodes()); random.shuffle(nodes)

    assignment = {node: 0 for node in nodes}; table_load = {i: 0 for i in range(1, num_tables + 1)}

    for group_id in nodes:
        try: node_size = graph.nodes[group_id]['size']
        except KeyError: node_size = 1

        assigned = False;
        table_options = list(range(1, num_tables + 1)); random.shuffle(table_options)

        for table_id in table_options:
            if table_load[table_id] + node_size <= table_capacity:
                has_conflict_at_table = False
                for other_group, assigned_table in assignment.items():
                    if assigned_table == table_id:
                        if graph.has_edge(group_id, other_group):
                            edge_data = graph.get_edge_data(group_id, other_group)
                            if edge_data and edge_data.get('conflict', False):
                                has_conflict_at_table = True; break
                if not has_conflict_at_table:
                    assignment[group_id] = table_id; table_load[table_id] += node_size; assigned = True; break
        if not assigned:
            logging.error(f"Greedy: Nie można przypisać grupy {group_id} (rozmiar {node_size}) do żadnego stołu.")
            logging.error(f"     Obecne obciążenie stołów: { {k: v for k, v in table_load.items() if v > 0} }")
            return None

    end_time = time.time();
    logging.info(f"Zakończono algorytm zachłanny w {end_time - start_time:.4f} sek.");
    return assignment


# --- Algorytm Tabu Search ---
def tabu_search_seating(graph, num_tables, table_capacity, initial_assignment,
                        max_iterations=1000, tabu_tenure=10, balance_weight=0.5,
                        WEIGHT_CONFLICT=1000, NO_IMPROVEMENT_STOP=100):
    """Algorytm Tabu Search do optymalizacji rozmieszczenia."""
    if initial_assignment is None: logging.error("TS: Brak rozwiązania początkowego."); return None, float('inf')
    if graph is None or graph.number_of_nodes() == 0: logging.error("TS: Brak grafu lub graf pusty."); return None, float('inf')
    logging.info(f"Rozpoczynanie Tabu Search (MaxIter={max_iterations}, Kadencja={tabu_tenure})...")

    start_time = time.time();
    current_assignment = copy.deepcopy(initial_assignment);
    current_score, current_conflicts = calculate_seating_score(graph, current_assignment, num_tables, balance_weight, WEIGHT_CONFLICT)

    if current_conflicts == -1: logging.error("TS: Błąd w ocenie rozwiązania początkowego."); return None, float('inf')
    if current_conflicts > 0: logging.warning(f"TS: Rozwiązanie początkowe zawiera {current_conflicts} konfliktów! Wynik początkowy: {current_score:.2f}")

    best_assignment = copy.deepcopy(current_assignment);
    best_score = current_score;
    best_conflicts = current_conflicts

    tabu_list = {};
    current_table_load = {i: 0 for i in range(1, num_tables + 1)}
    try:
        for group_id, table_id in current_assignment.items():
            if table_id > 0 and table_id <= num_tables:
                 node_data = graph.nodes.get(group_id)
                 if node_data:
                     node_size = node_data.get('size', 1)
                     if 'size' not in node_data: logging.warning(f"TS Init: Brak 'size' dla grupy {group_id}.")
                     current_table_load[table_id] += node_size
                 else:
                     logging.error(f"TS: Grupa {group_id} z przypisania nie istnieje w grafie! Przypisanie: {current_assignment}")
                     return None, float('inf') # Błąd krytyczny
            else:
                 logging.error(f"TS: Nieprawidłowy table_id={table_id} w initial_assignment dla grupy {group_id}.")
                 return None, float('inf')
    except Exception as e:
        logging.exception(f"TS: Błąd podczas inicjalizacji obciążenia stołów: {e}")
        return None, float('inf')

    last_improvement_iter = 0

    for iteration in range(max_iterations):
        best_neighbor_assignment = None
        best_neighbor_score = float('inf')
        best_move = None
        found_move_this_iteration = False

        nodes_to_consider = list(graph.nodes()); random.shuffle(nodes_to_consider)

        for group_id in nodes_to_consider:
            original_table = current_assignment.get(group_id, 0);
            if original_table <= 0 or original_table > num_tables: continue

            try: node_size = graph.nodes[group_id]['size']
            except KeyError: node_size = 1

            possible_targets = list(range(1, num_tables + 1)); random.shuffle(possible_targets)

            for target_table in possible_targets:
                if target_table == original_table: continue
                if current_table_load.get(target_table, 0) + node_size > table_capacity: continue

                move_creates_conflict = False
                for other_group, other_table in current_assignment.items():
                    if other_group != group_id and other_table == target_table:
                        if graph.has_edge(group_id, other_group):
                            edge_data = graph.get_edge_data(group_id, other_group)
                            if edge_data and edge_data.get('conflict', False):
                                move_creates_conflict = True; break
                if move_creates_conflict: continue

                temp_assignment = current_assignment.copy(); temp_assignment[group_id] = target_table
                neighbor_score, neighbor_conflicts = calculate_seating_score(graph, temp_assignment, num_tables, balance_weight, WEIGHT_CONFLICT)

                if neighbor_conflicts == -1 or neighbor_conflicts > 0: continue

                move = (group_id, target_table)
                tabu_check_key = (group_id, original_table)
                is_tabu = tabu_check_key in tabu_list and iteration < tabu_list[tabu_check_key];
                aspiration_met = (neighbor_score < best_score)

                is_best_neighbor_so_far = (neighbor_score < best_neighbor_score)

                if (not is_tabu and is_best_neighbor_so_far) or aspiration_met:
                    best_neighbor_score = neighbor_score
                    best_move = move
                    best_neighbor_assignment = temp_assignment
                    found_move_this_iteration = True

        if found_move_this_iteration:
            moved_group, target_table = best_move
            original_table = current_assignment[moved_group]

            current_assignment = best_neighbor_assignment
            current_score = best_neighbor_score

            try: group_size = graph.nodes[moved_group]['size']
            except KeyError: group_size = 1

            current_table_load[original_table] -= group_size
            current_table_load[target_table] += group_size

            reverse_move_tabu_key = (moved_group, original_table)
            tabu_list[reverse_move_tabu_key] = iteration + tabu_tenure

            if iteration % 50 == 0:
                 tabu_list = {k: v for k, v in tabu_list.items() if v > iteration}

            if current_score < best_score:
                best_assignment = copy.deepcopy(current_assignment)
                best_score = current_score
                best_conflicts = 0
                last_improvement_iter = iteration

        if iteration - last_improvement_iter > NO_IMPROVEMENT_STOP:
            logging.info(f"  Zatrzymanie TS: Brak poprawy przez {iteration - last_improvement_iter} iteracji.")
            break

    end_time = time.time();
    final_score, final_conflicts = calculate_seating_score(graph, best_assignment, num_tables, balance_weight, WEIGHT_CONFLICT)

    if final_conflicts > 0: logging.warning(f"TS: Najlepsze znalezione rozwiązanie nadal zawiera {final_conflicts} konfliktów! Wynik: {final_score:.2f}")
    elif final_conflicts == -1: logging.error(f"TS: Błąd w ocenie ostatecznego najlepszego rozwiązania.")

    logging.info(f"Zakończono Tabu Search w {end_time - start_time:.4f} sek. Najlepszy wynik: {final_score:.2f} (Konflikty: {final_conflicts if final_conflicts !=-1 else 'Błąd'})")
    return best_assignment, final_score


# --- Wizualizacja Planu Stołów ---
def visualize_seating_plan(assignment, graph, table_capacity, filename="results/seating_plan.png",
                           MAX_GUESTS_TO_LIST=3, DEFAULT_FONT_SIZE=7, REDUCTION_THRESHOLD=5, REDUCTION_FACTOR=0.8):
    """Generuje wizualizację rozmieszczenia grup przy stołach z dynamiczną czcionką."""
    if plt is None or patches is None: logging.error("Matplotlib/komponenty nie są dostępne. Pomijanie wizualizacji planu stołów."); return
    logging.info(f"Generowanie wizualizacji planu stołów ({filename})...")
    if assignment is None: logging.warning("Wiz Planu: Brak przypisania."); return
    if graph is None: logging.warning("Wiz Planu: Brak grafu."); return
    if not isinstance(assignment, dict): logging.error(f"Wiz Planu: Oczekiwano słownika, otrzymano {type(assignment)}"); return

    tables_content = {}
    all_groups = set(graph.nodes()); assigned_groups = set(); max_table_id = 0
    for group_id, table_id in assignment.items():
        if table_id is not None and isinstance(table_id, int) and table_id > 0 :
            if group_id in graph:
                if table_id not in tables_content: tables_content[table_id] = []
                try:
                    group_data = graph.nodes[group_id].copy(); group_data['group_id'] = group_id
                    tables_content[table_id].append(group_data); assigned_groups.add(group_id); max_table_id = max(max_table_id, table_id)
                except KeyError: logging.warning(f"Wiz Planu: Brak danych dla grupy {group_id} w grafie.")
            else: logging.warning(f"Wiz Planu: Grupa {group_id} z przypisania nie istnieje w grafie.")
    unassigned = all_groups - assigned_groups
    if unassigned: logging.warning(f"Wiz Planu: Grupy nieprzypisane w wizualizacji: {unassigned}")

    num_tables_used = len(tables_content)
    if num_tables_used == 0: logging.warning("Wiz Planu: Brak grup przypisanych do stołów."); return

    ncols = math.ceil(math.sqrt(num_tables_used))
    nrows = math.ceil(num_tables_used / ncols)
    fig, ax = plt.subplots(figsize=(ncols * 4.5, nrows * 4.5))
    ax.set_aspect('equal', adjustable='box'); plt.axis('off')
    plt.title(f"Wizualizacja Rozmieszczenia (Stoły: {num_tables_used}, Pojemność: {table_capacity})", fontsize=16, pad=20)

    col_width = 1.0 / ncols; row_height = 1.0 / nrows; padding = 0.04; radius = min(col_width, row_height) / 3.5

    current_table_index = 0; table_ids_sorted = sorted(tables_content.keys())

    for r in range(nrows):
        for c in range(ncols):
            if current_table_index >= num_tables_used: break
            table_id = table_ids_sorted[current_table_index]
            groups = tables_content[table_id]; table_load = sum(g.get('size', 1) for g in groups)
            center_x = (c + 0.5) * col_width; center_y = 1.0 - (r + 0.5) * row_height

            edge_color = 'red' if table_load > table_capacity else 'black'; face_color = 'lightcoral' if table_load > table_capacity else 'lightblue'
            circle = patches.Circle((center_x, center_y), radius, linewidth=1.5, edgecolor=edge_color, facecolor=face_color, alpha=0.5)
            ax.add_patch(circle)

            ax.text(center_x, center_y + radius + padding * 0.4, f"Stół {table_id}\n({table_load}/{table_capacity})", ha='center', va='bottom', fontsize=10, weight='bold')

            text_y = center_y - radius - padding * 0.3
            num_groups_at_table = len(groups)
            font_size = DEFAULT_FONT_SIZE
            if num_groups_at_table > REDUCTION_THRESHOLD:
                 font_size = max(4, DEFAULT_FONT_SIZE - (num_groups_at_table - REDUCTION_THRESHOLD))

            group_texts = []
            for group in groups:
                group_id = group.get('group_id', '?'); group_size = group.get('size', '?')
                group_label = f"Gr{group_id}({group_size})"
                guest_names = group.get('guest_names', [])
                guest_names_to_show = guest_names[:MAX_GUESTS_TO_LIST]
                guest_list_str = ",".join(guest_names_to_show)
                if len(guest_names) > MAX_GUESTS_TO_LIST: guest_list_str += ",.."
                group_texts.append(f"{group_label}: {guest_list_str}")

            full_text = "\n".join(group_texts)
            ax.text(center_x, text_y, full_text, ha='center', va='top', fontsize=font_size)

            current_table_index += 1
        if current_table_index >= num_tables_used: break

    ax.set_xlim(0 - padding, 1 + padding); ax.set_ylim(0 - padding, 1 + padding)

    try:
        results_dir = os.path.dirname(filename)
        if results_dir and not os.path.exists(results_dir):
            os.makedirs(results_dir)
        plt.savefig(filename); logging.info(f"Zapisano wizualizację planu stołów: {filename}")
    except Exception as e: logging.exception(f"BŁĄD podczas zapisywania wizualizacji planu stołów: {e}")
    finally:
        if 'fig' in locals() and fig is not None: plt.close(fig)


# --- Wizualizacja Grafu Grup ---
def visualize_group_graph(assignment, graph, num_tables, filename="results/group_graph.png",
                          WEIGHT_PREFER_WITH=-5, WEIGHT_PREFER_NOT_WITH=3, WEIGHT_CONFLICT=1000):
    """Generuje wizualizację grafu grup - próba poprawy czytelności."""
    if plt is None or cm is None or patches is None or nx is None: logging.error("Biblioteki wizualizacyjne (Matplotlib/NetworkX) nie są dostępne. Pomijanie wizualizacji grafu grup."); return
    logging.info(f"Generowanie wizualizacji grafu grup ({filename}) - próba poprawy czytelności...")
    if assignment is None: logging.warning("Wiz Grafu: Brak przypisania."); return
    if graph is None: logging.warning("Wiz Grafu: Brak grafu."); return
    if not isinstance(assignment, dict): logging.error(f"Wiz Grafu: Oczekiwano słownika przypisania, otrzymano {type(assignment)}"); return
    if num_tables <= 0 : logging.warning(f"Wiz Grafu: Nieprawidłowa liczba stołów ({num_tables}) do generowania mapy kolorów."); return

    # 1. Przygotuj kolory węzłów
    node_colors = []
    node_cmap_list = []
    try:
        cmap = cm.get_cmap('viridis', num_tables)
        node_cmap_list = [cmap(i) for i in range(num_tables)]
    except Exception as e:
        logging.error(f"Nie można uzyskać mapy kolorów matplotlib: {e}. Używam losowych kolorów.")
        for _ in range(num_tables): node_cmap_list.append(np.random.rand(3,).tolist())

    unassigned_color = 'grey'
    node_labels = {}

    for node in graph.nodes():
        table_id = assignment.get(node)
        if table_id is not None and isinstance(table_id, int) and 1 <= table_id <= num_tables and table_id - 1 < len(node_cmap_list):
            node_colors.append(node_cmap_list[table_id - 1])
        else: node_colors.append(unassigned_color)
        node_labels[node] = f"G{node}"

    # 2. Przygotuj kolory/style krawędzi
    relation_styles = {
        'Conflict':       {'color': 'red',     'style': 'solid',  'width': 1.8, 'alpha': 0.8},
        'PreferSitWith':  {'color': 'green',   'style': 'solid',  'width': 1.2, 'alpha': 0.7},
        'PreferNotSitWith':{'color': 'orange', 'style': 'dashed', 'width': 0.6, 'alpha': 0.4},
        'Neutral':        {'color': 'lightgrey','style': 'dotted', 'width': 0.4, 'alpha': 0.15} 
    }
    default_style = relation_styles['Neutral']
    edges_to_draw = []
    edge_attribute_list = {'color': [], 'style': [], 'width': [], 'alpha': []}
    node_count_check = graph.number_of_nodes()

    for u, v, data in graph.edges(data=True):
        edge_type = data.get('type', 'Neutral')
        if edge_type == 'Neutral':
            if data.get('conflict', False): edge_type = 'Conflict'

        # Pomijaj rysowanie krawędzi neutralnych, jeśli graf jest duży (np. > 50 węzłów)
        if edge_type == 'Neutral' and node_count_check > 50:
            continue

        style = relation_styles.get(edge_type, default_style)
        edges_to_draw.append((u, v))
        edge_attribute_list['color'].append(style['color'])
        edge_attribute_list['style'].append(style['style'])
        edge_attribute_list['width'].append(style['width'])
        edge_attribute_list['alpha'].append(style['alpha'])


    # 3. Układ grafu - Spring Layout z agresywnymi parametrami
    logging.info("Obliczanie układu grafu Spring Layout (może chwilę potrwać)...")
    start_layout_time = time.time()
    pos = None
    try:
        node_count = graph.number_of_nodes()
        if node_count == 0: raise ValueError("Graf nie ma węzłów.")
        # ===>>> WIĘKSZE k dla separacji <<<===
        k_val = 2.5 / math.sqrt(node_count) if node_count > 0 else 2.5
        pos = nx.spring_layout(graph, k=k_val, iterations=100, seed=42)
    except Exception as layout_error:
        logging.error(f"Błąd podczas obliczania układu spring_layout: {layout_error}. Używam random_layout.")
        try: pos = nx.random_layout(graph, seed=42)
        except Exception as random_layout_error:
             logging.error(f"Błąd podczas obliczania układu random_layout: {random_layout_error}.")
             return
    end_layout_time = time.time()
    logging.info(f"Układ Spring Layout obliczony w {end_layout_time - start_layout_time:.2f}s")

    if pos is None: logging.error("Nie udało się obliczyć pozycji węzłów."); return

    # 4. Rysowanie - większy rozmiar figury, mniejsze elementy
    fig, ax = plt.subplots(figsize=(24, 20)) # <<< WIĘKSZY ROZMIAR FIGURY
    plt.title(f"Wizualizacja Grafu Grup (Layout: Spring, Węzły=Stoły, Krawędzie=Relacje)", fontsize=16, pad=20)

    # ===>>> MNIEJSZE ROZMIARY <<<===
    node_size_param = max(30, 100 - node_count // 2)
    font_size_param = max(2, 5 - node_count // 30)

    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_size_param, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(graph, pos,
                           edgelist=edges_to_draw,
                           edge_color=edge_attribute_list['color'],
                           style=edge_attribute_list['style'],
                           width=edge_attribute_list['width'],
                           alpha=edge_attribute_list['alpha'],
                           ax=ax)
    if node_count < 75:
        nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=font_size_param, ax=ax)
    else:
        logging.info("Pominięto etykiety węzłów z powodu dużej liczby grup (>75).")


    # 5. Legenda
    legend_handles = [
        patches.Patch(color=relation_styles['Conflict']['color'], label='Konflikt'),
        patches.Patch(color=relation_styles['PreferSitWith']['color'], label='Preferencja Razem'),
        patches.Patch(color=relation_styles['PreferNotSitWith']['color'], label='Preferencja Osobno'),
        patches.Patch(color=relation_styles['Neutral']['color'], label='Neutralna/Brak'),
        patches.Patch(color=unassigned_color, label='Nieprzypisany')
    ]
    ax.legend(handles=legend_handles, loc='best', title="Typ Relacji/Stanu", fontsize=8)
    plt.axis('off')

    # 6. Zapisz plik
    try:
        results_dir = os.path.dirname(filename)
        if results_dir and not os.path.exists(results_dir):
            os.makedirs(results_dir)
        plt.savefig(filename, dpi=200, bbox_inches='tight');
        logging.info(f"Zapisano wizualizację grafu grup: {filename}")
    except Exception as e: logging.exception(f"BŁĄD podczas zapisywania wizualizacji grafu grup: {e}")
    finally:
        if 'fig' in locals() and fig is not None: plt.close(fig)


# --- Funkcja-Orkiestrator Wywoływana przez API (wersja z DB/Generator) ---
def run_solve_process(params: dict):
    """
    Główna funkcja uruchamiająca proces WSP. Obsługuje tryb DB ('event_id') lub generowania ('num_guests').
    Zapisuje wynik do bazy danych, jeśli event_id jest podany.
    Zwraca krotkę: (słownik_wyników, obiekt_grafu, obiekt_przypisania) lub (słownik_błędu, None, None).
    """
    config = load_config()
    if config is None:
        logging.warning("Nie można załadować konfiguracji. Używanie fallbacków.")
        fallback_config = {
            'DataGeneration': {'couple_ratio': '0.3', 'family_ratio': '0.1', 'conflict_prob': '0.1', 'prefer_like_prob': '0.1', 'prefer_dislike_prob': '0.1'},
            'SeatingParameters': {'table_capacity': '10', 'balance_weight': '0.5', 'weight_prefer_with': '-5', 'weight_prefer_not_with': '3', 'weight_conflict': '1000', 'table_estimation_factor': '2.5'},
            'TabuSearch': {'max_iterations': '500', 'tabu_tenure': '7', 'no_improvement_stop': '50'}
        }
        config = configparser.ConfigParser(); config.read_dict(fallback_config)

    group_graph = None; assignment = None
    db = SessionLocal()
    if db is None: return {"error": "Nie można utworzyć sesji bazy danych.", "status": "failure"}, None, None

    try:
        # --- Pobieranie Parametrów Algorytmów (wspólne) ---
        algorithm_choice = params.get('algorithm', 'Tabu Search')
        TABLE_CAPACITY = params.get('table_capacity', config.getint('SeatingParameters', 'table_capacity', fallback=10))
        BALANCE_WEIGHT = params.get('balance_weight', config.getfloat('SeatingParameters', 'balance_weight', fallback=0.5))
        MAX_ITERATIONS = params.get('ts_max_iterations', config.getint('TabuSearch', 'max_iterations', fallback=500))
        TABU_TENURE = params.get('ts_tabu_tenure', config.getint('TabuSearch', 'tabu_tenure', fallback=7))
        WEIGHT_PREFER_WITH = config.getint('SeatingParameters', 'weight_prefer_with', fallback=-5)
        WEIGHT_PREFER_NOT_WITH = config.getint('SeatingParameters', 'weight_prefer_not_with', fallback=3)
        WEIGHT_CONFLICT = config.getint('SeatingParameters', 'weight_conflict', fallback=1000)
        TABLE_ESTIMATION_FACTOR = config.getfloat('SeatingParameters', 'table_estimation_factor', fallback=2.5)
        NO_IMPROVEMENT_STOP = config.getint('TabuSearch', 'no_improvement_stop', fallback=50)

        # --- Decyzja: Baza danych czy Generowanie? ---
        event_id = params.get('event_id')
        num_guests_param = params.get('num_guests')
        num_guests = 0 # Zostanie ustawione
        num_groups = 0 # Zostanie ustawione
        event_name = "Dane generowane"

        if event_id is not None:
            # === Tryb Bazy Danych ===
            try: event_id = int(event_id)
            except ValueError: return {"error": "'event_id' musi być liczbą."}, None, None
            logging.info(f"Tryb BAZY DANYCH: Odczytywanie danych dla event_id={event_id}...")
            event = db.query(Event).filter(Event.id == event_id).first()
            if not event: return {"error": f"Nie znaleziono wydarzenia ID={event_id}."}, None, None
            event_name = event.name
            groups_from_db = db.query(Group).filter(Group.event_id == event_id).all()
            if not groups_from_db: return {"error": f"Brak grup dla event_id={event_id}."}, None, None
            relationships_from_db = db.query(Relationship).filter(Relationship.event_id == event_id).all()
            logging.info(f"Odczytano {len(groups_from_db)} grup i {len(relationships_from_db)} relacji dla '{event.name}'.")

            # Budowanie grafu z danych DB
            group_graph = nx.Graph()
            for group in groups_from_db:
                group_size = group.size or 1
                group_graph.add_node(group.id, size=group_size, guest_names=group.guest_names)
                num_guests += group_size
            num_groups = group_graph.number_of_nodes()
            added_edges = 0
            for rel in relationships_from_db:
                if rel.group1_id in group_graph and rel.group2_id in group_graph:
                    group_graph.add_edge(rel.group1_id, rel.group2_id, weight=rel.weight, conflict=(rel.rel_type == 'Conflict'), type=rel.rel_type)
                    added_edges += 1
            logging.info(f"Zbudowano graf z DB: Węzły={num_groups}, Krawędzie={added_edges}, Goście={num_guests}")

        elif num_guests_param is not None:
            # === Tryb Generowania Danych ===
            try: num_guests = int(num_guests_param)
            except ValueError: return {"error": "'num_guests' musi być liczbą."}, None, None
            logging.info(f"Tryb GENEROWANIA: Generowanie danych dla {num_guests} gości...")
            COUPLE_RATIO = config.getfloat('DataGeneration', 'couple_ratio', fallback=0.3)
            FAMILY_RATIO = config.getfloat('DataGeneration', 'family_ratio', fallback=0.1)
            CONFLICT_PROB = config.getfloat('DataGeneration', 'conflict_prob', fallback=0.1)
            PREFER_LIKE_PROB = config.getfloat('DataGeneration', 'prefer_like_prob', fallback=0.1)
            PREFER_DISLIKE_PROB = config.getfloat('DataGeneration', 'prefer_dislike_prob', fallback=0.1)
            guest_df, relationships, num_groups_gen = generate_data(
                num_guests=num_guests, couple_ratio=COUPLE_RATIO, family_ratio=FAMILY_RATIO,
                conflict_prob=CONFLICT_PROB, prefer_like_prob=PREFER_LIKE_PROB, prefer_dislike_prob=PREFER_DISLIKE_PROB,
                WEIGHT_PREFER_WITH=WEIGHT_PREFER_WITH, WEIGHT_PREFER_NOT_WITH=WEIGHT_PREFER_NOT_WITH, WEIGHT_CONFLICT=WEIGHT_CONFLICT
            )
            if guest_df is None: return {"error": "Nie udało się wygenerować danych."}, None, None
            group_graph = build_group_graph(guest_df, relationships, WEIGHT_PREFER_WITH, WEIGHT_PREFER_NOT_WITH, WEIGHT_CONFLICT)
            if group_graph is None: return {"error": "Nie udało się zbudować grafu."}, None, None
            num_groups = group_graph.number_of_nodes()
            num_guests = sum(data.get('size', 1) for node, data in group_graph.nodes(data=True))

        else:
            return {"error": "Należy podać 'event_id' lub 'num_guests'."}, None, None

        # --- Wspólna część dla obu trybów ---
        if group_graph is None or group_graph.number_of_nodes() == 0:
             return {"error": "Graf nie został poprawnie utworzony."}, None, None

        num_tables_try = max(math.ceil(num_groups / (TABLE_CAPACITY / TABLE_ESTIMATION_FACTOR)), math.ceil(num_guests / TABLE_CAPACITY), 2);
        logging.info(f"Szacowana liczba stołów: {num_tables_try}")

        final_score = float('inf'); final_conflicts = -1
        start_algo_time = time.time()

        # Wybór i uruchomienie algorytmu
        if algorithm_choice == 'Greedy':
            assignment = greedy_seating(group_graph, num_tables_try, TABLE_CAPACITY, WEIGHT_CONFLICT)
        elif algorithm_choice == 'DSatur':
            assignment = color_graph_dsatur(group_graph, num_tables_try, TABLE_CAPACITY)
        elif algorithm_choice == 'Tabu Search':
             initial_assignment_ts = color_graph_dsatur(group_graph, num_tables_try, TABLE_CAPACITY)
             if initial_assignment_ts:
                 init_score, init_conflicts = calculate_seating_score(group_graph, initial_assignment_ts, num_tables_try, BALANCE_WEIGHT, WEIGHT_CONFLICT)
                 if init_conflicts == 0:
                     logging.info("Uruchamianie TS z DSatur.")
                     assignment, final_score_ts = tabu_search_seating(graph=group_graph, num_tables=num_tables_try, table_capacity=TABLE_CAPACITY, initial_assignment=initial_assignment_ts, max_iterations=MAX_ITERATIONS, tabu_tenure=TABU_TENURE, balance_weight=BALANCE_WEIGHT, WEIGHT_CONFLICT=WEIGHT_CONFLICT, NO_IMPROVEMENT_STOP=NO_IMPROVEMENT_STOP)
                 else:
                     logging.warning(f"Start DSatur miał {init_conflicts} konfliktów. Zwracam DSatur."); assignment = initial_assignment_ts
             else:
                 logging.warning("DSatur nie dał rozwiązania. Próbuję Greedy dla TS.")
                 initial_assignment_ts = greedy_seating(group_graph, num_tables_try, TABLE_CAPACITY, WEIGHT_CONFLICT)
                 if initial_assignment_ts:
                     init_score, init_conflicts = calculate_seating_score(group_graph, initial_assignment_ts, num_tables_try, BALANCE_WEIGHT, WEIGHT_CONFLICT)
                     if init_conflicts == 0:
                         logging.info("Uruchamianie TS z Greedy."); assignment, final_score_ts = tabu_search_seating(graph=group_graph, num_tables=num_tables_try, table_capacity=TABLE_CAPACITY, initial_assignment=initial_assignment_ts, max_iterations=MAX_ITERATIONS, tabu_tenure=TABU_TENURE, balance_weight=BALANCE_WEIGHT, WEIGHT_CONFLICT=WEIGHT_CONFLICT, NO_IMPROVEMENT_STOP=NO_IMPROVEMENT_STOP)
                     else:
                         logging.warning(f"Start Greedy miał {init_conflicts} konfliktów. Zwracam Greedy."); assignment = initial_assignment_ts
                 else: logging.error("Ani DSatur, ani Greedy nie dały startu dla TS."); return {"error": "Brak rozwiązania startowego dla TS."}, group_graph, None
        else: return {"error": f"Nieznany algorytm: {algorithm_choice}"}, group_graph, None

        end_algo_time = time.time()
        logging.info(f"Algorytm {algorithm_choice} zakończony w {end_algo_time - start_algo_time:.4f}s")

        if assignment is None: return {"error": f"Algorytm {algorithm_choice} nie zwrócił przypisania."}, group_graph, None

        final_score, final_conflicts = calculate_seating_score(graph=group_graph, assignment=assignment, num_tables=num_tables_try, balance_weight=BALANCE_WEIGHT, WEIGHT_CONFLICT=WEIGHT_CONFLICT)
        if final_conflicts == -1: return {"error": f"Błąd oceny wyniku algorytmu {algorithm_choice}."}, group_graph, assignment

        # --- Przygotowanie Wyniku ---
        actual_tables_used = set(assignment.values()) - {0}
        num_actual_tables = len(actual_tables_used)
        group_graph_path = None
        status = "success"

        # --- Przygotowanie Słownika Wyników ---
        results_summary = {
            "event_id": event_id, "event_name": event_name,
            "algorithm": algorithm_choice,
            "num_guests_processed": num_guests,
            "num_groups": num_groups,
            "num_tables_used": num_actual_tables,
            "table_capacity": TABLE_CAPACITY,
            "score": final_score if final_score != float('inf') else None,
            "conflicts": final_conflicts,
            "status": status,
            "group_graph_path": group_graph_path
        }

        # --- Opcjonalna Wizualizacja Grafu (po stworzeniu results_summary) ---
        if params.get('generate_graph_viz', False) and MATPLOTLIB_AVAILABLE:
            results_dir = 'results'; os.makedirs(results_dir, exist_ok=True)
            timestamp = int(time.time()); algo_safe = "".join(c if c.isalnum() else "_" for c in algorithm_choice)
            event_suffix = f"ev{event_id}" if event_id is not None else f"{num_guests}g"
            graph_filename = os.path.join(results_dir, f"dashboard_group_graph_{algo_safe}_{event_suffix}_{timestamp}.png")
            visualize_group_graph(
                assignment=assignment, graph=group_graph, num_tables=num_tables_try,
                filename=graph_filename,
                WEIGHT_PREFER_WITH=WEIGHT_PREFER_WITH, WEIGHT_PREFER_NOT_WITH=WEIGHT_PREFER_NOT_WITH, WEIGHT_CONFLICT=WEIGHT_CONFLICT
            )
            if os.path.exists(graph_filename):
                 results_summary["group_graph_path"] = graph_filename

        # ===>>> POCZĄTEK BLOKU ZAPISU DO BAZY DANYCH <<<===
        logging.info("Zapisywanie wyniku do bazy danych...")
        try:

            params_to_save = {k:v for k,v in params.items() if k not in ['event_id', 'num_guests', 'generate_graph_viz']}
            assignment_to_save = {str(k): v for k, v in assignment.items()} if assignment else None

            db_event_id = event_id if event_id is not None else -1

            new_result = AssignmentResult(
                event_id=db_event_id,
                run_timestamp=datetime.datetime.utcnow(),
                algorithm=results_summary.get("algorithm"),
                parameters_json=json.dumps(params_to_save) if params_to_save else None,
                score=results_summary.get("score"),
                conflicts=results_summary.get("conflicts"),
                assignment_json=json.dumps(assignment_to_save) if assignment_to_save else None,
                status=results_summary.get("status", "failure")
            )
            db.add(new_result)
            db.commit()
            logging.info(f"Zapisano wynik (ID: {new_result.id}) dla event_id: {db_event_id}")
        except Exception as save_e:
            logging.exception(f"Nie udało się zapisać wyniku do bazy danych: {save_e}")
            db.rollback()
        # ===>>> KONIEC BLOKU ZAPISU DO BAZY DANYCH <<<===

        logging.info(f"Zakończono proces dla event_id={event_id if event_id is not None else 'Generowane'}. Wynik: {results_summary.get('score')}, Konflikty: {results_summary.get('conflicts')}")
        return results_summary, group_graph, assignment

    except Exception as e:
        logging.exception(f"Wystąpił nieoczekiwany błąd w run_solve_process dla params={params}: {e}")
        db.rollback()
        return {"error": f"Wystąpił wewnętrzny błąd serwera: {e}", "status": "failure"}, group_graph, assignment
    finally:
        if db: db.close() 

# koniec pliku: src/solver.py
