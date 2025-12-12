# -----------------------------------------------------------
# SIMULACIÓN DE RESPONDENT-DRIVEN SAMPLING (RDS)
# Estimador RDS-II, gráficos, cálculos y tablas
# -----------------------------------------------------------

import networkx as nx
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque, defaultdict

# ------------------------------------------
# 1. PARÁMETROS (PUEDES MODIFICARLOS)
# ------------------------------------------
N = 1500                # tamaño de la población
m_links = 3             # parámetro del modelo Barabási-Albert
seeds = 6               # número de semillas
coupons = 3             # cupones por participante
target_sample = 500     # tamaño objetivo de la muestra
p_attr = 0.20           # prevalencia del atributo
reported_noise = 0.15   # ruido en grado reportado

random.seed(42)
np.random.seed(42)

# ------------------------------------------
# 2. CREAR RED POBLACIONAL
# ------------------------------------------
G = nx.barabasi_albert_graph(N, m_links, seed=42)

# Atributo (variable binaria)
attr = {i: (1 if random.random() < p_attr else 0) for i in G.nodes()}
nx.set_node_attributes(G, attr, 'attr')

# Grado verdadero
true_degree = dict(G.degree())
nx.set_node_attributes(G, true_degree, 'degree')

# ------------------------------------------
# 3. FUNCIÓN PARA SIMULAR RDS
# ------------------------------------------
def simulate_rds(G, seeds, coupons, target_sample, reported_noise=0.0):

    nodes = list(G.nodes())
    if isinstance(seeds, int):
        seed_nodes = random.sample(nodes, seeds)
    else:
        seed_nodes = seeds

    sampled = []
    recruited_by = {}
    waves = {}
    coupons_left = defaultdict(lambda: coupons)
    q = deque()

    # Registrar semillas
    for s in seed_nodes:
        q.append(s)
        sampled.append(s)
        waves[s] = 0
        coupons_left[s] = coupons

    visited = set(sampled)

    # Proceso RDS
    while q and len(sampled) < target_sample:
        current = q.popleft()
        neighbors = [nbr for nbr in G.neighbors(current) if nbr not in visited]
        random.shuffle(neighbors)

        max_recruits = min(coupons_left[current], len(neighbors), target_sample - len(sampled))

        for nbr in neighbors[:max_recruits]:
            sampled.append(nbr)
            visited.add(nbr)
            recruited_by[nbr] = current
            waves[nbr] = waves[current] + 1
            coupons_left[nbr] = coupons
            q.append(nbr)

    # DataFrame del muestreo
    rows = []
    for idx, node in enumerate(sampled, start=1):
        td = G.nodes[node]['degree']
        if reported_noise > 0:
            noise = np.random.poisson(max(0, td * reported_noise))
            reported = max(1, td + noise)
        else:
            reported = max(1, td)

        rows.append({
            'order': idx,
            'node': node,
            'attr': G.nodes[node]['attr'],
            'true_degree': td,
            'reported_degree': reported,
            'wave': waves.get(node, None),
            'recruited_by': recruited_by.get(node, None)
        })

    df = pd.DataFrame(rows)
    return sampled, recruited_by, df

# Ejecutar simulación
sampled_nodes, recruited_by, sample_df = simulate_rds(G, seeds, coupons, target_sample, reported_noise)

# ------------------------------------------
# 4. ESTIMADOR RDS-II
# ------------------------------------------
def rds_II_estimator(df, attr_col='attr', degree_col='reported_degree'):
    df = df.copy()
    df['inv_deg'] = 1.0 / df[degree_col]
    return (df[attr_col] * df['inv_deg']).sum() / df['inv_deg'].sum()

rds_est = rds_II_estimator(sample_df)

true_prev = sum(attr.values()) / N
naive_prev = sample_df['attr'].mean()

# ------------------------------------------
# 5. CÁLCULO DE TAMAÑO DE MUESTRA
# ------------------------------------------
def sample_size_proportion(p=0.5, z=1.96, e=0.05, deff=2.0, nonresponse=0.20):
    n0 = (z*z * p * (1-p)) / (e*e)
    n1 = n0 * deff
    n_final = int(np.ceil(n1 / (1 - nonresponse)))
    return n0, n1, n_final

n0, n1, n_final = sample_size_proportion()

# ------------------------------------------
# 6. TABLA RESUMEN
# ------------------------------------------
summary = {
    'Tamaño población (N)': N,
    'Muestra objetivo': target_sample,
    'Muestra obtenida': len(sample_df),
    'Semillas': seeds,
    'Cupones': coupons,
    'Prevalencia real': true_prev,
    'Prevalencia naive': naive_prev,
    'Estimador RDS-II': rds_est,
    'n0 (simple)': int(np.ceil(n0)),
    'n con DEFF': int(np.ceil(n1)),
    'n final (ajuste NR)': n_final
}

print("\n----- RESUMEN DEL EXPERIMENTO -----")
for k, v in summary.items():
    print(f"{k}: {v}")

# ------------------------------------------
# 7. GRÁFICOS
# ------------------------------------------

# A) Subgrafo y nodos muestreados
plt.figure(figsize=(10,7))
sub = list(range(400))
H = G.subgraph(sub)
pos = nx.spring_layout(H, seed=42)
nx.draw_networkx_nodes(H, pos, node_size=20)
sampled_set = set(sample_df['node'])
highlight = [n for n in H.nodes() if n in sampled_set]
nx.draw_networkx_nodes(H, pos, nodelist=highlight, node_size=40)
nx.draw_networkx_edges(H, pos, alpha=0.2)
plt.title("Subgrafo de la red con nodos muestreados resaltados")
plt.axis("off")
plt.show()

# B) Árbol de reclutamiento
D = nx.DiGraph()
for child, parent in recruited_by.items():
    D.add_edge(parent, child)

plt.figure(figsize=(10,8))
pos_tree = nx.spring_layout(D, seed=24)
nx.draw(D, pos_tree, node_size=30, arrowsize=6)
plt.title("Árbol de reclutamiento RDS")
plt.axis("off")
plt.show()

# C) Distribución de grados población
plt.figure(figsize=(8,5))
deg_vals = [G.degree(n) for n in G.nodes()]
plt.hist(deg_vals, bins=30)
plt.title("Distribución del grado en la población")
plt.xlabel("Grado")
plt.ylabel("Frecuencia")
plt.show()

# D) Distribución del grado en muestra
plt.figure(figsize=(8,5))
plt.hist(sample_df['reported_degree'], bins=20)
plt.title("Distribución del grado reportado (muestra RDS)")
plt.xlabel("Grado reportado")
plt.ylabel("Frecuencia")
plt.show()

# E) Convergencia del estimador por oleadas
cumulative = []
for w in sorted(sample_df['wave'].dropna().unique()):
    df_w = sample_df[sample_df['wave'] <= w]
    cumulative.append({
        'wave': int(w),
        'n': len(df_w),
        'rdsII': rds_II_estimator(df_w),
        'naive': df_w['attr'].mean()
    })

cum_df = pd.DataFrame(cumulative)

plt.figure(figsize=(8,5))
plt.plot(cum_df['wave'], cum_df['rdsII'], marker='o')
plt.plot(cum_df['wave'], cum_df['naive'], marker='x')
plt.xlabel("Oleada")
plt.ylabel("Estimador")
plt.title("Convergencia del estimador por oleadas (RDS-II vs Naive)")
plt.grid(True)
plt.legend(["RDS-II", "Naive"])
plt.show()

