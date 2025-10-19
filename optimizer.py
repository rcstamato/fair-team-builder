import random

import pandas as pd
from ortools.linear_solver import pywraplp
from itertools import chain


def separar_reservas(jogadores):
    num_reservas = len(jogadores) % 4
    if num_reservas == 0:
        return jogadores, []

    jogadores_sorted = jogadores.sort_values("score", ascending=False)
    titulares = jogadores_sorted[:-num_reservas]
    reservas = jogadores_sorted[-num_reservas:]

    return titulares, reservas


def adicionar_reservas(times, reservas, restricoes):
    random.shuffle(times)

    for _i, reserva in reservas.iterrows():
        allocated = False
        for time in times:
            if len(time) > 4: # Já tem número máximo de jogadores
                continue
            if can_be_placed_in_team(reserva, time, restricoes):
                time.append(reserva[["id", "score"]].astype(int))
                allocated = True
                break

        if not allocated:
            raise RuntimeError("Reserva {} não pode ser alocado em nenhum time".format(reserva))


def can_be_placed_in_team(reserva, time, restricoes):
    reserva_id = reserva[0]
    restricoes_reserva = list(filter(lambda x: x[0] == reserva_id or x[1] == reserva_id, restricoes))
    parceiros_restritos = list(filter(lambda x: x != reserva_id, list(chain.from_iterable(restricoes_reserva))))
    jogadores_time = [x[0] for x in time]

    if len(set(parceiros_restritos) & set(jogadores_time)) > 0:
        return False

    return True


def processar(jogadores_df, restricoes):
    jogadores_df["id"] = jogadores_df.index
    titulares_df, reservas_df = separar_reservas(jogadores_df)

    titulares = [tuple(row) for row in titulares_df[["id", "score", "sexo"]].itertuples(index=False)]
    incompatibilidades = [(row[0], row[2]) for row in restricoes.itertuples(index=False)]

    times, metricas = formar_quartetos_balanceados(
        titulares,
        peso_range=0.0,
        incompatibilidades=incompatibilidades
    )

    if len(reservas_df) > 0:
        adicionar_reservas(times, reservas_df, incompatibilidades)

    return times, metricas


def formar_quartetos_balanceados(
    jogadores,
    peso_range=0.0,
    incompatibilidades=None   # lista de pares:  [(idA, idB), (idC, idD), ...]
):
    """
    Monta quartetos minimizando o desvio absoluto total das somas por time
    em relação à média-alvo.

    Parâmetros:
      - jogadores: lista de tuplas (id, nota, sexo), ex.: [(1, 8, m), (2, 5, f), ...]
      - peso_range: peso >= 0 para tie-break (max_score - min_score) no objetivo
      - incompatibilidades: lista de tuplas/pares de ids (não podem estar juntos)

    Retorna:
      - lista de times; cada time é uma lista de (id, nota)
      - dicionário com métricas
    """
    incompatibilidades = incompatibilidades or []

    n = len(jogadores)
    assert n % 4 == 0, "O número de jogadores precisa ser múltiplo de 4."
    num_times = n // 4

    ids = [j[0] for j in jogadores]
    notas = [j[1] for j in jogadores]
    homem = [j[2]=="m" for j in jogadores]
    mulher = [j[2]=="f" for j in jogadores]
    id_to_idx = {pid: i for i, pid in enumerate(ids)}

    # Valida entradas de afinidade/incompatibilidade
    # for a, b in incompatibilidades:
    #     if a not in id_to_idx or b not in id_to_idx:
    #         raise ValueError(f"Par de incompatibilidade ({a}, {b}) contém ID não listado em 'jogadores'.")

    soma_total = sum(notas)
    media_alvo = soma_total / num_times  # soma por time desejada

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("Não foi possível criar o solver SCIP (OR-Tools).")

    # Variáveis binárias x[i,t] = 1 se jogador i está no time t
    x = {}
    for i in range(n):
        for t in range(num_times):
            x[i, t] = solver.BoolVar(f"x_{i}_{t}")

    # Cada jogador em exatamente um time
    for i in range(n):
        solver.Add(sum(x[i, t] for t in range(num_times)) == 1)

    # Cada time com exatamente 4 jogadores
    for t in range(num_times):
        solver.Add(sum(x[i, t] for i in range(n)) == 4)

    # Cada time com pelo menos 1 mulher
    for t in range(num_times):
        solver.Add(sum(x[i, t] * mulher[i] for i in range(n)) >= 1)

    # Cada time com pelo menos 1 homem
    for t in range(num_times):
        solver.Add(sum(x[i, t] * homem[i] for i in range(n)) >= 1)

    # Soma de pontos por time
    team_score = []
    for t in range(num_times):
        score_t = solver.Sum(x[i, t] * notas[i] for i in range(n))
        team_score.append(score_t)

    # Desvio absoluto em relação à média-alvo
    d = []
    for t in range(num_times):
        d_t = solver.NumVar(0, solver.infinity(), f"d_{t}")
        d.append(d_t)
        solver.Add(d_t >= team_score[t] - media_alvo)
        solver.Add(d_t >= media_alvo - team_score[t])

    # Tie-break opcional: range
    max_score = solver.NumVar(0, solver.infinity(), "max_score")
    min_score = solver.NumVar(0, solver.infinity(), "min_score")
    for t in range(num_times):
        solver.Add(team_score[t] <= max_score)
        solver.Add(team_score[t] >= min_score)

    # ==========
    # Restrições extra
    # ==========

    # 1) Incompatibilidades (pares NÃO PODEM ficar juntos)
    # Para cada par (a,b) e cada time t: x[a,t] + x[b,t] <= 1
    for a, b in incompatibilidades:
        ia = id_to_idx.get(a)
        ib = id_to_idx.get(b)
        if ia is None or ib is None:
            continue

        for t in range(num_times):
            solver.Add(x[ia, t] + x[ib, t] <= 1)

    # Objetivo: minimizar sum(d_t) + peso_range*(max - min)
    objective = solver.Sum(d)
    if peso_range > 0:
        objective = objective + peso_range * (max_score - min_score)
    solver.Minimize(objective)

    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError("Não foi encontrada solução ótima (verifique se as restrições tornam o problema inviável).")

    # Monta a solução
    times = []
    for t in range(num_times):
        membros = [(ids[i], notas[i]) for i in range(n) if x[i, t].solution_value() > 0.5]
        times.append(membros)

    metricas = {
        "media_alvo_por_time": media_alvo,
        "desvios_absolutos": [d[t].solution_value() for t in range(num_times)],
        "somas_por_time": [sum(n for _, n in times[t]) for t in range(num_times)],
        "range": (max_score.solution_value() - min_score.solution_value()),
        "objetivo": solver.Objective().Value(),
    }
    return times, metricas



def main():
    jogadores_df = pd.read_csv("quartetos-jogadores.csv")
    restricoes = pd.read_csv("quartetos-restricoes.csv")

    jogadores_df.set_index("id", inplace=True)
    processar(jogadores_df, restricoes)


if __name__ == "__main__":
    main()