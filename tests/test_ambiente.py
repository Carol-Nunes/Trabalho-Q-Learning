import heapq

import pytest


# Função que verifica o formato do mapa e a posição dos elementos
# descritos no README e no notebook.
def test_mapa_tem_formato_e_elementos_esperados(config):

    mapa_labirinto = config["mapa_labirinto"]

    assert len(mapa_labirinto) == 7

    for linha in mapa_labirinto:

        assert len(linha) == 9

        assert set(linha) <= {"#", ".", "S", "E", "T"}

    linha_inicial, coluna_inicial = config["posicao_inicial"]
    linha_saida, coluna_saida = config["posicao_saida"]

    assert config["posicao_inicial"] == (1, 1)
    assert config["posicao_saida"] == (1, 7)

    assert mapa_labirinto[linha_inicial][coluna_inicial] == "S"
    assert mapa_labirinto[linha_saida][coluna_saida] == "E"

    armadilhas = [(l, c) for l in range(7) for c in range(9) if mapa_labirinto[l][c] == "T"]

    assert armadilhas == config["posicao_armadilhas"]


# Função que verifica se as bordas do mapa são paredes.
def test_bordas_do_mapa_sao_paredes(config):

    mapa_labirinto = config["mapa_labirinto"]

    assert set(mapa_labirinto[0]) == {"#"}
    assert set(mapa_labirinto[-1]) == {"#"}

    for linha in mapa_labirinto:

        assert linha[0] == "#"
        assert linha[-1] == "#"


# Função que testa o cálculo da posição candidata para cada ação.
@pytest.mark.parametrize("acao, esperada", [

    ("cima", (2, 3)),
    ("baixo", (4, 3)),
    ("direita", (3, 4)),
    ("esquerda", (3, 2)),
    ("invalida", (3, 3)),
])
def test_calcular_posicao_candidata(nb, acao, esperada):

    assert nb["calcular_posicao_candidata"]((3, 3), acao) == esperada


# Função que testa os limites do mapa, incluindo índices negativos.
def test_posicao_eh_valida(nb, config):

    mapa_labirinto = config["mapa_labirinto"]

    assert nb["posicao_eh_valida"]((0, 0), mapa_labirinto)
    assert nb["posicao_eh_valida"]((6, 8), mapa_labirinto)

    assert not nb["posicao_eh_valida"]((-1, 0), mapa_labirinto)
    assert not nb["posicao_eh_valida"]((0, -1), mapa_labirinto)
    assert not nb["posicao_eh_valida"]((7, 0), mapa_labirinto)
    assert not nb["posicao_eh_valida"]((0, 9), mapa_labirinto)


# Função que testa a identificação de paredes, armadilhas e saída.
def test_identifica_tipos_de_celula(nb, config):

    mapa_labirinto = config["mapa_labirinto"]

    assert nb["posicao_eh_parede"]((0, 0), mapa_labirinto)
    assert not nb["posicao_eh_parede"]((1, 2), mapa_labirinto)

    assert nb["posicao_eh_armadilha"]((2, 4), mapa_labirinto)
    assert nb["posicao_eh_armadilha"]((5, 5), mapa_labirinto)
    assert not nb["posicao_eh_armadilha"]((1, 2), mapa_labirinto)

    assert nb["posicao_eh_saida"]((1, 7), mapa_labirinto)
    assert not nb["posicao_eh_saida"]((1, 1), mapa_labirinto)


# Função que testa a nova posição, a recompensa e o fim do episódio.
@pytest.mark.parametrize("posicao, acao, esperado", [

    ((1, 1), "direita", ((1, 2), -1, False)),   # Caminho livre
    ((1, 2), "esquerda", ((1, 1), -1, False)),  # Volta ao início
    ((1, 1), "cima", ((1, 1), -3, False)),      # Parede
    ((1, 4), "direita", ((1, 4), -3, False)),   # Parede entre o início e a saída
    ((1, 4), "baixo", ((2, 4), -10, False)),    # Armadilha
    ((5, 6), "esquerda", ((5, 5), -10, False)), # Armadilha
    ((2, 4), "baixo", ((3, 4), -1, False)),     # Sai da armadilha e continua
    ((1, 6), "direita", ((1, 7), 30, True)),    # Saída
])
def test_movimentar(nb, config, posicao, acao, esperado):

    assert nb["movimentar"](posicao, acao, config["mapa_labirinto"]) == esperado


# Função que testa a colisão com o limite do mapa, usando um mapa sem
# paredes na borda.
def test_colisao_com_limite_do_mapa(nb):

    mapa_sem_borda = [["S", "."], [".", "E"]]

    assert nb["movimentar"]((0, 0), "cima", mapa_sem_borda) == ((0, 0), -3, False)
    assert nb["movimentar"]((0, 0), "esquerda", mapa_sem_borda) == ((0, 0), -3, False)
    assert nb["movimentar"]((0, 1), "baixo", mapa_sem_borda) == ((1, 1), 30, True)


# Função que calcula a maior pontuação possível com o algoritmo de Dijkstra.
# O custo de entrar em uma célula é o oposto da recompensa recebida, e a
# chegada à saída soma +30. Colisões só reduzem a pontuação e são ignoradas.
def calcular_melhor_pontuacao(mapa_labirinto, posicao_inicial, posicao_saida, movimentar, acoes):

    fila = [(0, 0, posicao_inicial)]

    menor_custo = {posicao_inicial: 0}

    melhor = None

    while fila:

        custo, movimentos, posicao = heapq.heappop(fila)

        if custo > menor_custo.get(posicao, float("inf")):

            continue

        for acao in acoes:

            proxima_posicao, recompensa, chegou_saida = movimentar(posicao, acao, mapa_labirinto)

            if proxima_posicao == posicao:

                continue

            if chegou_saida:

                candidato = (-(custo - recompensa), movimentos + 1)

                if melhor is None or candidato[0] > melhor[0]:

                    melhor = candidato

                continue

            novo_custo = custo - recompensa

            if novo_custo < menor_custo.get(proxima_posicao, float("inf")):

                menor_custo[proxima_posicao] = novo_custo

                heapq.heappush(fila, (novo_custo, movimentos + 1, proxima_posicao))

    return melhor


# Função que verifica que a saída é alcançável e que 17 pontos em
# 14 movimentos é a maior pontuação possível no labirinto, como
# obtido pelo modelo ilustrado.
def test_melhor_pontuacao_possivel_eh_17(nb, config):

    melhor = calcular_melhor_pontuacao(config["mapa_labirinto"], config["posicao_inicial"], config["posicao_saida"], nb["movimentar"], config["acoes_disponiveis"])

    assert melhor == (17, 14)


# Função que verifica que o menor caminho em movimentos (10) passa
# por uma armadilha e vale 12 pontos, como discutido no notebook.
def test_caminho_mais_curto_passa_pela_armadilha(nb, config):

    mapa_labirinto = config["mapa_labirinto"]

    caminho = [(1, 1), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4), (3, 5), (3, 6), (2, 6), (1, 6), (1, 7)]

    acoes = {(-1, 0): "cima", (1, 0): "baixo", (0, 1): "direita", (0, -1): "esquerda"}

    pontuacao = 0

    for atual, proxima in zip(caminho, caminho[1:]):

        acao = acoes[(proxima[0] - atual[0], proxima[1] - atual[1])]

        nova_posicao, recompensa, chegou_saida = nb["movimentar"](atual, acao, mapa_labirinto)

        assert nova_posicao == proxima

        pontuacao += recompensa

    assert chegou_saida
    assert len(caminho) - 1 == 10
    assert pontuacao == 12

    # Nenhum caminho até a saída tem menos de 10 movimentos.
    visitados = {config["posicao_inicial"]: 0}

    fronteira = [config["posicao_inicial"]]

    while fronteira:

        proxima_fronteira = []

        for posicao in fronteira:

            for acao in config["acoes_disponiveis"]:

                nova_posicao, _, _ = nb["movimentar"](posicao, acao, mapa_labirinto)

                if nova_posicao not in visitados:

                    visitados[nova_posicao] = visitados[posicao] + 1

                    proxima_fronteira.append(nova_posicao)

        fronteira = proxima_fronteira

    assert visitados[config["posicao_saida"]] == 10
