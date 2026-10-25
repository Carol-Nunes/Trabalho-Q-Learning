import numpy as np
import pytest


# Função que extrai as características usando a configuração do notebook.
def extrair(nb, config, posicao, acao):

    return nb["extrair_caracteristicas"](posicao, acao, config["mapa_labirinto"], config["posicao_saida"])


# Função que treina um agente com a configuração do notebook, permitindo
# alterar apenas os parâmetros necessários em cada teste.
def treinar(nb, config, alpha=0.01, seed=42, qtd_episodios=5, epsilon=None, limite_movimentos=None):

    return nb["treinar_agente"](
        config["mapa_labirinto"],
        alpha,
        config["gama"],
        config["epsilon"] if epsilon is None else epsilon,
        config["acoes_disponiveis"],
        config["limite_movimentos"] if limite_movimentos is None else limite_movimentos,
        config["posicao_inicial"],
        config["posicao_saida"],
        seed,
        qtd_episodios
    )


# Função que verifica o tamanho e o tipo do vetor de características.
def test_vetor_tem_17_caracteristicas(nb, config):

    vetor = extrair(nb, config, (1, 1), "direita")

    assert vetor.shape == (17,)
    assert vetor.dtype == np.float64

    # Bias
    assert vetor[0] == 1.0


# Função que verifica que, na colisão, a posição resultante continua
# sendo a posição atual.
def test_caracteristicas_de_colisao(nb, config):

    vetor = extrair(nb, config, (1, 1), "cima")

    assert vetor[1] == 1.0
    assert vetor[2] == 0.0
    assert vetor[3] == 0.0

    # Distância de (1, 1) até (1, 7), normalizada por 14.
    assert vetor[4] == pytest.approx(6 / 14)

    # A primeira base gaussiana é centrada em (1, 1).
    assert vetor[5] == pytest.approx(1.0)


# Função que verifica os indicadores de armadilha e de saída.
def test_caracteristicas_de_armadilha_e_saida(nb, config):

    armadilha = extrair(nb, config, (1, 4), "baixo")

    assert list(armadilha[1:4]) == [0.0, 1.0, 0.0]

    saida = extrair(nb, config, (1, 6), "direita")

    assert list(saida[1:4]) == [0.0, 0.0, 1.0]
    assert saida[4] == 0.0


# Função que verifica a distância Manhattan normalizada e o intervalo
# das bases gaussianas em todas as células livres do mapa.
def test_distancia_normalizada_e_bases_gaussianas(nb, config):

    mapa_labirinto = config["mapa_labirinto"]

    for linha in range(7):

        for coluna in range(9):

            if mapa_labirinto[linha][coluna] == "#":

                continue

            for acao in config["acoes_disponiveis"]:

                vetor = extrair(nb, config, (linha, coluna), acao)

                assert 0.0 <= vetor[4] <= 1.0

                assert np.all(vetor[5:] > 0.0)
                assert np.all(vetor[5:] <= 1.0)

    vetor = extrair(nb, config, (1, 1), "direita")

    # Posição resultante (1, 2): distância 5 até a saída.
    assert vetor[4] == pytest.approx(5 / 14)

    # Base centrada em (1, 1): distância quadrada 1.
    assert vetor[5] == pytest.approx(np.exp(-1 / 4))


# Função que verifica o produto escalar entre pesos e características.
def test_calcular_q(nb):

    pesos = np.array([1.0, -2.0, 0.5])
    caracteristicas = np.array([2.0, 1.0, 4.0])

    assert nb["calcular_q"](pesos, caracteristicas) == pytest.approx(2.0)


# Função que verifica que, com epsilon zero, o agente escolhe a ação
# de maior Q.
def test_escolher_acao_aproveita_maior_q(nb, config):

    pesos = np.zeros(17)

    # Peso alto para a característica de saída.
    pesos[3] = 100.0

    for seed in range(20):

        gerador = np.random.default_rng(seed)

        acao = nb["escolher_acao"]((1, 6), config["posicao_saida"], config["mapa_labirinto"], 0, config["acoes_disponiveis"], gerador, pesos)

        assert acao == "direita"


# Função que verifica que empates são resolvidos por sorteio apenas entre
# as ações empatadas.
def test_escolher_acao_sorteia_empates(nb, config):

    pesos = np.zeros(17)

    # Apenas a colisão é penalizada: direita e baixo empatam em (1, 1).
    pesos[1] = -1.0

    gerador = np.random.default_rng(0)

    escolhas = set()

    for _ in range(200):

        escolhas.add(str(nb["escolher_acao"]((1, 1), config["posicao_saida"], config["mapa_labirinto"], 0, config["acoes_disponiveis"], gerador, pesos)))

    assert escolhas == {"direita", "baixo"}


# Função que verifica que, com epsilon igual a um, o agente explora
# todas as ações, mesmo as piores.
def test_escolher_acao_explora_com_epsilon_um(nb, config):

    pesos = np.zeros(17)
    pesos[3] = 100.0

    gerador = np.random.default_rng(0)

    escolhas = set()

    for _ in range(200):

        escolhas.add(str(nb["escolher_acao"]((1, 6), config["posicao_saida"], config["mapa_labirinto"], 1, config["acoes_disponiveis"], gerador, pesos)))

    assert escolhas == set(config["acoes_disponiveis"])


# Função que verifica que alpha igual a zero mantém os pesos.
def test_alpha_zero_mantem_pesos(nb, config):

    pesos = np.arange(17, dtype=np.float64)

    caracteristicas = extrair(nb, config, (1, 1), "direita")

    novos_pesos = nb["atualizar_pesos"](pesos, caracteristicas, -1, (1, 2), False, config["mapa_labirinto"], 0.0, config["gama"], config["acoes_disponiveis"], config["posicao_saida"])

    assert np.array_equal(novos_pesos, pesos)


# Função que verifica que, ao chegar à saída, o alvo é apenas a recompensa.
def test_atualizacao_terminal_usa_apenas_recompensa(nb, config):

    pesos = np.full(17, 0.1)

    caracteristicas = extrair(nb, config, (1, 6), "direita")

    novos_pesos = nb["atualizar_pesos"](pesos, caracteristicas, 30, (1, 7), True, config["mapa_labirinto"], 0.5, config["gama"], config["acoes_disponiveis"], config["posicao_saida"])

    erro_temporal = 30 - np.dot(pesos, caracteristicas)

    assert np.allclose(novos_pesos, pesos + 0.5 * erro_temporal * caracteristicas)


# Função que verifica que, fora da saída, o alvo usa o maior Q futuro
# descontado por gama.
def test_atualizacao_nao_terminal_usa_maior_q_futuro(nb, config):

    pesos = np.linspace(-1, 1, 17)

    caracteristicas = extrair(nb, config, (1, 1), "direita")

    novos_pesos = nb["atualizar_pesos"](pesos, caracteristicas, -1, (1, 2), False, config["mapa_labirinto"], 0.1, config["gama"], config["acoes_disponiveis"], config["posicao_saida"])

    q_futuros = [np.dot(pesos, extrair(nb, config, (1, 2), acao)) for acao in config["acoes_disponiveis"]]

    erro_temporal = -1 + config["gama"] * max(q_futuros) - np.dot(pesos, caracteristicas)

    assert np.allclose(novos_pesos, pesos + 0.1 * erro_temporal * caracteristicas)


# Função que verifica que a atualização aproxima Q do alvo temporal
# e não altera o vetor de pesos recebido.
def test_atualizacao_reduz_erro_e_nao_altera_entrada(nb, config):

    pesos = np.zeros(17)
    copia = pesos.copy()

    caracteristicas = extrair(nb, config, (1, 6), "direita")

    novos_pesos = nb["atualizar_pesos"](pesos, caracteristicas, 30, (1, 7), True, config["mapa_labirinto"], 0.05, config["gama"], config["acoes_disponiveis"], config["posicao_saida"])

    erro_antes = abs(30 - np.dot(pesos, caracteristicas))
    erro_depois = abs(30 - np.dot(novos_pesos, caracteristicas))

    assert erro_depois < erro_antes
    assert np.array_equal(pesos, copia)


# Função que verifica que o episódio respeita o limite de movimentos.
def test_episodio_respeita_limite_de_movimentos(nb, config):

    pesos = np.zeros(17)

    gerador = np.random.default_rng(0)

    _, pontuacao, movimentos, chegou_saida = nb["treinar_episodio"](config["mapa_labirinto"], pesos, 0.01, config["gama"], config["epsilon"], config["acoes_disponiveis"], gerador, 5, config["posicao_inicial"], config["posicao_saida"])

    # Com 5 movimentos não é possível chegar à saída (são necessários 10).
    assert movimentos == 5
    assert not chegou_saida

    # Cada movimento vale -1, -3 ou -10.
    assert -50 <= pontuacao <= -5


# Função que verifica que o episódio termina ao chegar à saída.
def test_episodio_termina_ao_chegar_na_saida(nb, config):

    pesos = np.zeros(17)
    pesos[3] = 100.0

    gerador = np.random.default_rng(0)

    _, pontuacao, movimentos, chegou_saida = nb["treinar_episodio"](config["mapa_labirinto"], pesos, 0.01, config["gama"], 0, config["acoes_disponiveis"], gerador, 150, (1, 6), config["posicao_saida"])

    assert (pontuacao, movimentos, chegou_saida) == (30, 1, True)


# Função que verifica o histórico e a reprodutibilidade do treinamento.
def test_treinamento_eh_reprodutivel_com_mesma_semente(nb, config):

    pesos_a, historico_a = treinar(nb, config, seed=42)
    pesos_b, historico_b = treinar(nb, config, seed=42)
    pesos_c, historico_c = treinar(nb, config, seed=43)

    assert np.array_equal(pesos_a, pesos_b)
    assert historico_a == historico_b

    assert not np.array_equal(pesos_a, pesos_c)

    assert [registro["episodio"] for registro in historico_a] == [1, 2, 3, 4, 5]

    for registro in historico_a:

        assert set(registro) == {"episodio", "pontuacao", "movimentos", "chegou_saida"}

        assert registro["movimentos"] <= config["limite_movimentos"]


# Função que verifica que, sem episódios, os pesos continuam zerados.
def test_treinamento_comeca_com_pesos_zerados(nb, config):

    pesos, historico = treinar(nb, config, qtd_episodios=0)

    assert np.array_equal(pesos, np.zeros(17))
    assert historico == []


# Função que verifica que a avaliação não altera os pesos e que o caminho
# percorrido é contínuo e nunca atravessa paredes.
def test_avaliacao_mantem_pesos_e_gera_caminho_valido(nb, config):

    pesos, _ = treinar(nb, config, alpha=0.05, qtd_episodios=50)

    copia = pesos.copy()

    gerador = np.random.default_rng(123)

    pontuacao, movimentos, chegou_saida, caminho = nb["avaliar_agente"](pesos, config["mapa_labirinto"], config["posicao_inicial"], config["posicao_saida"], config["acoes_disponiveis"], gerador, config["limite_movimentos"])

    assert np.array_equal(pesos, copia)

    assert caminho[0] == config["posicao_inicial"]
    assert len(caminho) == movimentos + 1

    for atual, proxima in zip(caminho, caminho[1:]):

        assert abs(atual[0] - proxima[0]) + abs(atual[1] - proxima[1]) <= 1

        assert config["mapa_labirinto"][proxima[0]][proxima[1]] != "#"

    if chegou_saida:

        assert caminho[-1] == config["posicao_saida"]

    else:

        assert movimentos == config["limite_movimentos"]


# Função que verifica que a avaliação é determinística para a mesma semente.
def test_avaliacao_eh_reprodutivel(nb, config):

    pesos, _ = treinar(nb, config, qtd_episodios=20)

    resultados = []

    for _ in range(2):

        gerador = np.random.default_rng(123)

        resultados.append(nb["avaliar_agente"](pesos, config["mapa_labirinto"], config["posicao_inicial"], config["posicao_saida"], config["acoes_disponiveis"], gerador, config["limite_movimentos"]))

    assert resultados[0] == resultados[1]
