import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


CAMINHO_README = os.path.join(os.path.dirname(__file__), "..", "README.md")

# Resultados registrados na saída do notebook:
# (parâmetro, semente, pontuação acumulada, movimentos, chegou à saída)
RESULTADOS_ALPHAS = [

    (0.005, 42, -150, 150, False),
    (0.005, 43, -150, 150, False),
    (0.005, 44, -150, 150, False),
    (0.01, 42, 17, 14, True),
    (0.01, 43, -150, 150, False),
    (0.01, 44, 17, 14, True),
    (0.05, 42, 17, 14, True),
    (0.05, 43, 12, 10, True),
    (0.05, 44, 12, 10, True),
]

RESULTADOS_EPISODIOS = [

    (100, 42, -150, 150, False),
    (100, 43, -150, 150, False),
    (100, 44, -150, 150, False),
    (500, 42, 17, 14, True),
    (500, 43, -150, 150, False),
    (500, 44, 17, 14, True),
    (1000, 42, 17, 14, True),
    (1000, 43, -150, 150, False),
    (1000, 44, 17, 14, True),
]


# Função que converte os resultados esperados em um DataFrame no mesmo
# formato produzido pelas funções de comparação do notebook.
def criar_dataframe(resultados, parametro):

    registros = []

    for valor, seed, pontuacao, movimentos, chegou_saida in resultados:

        registros.append({

            parametro: valor,
            "seed": seed,
            "pontuacao_acumulada": pontuacao,
            "qtd_movimentos": movimentos,
            "chegou_saida": chegou_saida,
        })

    return pd.DataFrame(registros)


# Função que lê uma tabela de resultados impressa na saída do notebook.
def ler_tabela_saida(saida_notebook, titulo):

    linhas = saida_notebook.split(titulo, 1)[1].strip().splitlines()

    resultados = []

    # A primeira linha é o cabeçalho; a tabela termina na primeira linha vazia.
    for linha in linhas[1:]:

        if not linha.strip():

            break

        valor, seed, _, pontuacao, movimentos, chegou_saida = linha.split()

        resultados.append((float(valor), int(seed), int(pontuacao), int(movimentos), chegou_saida == "True"))

    return resultados


# Função que converte um número escrito no README (ex.: "1.000", "−38,67")
# para float.
def converter_numero_readme(texto):

    texto = texto.strip().replace("−", "-").replace(".", "").replace(",", ".")

    return float(texto)


# Função que lê uma tabela de resumo do README a partir do seu título.
def ler_tabela_readme(titulo):

    with open(CAMINHO_README, encoding="utf-8") as arquivo:

        conteudo = arquivo.read()

    linhas = conteudo.split(titulo, 1)[1].strip().splitlines()

    tabela = []

    # As duas primeiras linhas são o cabeçalho e o separador.
    for linha in linhas[2:]:

        if not linha.startswith("|"):

            break

        colunas = [coluna.strip() for coluna in linha.strip("|").split("|")]

        sucessos = int(re.match(r"(\d+) de 3", colunas[1]).group(1))

        tabela.append((converter_numero_readme(colunas[0]), sucessos, converter_numero_readme(colunas[2]), converter_numero_readme(colunas[3])))

    return tabela


# Função que verifica o cálculo da taxa de sucesso em porcentagem e dos
# desvios padrão amostrais.
def test_resumir_resultados(nb):

    df = pd.DataFrame({

        "alpha": [0.1, 0.1, 0.2, 0.2],
        "pontuacao_acumulada": [10, 20, -150, -150],
        "qtd_movimentos": [5, 15, 150, 150],
        "chegou_saida": [True, False, False, False],
    })

    resumo = nb["resumir_resultados"](df, "alpha")

    assert list(resumo.columns) == ["alpha", "taxa_sucesso", "pontuacao_media", "pontuacao_desvio", "movimentos_medios", "movimentos_desvio"]

    assert list(resumo["taxa_sucesso"]) == [50.0, 0.0]
    assert list(resumo["pontuacao_media"]) == [15.0, -150.0]
    assert list(resumo["movimentos_medios"]) == [10.0, 150.0]

    # Desvio padrão amostral (ddof=1).
    assert resumo["pontuacao_desvio"].iloc[0] == pytest.approx(np.std([10, 20], ddof=1))
    assert resumo["movimentos_desvio"].iloc[1] == 0.0


# Função que verifica se as funções de comparação geram um registro para
# cada combinação de parâmetro e semente.
def test_comparacoes_registram_cada_combinacao(nb, config):

    argumentos = (config["mapa_labirinto"],)

    resultados_alphas = nb["comparar_alphas"]([0.01, 0.05], [1, 2], *argumentos, config["gama"], config["epsilon"], config["acoes_disponiveis"], config["limite_movimentos"], config["posicao_inicial"], config["posicao_saida"], 3)

    assert [(r["alpha"], r["seed"], r["episodios"]) for r in resultados_alphas] == [(0.01, 1, 3), (0.01, 2, 3), (0.05, 1, 3), (0.05, 2, 3)]

    resultados_episodios = nb["comparar_episodios"]([1, 2], [1, 2], *argumentos, 0.01, config["gama"], config["epsilon"], config["acoes_disponiveis"], config["limite_movimentos"], config["posicao_inicial"], config["posicao_saida"])

    assert [(r["episodios"], r["seed"], r["alpha"]) for r in resultados_episodios] == [(1, 1, 0.01), (1, 2, 0.01), (2, 1, 0.01), (2, 2, 0.01)]

    for registro in resultados_alphas + resultados_episodios:

        assert set(registro) == {"alpha", "seed", "episodios", "pontuacao_acumulada", "qtd_movimentos", "chegou_saida"}


# Função que verifica o desenho do labirinto e do caminho percorrido.
def test_desenhar_jogo_e_caminho(nb, config, sem_janela):

    figura, marcador = nb["desenhar_jogo"]((1, 1), config["mapa_labirinto"])

    assert list(marcador.get_xdata()) == [1]
    assert list(marcador.get_ydata()) == [1]

    plt.close(figura)

    caminho = [(1, 1), (1, 2), (1, 3)]

    nb["desenhar_caminho"](config["mapa_labirinto"], caminho, "Teste")

    ax = plt.gca()

    linha_caminho = [linha for linha in ax.get_lines() if linha.get_label() == "Caminho"][0]

    assert list(linha_caminho.get_xdata()) == [1, 2, 3]
    assert list(linha_caminho.get_ydata()) == [1, 1, 1]

    agente = [linha for linha in ax.get_lines() if linha.get_label() == "Agente"][0]

    # O marcador termina na última posição do caminho.
    assert list(agente.get_xdata()) == [3]
    assert ax.get_title() == "Teste"


# Função que verifica que um caminho vazio não gera desenho.
def test_desenhar_caminho_vazio(nb, config, sem_janela):

    assert nb["desenhar_caminho"](config["mapa_labirinto"], [], "Vazio") is None
    assert plt.get_fignums() == []


# Função que verifica o gráfico de barras da taxa de sucesso.
def test_plotar_taxa_sucesso(nb, sem_janela):

    resumo = nb["resumir_resultados"](criar_dataframe(RESULTADOS_ALPHAS, "alpha"), "alpha")

    nb["plotar_taxa_sucesso"](resumo, "alpha", "Taxa de Sucesso por Alpha", "Alpha", "#2ca02c")

    alturas = [barra.get_height() for barra in plt.gca().patches]

    assert alturas == pytest.approx([0.0, 200 / 3, 100.0])


# Função que verifica se os resultados esperados nos testes são os mesmos
# registrados na saída salva do notebook.
def test_resultados_esperados_conferem_com_saida_do_notebook(saida_notebook):

    assert ler_tabela_saida(saida_notebook, "--- RESULTADOS DOS ALPHAS ---") == RESULTADOS_ALPHAS
    assert ler_tabela_saida(saida_notebook, "--- RESULTADOS DOS EPISÓDIOS ---") == RESULTADOS_EPISODIOS

    assert "Pontuação: 17" in saida_notebook
    assert "Movimentos: 14" in saida_notebook
    assert "Chegou à saída: True" in saida_notebook


# Função que verifica se as tabelas de resumo do README conferem com
# os resultados do notebook.
@pytest.mark.parametrize("titulo, resultados, parametro", [

    ("### Comparação de alpha", RESULTADOS_ALPHAS, "alpha"),
    ("### Comparação de episódios", RESULTADOS_EPISODIOS, "episodios"),
])
def test_resumo_do_readme_confere_com_notebook(nb, titulo, resultados, parametro):

    resumo = nb["resumir_resultados"](criar_dataframe(resultados, parametro), parametro)

    esperado = []

    for _, linha in resumo.iterrows():

        sucessos = int(round(linha["taxa_sucesso"] * 3 / 100))

        esperado.append((linha[parametro], sucessos, round(linha["pontuacao_media"], 2), round(linha["movimentos_medios"], 2)))

    assert ler_tabela_readme(titulo) == esperado


# Função que converte os registros das comparações para o formato dos
# resultados esperados.
def para_tuplas(resultados, parametro):

    return [(r[parametro], r["seed"], r["pontuacao_acumulada"], r["qtd_movimentos"], r["chegou_saida"]) for r in resultados]


# Função que reproduz a comparação dos alphas com a configuração do notebook.
@pytest.mark.lento
def test_reproduz_comparacao_dos_alphas(nb, config):

    resultados = nb["comparar_alphas"](config["valores_alpha"], config["seeds"], config["mapa_labirinto"], config["gama"], config["epsilon"], config["acoes_disponiveis"], config["limite_movimentos"], config["posicao_inicial"], config["posicao_saida"], config["qtd_episodios"])

    assert para_tuplas(resultados, "alpha") == RESULTADOS_ALPHAS


# Função que reproduz a comparação dos episódios com a configuração do notebook.
@pytest.mark.lento
def test_reproduz_comparacao_dos_episodios(nb, config):

    resultados = nb["comparar_episodios"](config["lista_episodios"], config["seeds"], config["mapa_labirinto"], config["alpha_referencia"], config["gama"], config["epsilon"], config["acoes_disponiveis"], config["limite_movimentos"], config["posicao_inicial"], config["posicao_saida"])

    assert para_tuplas(resultados, "episodios") == RESULTADOS_EPISODIOS


# Função que reproduz o modelo ilustrado (alpha 0,01, 1.000 episódios e
# semente 42) e verifica que o caminho evita as armadilhas.
@pytest.mark.lento
def test_reproduz_modelo_ilustrado(nb, config, sem_janela, capsys):

    nb["ilustrar_modelo"](config["mapa_labirinto"], 0.01, config["gama"], config["epsilon"], config["acoes_disponiveis"], config["limite_movimentos"], config["posicao_inicial"], config["posicao_saida"], 42, 1000)

    saida = capsys.readouterr().out

    assert "Pontuação: 17" in saida
    assert "Movimentos: 14" in saida
    assert "Chegou à saída: True" in saida

    ax = plt.gca()

    linha_caminho = [linha for linha in ax.get_lines() if linha.get_label() == "Caminho"][0]

    caminho = list(zip(linha_caminho.get_ydata(), linha_caminho.get_xdata()))

    assert caminho[0] == config["posicao_inicial"]
    assert caminho[-1] == config["posicao_saida"]

    for armadilha in config["posicao_armadilhas"]:

        assert armadilha not in caminho
