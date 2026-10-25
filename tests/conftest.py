import ast
import json
import os

import matplotlib

# Backend sem janela, para que os gráficos possam ser gerados nos testes.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest


CAMINHO_NOTEBOOK = os.path.join(os.path.dirname(__file__), "..", "q_learning.ipynb")


# Função que registra o marcador dos testes que reproduzem os experimentos
# completos do notebook, permitindo pulá-los com -m "not lento".
def pytest_configure(config):

    config.addinivalue_line("markers", "lento: reproduz os experimentos completos do notebook")


# Função que lê o notebook e separa as células de código.
def ler_celulas_codigo():

    with open(CAMINHO_NOTEBOOK, encoding="utf-8") as arquivo:

        notebook = json.load(arquivo)

    celulas = []

    for celula in notebook["cells"]:

        if celula["cell_type"] == "code":

            celulas.append(celula)

    return celulas


# Função que verifica se a célula é a execução principal do notebook.
def eh_execucao_principal(codigo):

    return '__name__ == "__main__"' in codigo


# Função que executa as células com as funções do notebook e devolve um
# dicionário com elas. As linhas de instalação (%pip) e a execução principal
# são ignoradas, pois os testes chamam as funções diretamente.
def carregar_funcoes_notebook():

    namespace = {"__name__": "q_learning_notebook"}

    for celula in ler_celulas_codigo():

        codigo = "".join(celula["source"])

        if eh_execucao_principal(codigo):

            continue

        linhas = [linha for linha in codigo.splitlines() if not linha.lstrip().startswith(("%", "!"))]

        exec(compile("\n".join(linhas), CAMINHO_NOTEBOOK, "exec"), namespace)

    return namespace


# Função que lê as atribuições da execução principal (mapa, posições e
# parâmetros) sem executar os experimentos, garantindo que os testes usem
# exatamente a mesma configuração do notebook.
def carregar_configuracao_notebook():

    for celula in ler_celulas_codigo():

        codigo = "".join(celula["source"])

        if not eh_execucao_principal(codigo):

            continue

        configuracao = {}

        for no in ast.parse(codigo).body:

            if isinstance(no, ast.If):

                for instrucao in no.body:

                    # Apenas atribuições literais; as chamadas dos experimentos são ignoradas.
                    if isinstance(instrucao, ast.Assign) and not isinstance(instrucao.value, ast.Call):

                        nome = instrucao.targets[0].id

                        configuracao[nome] = ast.literal_eval(instrucao.value)

        return configuracao

    raise RuntimeError("Execução principal não encontrada no notebook.")


# Função que devolve as saídas de texto registradas na execução principal.
def ler_saida_execucao_principal():

    for celula in ler_celulas_codigo():

        if eh_execucao_principal("".join(celula["source"])):

            textos = []

            for saida in celula.get("outputs", []):

                if saida["output_type"] == "stream":

                    textos.append("".join(saida["text"]))

            return "".join(textos)

    return ""


@pytest.fixture(scope="session")
def nb():

    return carregar_funcoes_notebook()


@pytest.fixture(scope="session")
def config():

    return carregar_configuracao_notebook()


@pytest.fixture(scope="session")
def saida_notebook():

    return ler_saida_execucao_principal()


@pytest.fixture
def sem_janela(monkeypatch):

    # Evita o aviso do plt.show no backend sem janela.
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)


@pytest.fixture(autouse=True)
def fechar_figuras():

    yield

    plt.close("all")
