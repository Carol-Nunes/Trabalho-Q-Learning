# Q-Learning Aproximado em um Labirinto com Armadilhas

Trabalho da disciplina de Inteligência Artificial da FCT/UNESP, sob orientação do professor Danillo Roberto Pereira.

## Integrantes

- **Caroline Nunes Araujo** — [@Carol-Nunes](https://github.com/Carol-Nunes)
- **Gabriel Ciriaco de Carvalho** — [@Gabriel-Ciriaco](https://github.com/Gabriel-Ciriaco)
- **João Gabriel de Morais Bezerra** — [@joaobezcerra](https://github.com/joaobezcerra)

## Objetivo

Implementar um jogo de labirinto e um agente que aprenda a alcançar a saída utilizando Q-learning com aproximação linear. Analisar a influência da taxa de aprendizado e da quantidade de episódios no desempenho, considerando diferentes sementes aleatórias.

## Ambiente

O labirinto possui 7 linhas e 9 colunas, com mapa e posições fixos.

| Símbolo | Significado |
|---------|-------------|
| `#` | Parede |
| `.` | Caminho livre |
| `S` | Início |
| `E` | Saída |
| `T` | Armadilha |

O agente começa em `(1, 1)` e deve alcançar a saída em `(1, 7)`. As coordenadas seguem o formato `(linha, coluna)`, com índices iniciados em zero.

As ações disponíveis são cima, baixo, esquerda e direita.

| Resultado da ação | Recompensa |
|------------------|------------|
| Movimento para caminho livre ou início | −1 |
| Tentativa de atravessar parede ou limite do mapa | −3 |
| Entrada em armadilha | −10 |
| Chegada à saída | +30 |

Cada ação recebe apenas uma dessas recompensas. Colisões mantêm a posição atual; armadilhas permitem continuar o percurso.

O episódio termina ao alcançar a saída ou completar 150 movimentos, incluindo colisões.

## Algoritmo

O agente estima Q pelo produto escalar entre um vetor de pesos e 17 características:

- Bias.
- Indicador de colisão.
- Indicador de entrada em armadilha.
- Indicador de chegada à saída.
- Distância Manhattan normalizada até a saída.
- 12 bases gaussianas que representam a proximidade a pontos de referência do mapa.

Os pesos começam em zero e são atualizados após cada movimento. Durante o treinamento, a escolha das ações utiliza epsilon-greedy.

Na avaliação, epsilon é zero e os pesos permanecem fixos. Empates entre ações com o mesmo Q são resolvidos por sorteio.

## Tecnologias

A implementação utiliza Python com as seguintes bibliotecas:

- **NumPy:** operações numéricas e geração de números aleatórios.
- **pandas:** organização e resumo dos resultados.
- **matplotlib:** gráficos e visualização do labirinto.

O notebook deve ser executado em um ambiente compatível com Jupyter.

## Como executar

1. Abra o arquivo `q_learning.ipynb` no Jupyter ou no VSCode com suporte a notebooks.
2. Selecione um kernel Python.
3. Execute a célula de instalação das bibliotecas, caso necessário:

   ```python
   %pip install matplotlib numpy pandas
   ```

4. Execute todas as células em ordem, do início ao fim.

A execução principal realiza as comparações de parâmetros, apresenta as tabelas e os gráficos e desenha o percurso de um modelo treinado. Não é necessário fornecer movimentos manualmente.


## Experimentos

Foram realizadas duas comparações:

| Comparação | Valores testados | Configuração fixa |
|------------|------------------|-------------------|
| Taxa de aprendizado | 0,005; 0,01; 0,05 | 1.000 episódios |
| Quantidade de episódios | 100; 500; 1.000 | Alpha = 0,01 |

Configurações comuns:

- Gama: `0.95`.
- Epsilon no treinamento: `0.3`.
- Limite: 150 movimentos por episódio.
- Sementes de treinamento: `42`, `43` e `44`.
- Semente do gerador de avaliação: `123`.

Cada treinamento começa com os pesos zerados. Cada modelo é avaliado uma vez, sem exploração e sem atualização dos pesos.

## Resultados

### Comparação de alpha

| Alpha | Modelos com sucesso | Pontuação média | Movimentos médios |
|-------|--------------------|-----------------|-------------------|
| 0,005 | 0 de 3 — 0% | −150,00 | 150,00 |
| 0,01 | 2 de 3 — 66,7% | −38,67 | 59,33 |
| 0,05 | 3 de 3 — 100% | 13,67 | 11,33 |

### Comparação de episódios

| Episódios | Modelos com sucesso | Pontuação média | Movimentos médios |
|-----------|--------------------|-----------------|-------------------|
| 100 | 0 de 3 — 0% | −150,00 | 150,00 |
| 500 | 2 de 3 — 66,7% | −38,67 | 59,33 |
| 1.000 | 2 de 3 — 66,7% | −38,67 | 59,33 |

As médias incluem as avaliações sem sucesso. A pontuação corresponde à soma das recompensas sem desconto, enquanto gama é utilizado no aprendizado.

O modelo ilustrado, treinado com alpha `0.01`, 1.000 episódios e semente `42`, alcançou a saída em **14 movimentos**, evitando armadilhas e acumulando **17 pontos**.

O relatório apresenta os resultados individuais, os desvios padrão, os gráficos e a discussão dos experimentos.

## Limitações

Os experimentos utilizam um mapa e uma posição inicial fixos, três sementes de treinamento e uma avaliação por modelo. As conclusões se restringem às configurações testadas, sem garantir desempenho em outros labirintos, convergência ou obtenção da política ótima.