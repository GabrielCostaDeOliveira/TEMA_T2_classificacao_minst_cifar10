# SEGUNDO PROJETO DE TÓPICOS ESPECIAIS EM MATEMÁTICA APLICADA

## Introdução

Neste projeto, serão aplicadas técnicas de classificação para diferenciar imagens de dois conjuntos de dados: o MNIST, que contém dígitos manuscritos, e o CIFAR-10, composto por imagens coloridas de diferentes objetos. O objetivo é utilizar quatro classificadores estudados em aula — Regressão Logística, Análise do Discriminante Linear (LDA), Análise do Discriminante Quadrático (QDA) e Naïve Bayes — para realizar a classificação dessas imagens. As imagens do MNIST são em tons de cinza, com resolução de 28x28 pixels, enquanto as do CIFAR-10 são coloridas e possuem 32x32 pixels, com três canais de cor (vermelho, verde e azul). Essas diferenças nas características das imagens impõem desafios distintos para os classificadores, e, como resultado, espera-se que o desempenho varie significativamente entre os dois conjuntos de dados. O projeto tem como objetivo explorar essas diferenças, justificando os resultados obtidos e fornecendo uma análise detalhada de cada abordagem. A entrega final está prevista para o dia 15 de setembro de 2024.

## Por que não utilizar o Jupyter nesse trabalho?

Jupyter Notebooks, apesar de úteis para prototipagem e visualização, podem não ser ideais para projetos com grandes bases de dados como MNIST e CIFAR-10 devido a algumas limitações:

1. **Consumo de Memória**: Jupyter Notebooks podem consumir mais memória, pois mantêm o estado das variáveis e resultados intermediários em cada célula. Isso pode ser um problema com grandes conjuntos de dados.

2. **Desempenho**: Scripts Python (`.py`) oferecem maior controle sobre a execução e otimização, resultando em melhor desempenho e uso mais eficiente dos recursos, especialmente ao lidar com dados complexos.

Portanto, para garantir melhor desempenho e eficiência, o uso de scripts Python é preferível para trabalhar com grandes volumes de dados e operações intensivas.

## Integrantes

<table align="center">
    <colgroup>
        <col style="background-color: #722f37" />
        <col span="2" />
    </colgroup>
    <thead>
        <tr>
            <th>ID</th>
            <th>Nome</th>
            <th>Matrícula</th>
            <th>Foto</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="center">1</td>
            <td align="center"><a href="https://github.com/gustavomartins-github">Gustavo Martins Ribeiro</a></td>
            <td align="center">200019228</td>
            <td align="center"><img src="https://github.com/gustavomartins-github.png" width="50%"></td>
        </tr>
        <tr>
            <td align="center">2</td>
            <td align="center"><a href="https://github.com/GabrielCostaDeOliveira">Gabriel Costa de Oliveira</a></td>
            <td align="center">190045817</td>
            <td align="center"><img src="https://github.com/GabrielCostaDeOliveira.png" width="50%"></td>
        </tr>
    </tbody>
</table>

## Bibliotecas

1. **[scikit-learn](https://scikit-learn.org/stable/)**: Uma biblioteca essencial para machine learning em Python, contendo implementações eficientes de diversos algoritmos de classificação, regressão e clustering.
2. **[pandas](https://pandas.pydata.org/)**: Utilizada para manipulação e análise de dados, permite a estruturação eficiente dos dados para uso nos classificadores.
3. **[matplotlib](https://matplotlib.org/)**: Biblioteca para visualização de dados, ideal para a criação de gráficos e análises visuais dos resultados obtidos.
4. **[tensorflow](https://www.tensorflow.org/?hl=pt-br)**: Utilizada para carregar e processar os conjuntos de dados MNIST e CIFAR-10, além de possibilitar a criação de modelos de machine learning.
5. **[seaborn](https://seaborn.pydata.org/)**: Uma biblioteca de visualização de dados baseada no Matplotlib, que oferece gráficos mais atraentes e informativos.

## Classificadores

1. **Regressão Logística**: Um modelo estatístico usado para prever a probabilidade de uma classe com base nas entradas. É particularmente útil em problemas de classificação binária, mas pode ser estendido para multiclasse.
  
2. **Análise do Discriminante Linear (LDA)**: Um classificador que projeta os dados em um espaço de menor dimensão, maximizando a separação entre classes. É eficaz quando as classes possuem distribuições Gaussianas com covariâncias iguais.
  
3. **Análise do Discriminante Quadrático (QDA)**: Similar ao LDA, mas permite covariâncias diferentes entre as classes, proporcionando mais flexibilidade em problemas onde as classes não são linearmente separáveis.
  
4. **Naïve Bayes**: Baseado no teorema de Bayes, esse classificador assume que as características são independentes umas das outras, simplificando o cálculo das probabilidades e sendo muito eficiente para grandes conjuntos de dados.

## Algoritmos Implementados

- **GridSearchCV**: Algoritmo da biblioteca `scikit-learn` que combina `GridSearch` com `Cross-Validation`. Ele realiza uma busca exaustiva sobre uma grade de valores de parâmetros especificados para um estimador e avalia cada combinação de parâmetros usando validação cruzada. O objetivo é encontrar a melhor combinação de hiperparâmetros para otimizar o desempenho do modelo.

  - **GridSearch**: Parte do `GridSearchCV`, é a técnica que realiza a busca sistemática através de uma grade de combinações de parâmetros. Cada combinação é avaliada para encontrar a que produz o melhor desempenho do modelo.

  - **Cross-Validation**: Técnica utilizada pelo `GridSearchCV` para avaliar o desempenho do modelo de forma robusta. Os dados são divididos em múltiplos subconjuntos (folds), e o modelo é treinado em um subconjunto e testado em outro. Isso ajuda a garantir uma avaliação mais precisa e reduz o risco de overfitting, que ocorre quando um modelo se ajusta excessivamente aos dados de treino e tem um desempenho ruim em novos dados.

## Tutorial: Configuração e Execução do Projeto

Siga os passos abaixo para configurar o ambiente de desenvolvimento e executar o projeto corretamente:

### 1. Verifique a Instalação do Python

Antes de iniciar, é essencial garantir que você tenha o [Python](https://www.python.org/downloads/) instalado em seu sistema. Caso ainda não o tenha, faça o download e instale a versão mais recente disponível no site oficial.

### 2. Crie e Ative um Ambiente Virtual

Para evitar conflitos de dependências e manter o projeto organizado, recomenda-se a criação de um ambiente virtual. Para isso, siga os passos abaixo:

- No terminal, crie o ambiente virtual executando o seguinte comando:

   ```bash
   python -m venv .venv
   ```

- Em seguida, ative o ambiente virtual conforme o seu sistema operacional:

   - **Windows:**
   
     ```bash
     .venv\Scripts\activate
     ```
   
   - **macOS/Linux:**
   
     ```bash
     source .venv/bin/activate
     ```

### 3. Instale as Dependências do Projeto

Com o ambiente virtual ativado, instale todas as bibliotecas e dependências necessárias para o projeto utilizando o arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Execute o Projeto

Após a instalação das dependências, você pode executar o programa principal do projeto usando o seguinte comando:

```bash
python src/main.py
```

## Analisando Resultados

Para avaliar a performance dos classificadores aplicados aos conjuntos de dados MNIST e CIFAR-10, vamos analisar as métricas de desempenho fornecidas pelos resultados. Abaixo está um exemplo de como os resultados são estruturados e interpretados:

### Estrutura da Tabela de Resultados

Os resultados de desempenho dos classificadores são frequentemente apresentados em tabelas como a seguinte:

```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        50
           1       0.87      0.97      0.92        40
           2       0.93      0.96      0.95        45
           3       0.98      0.90      0.94        50
           4       1.00      1.00      1.00        36
           5       1.00      0.94      0.97        51
           6       0.98      0.96      0.97        53
           7       1.00      1.00      1.00        37
           8       0.89      0.93      0.91        42
           9       0.98      0.98      0.98        46

    accuracy                           0.96       450
   macro avg       0.96      0.96      0.96       450
weighted avg       0.96      0.96      0.96       450
```

- **Precision**: A proporção de verdadeiros positivos sobre o total de positivos previstos. Indica a acurácia das previsões positivas.
- **Recall**: A proporção de verdadeiros positivos sobre o total de reais positivos. Mede a capacidade do modelo de identificar todas as instâncias positivas.
- **F1-Score**: A média harmônica entre precisão e recall. É uma métrica única que considera tanto a precisão quanto o recall.
- **Support**: O número de ocorrências reais da classe em questão.
- **Accuracy**: A proporção geral de previsões corretas sobre o total de instâncias.
- **Macro Avg**: A média das métricas para cada classe, tratando todas as classes igualmente.
- **Weighted Avg**: A média ponderada das métricas para cada classe, ajustada pelo número de instâncias em cada classe.

### Comparação de Resultados (MNIST vs CIFAR-10)

A comparação dos resultados entre MNIST e CIFAR-10 revela diferenças no desempenho dos classificadores devido às características distintas dos conjuntos de dados. MNIST, com suas imagens de dígitos manuscritos em tons de cinza, e CIFAR-10, com imagens coloridas e complexas, apresentam desafios variados para os classificadores. O desempenho em cada conjunto de dados pode variar significativamente, refletindo como as características dos dados influenciam a eficácia dos modelos.

### Conclusão

A análise detalhada dos resultados permite uma compreensão mais profunda das capacidades e limitações dos classificadores aplicados a diferentes conjuntos de dados. A comparação entre MNIST e CIFAR-10 destaca as influências das características dos dados no desempenho dos modelos, oferecendo insights valiosos sobre como melhorar e ajustar as abordagens de classificação para diferentes tipos de dados.
