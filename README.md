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
python3 src/main.py
```

## Analisando Resultados

Nesta análise, comparamos o desempenho de quatro classificadores — Logistic Regression, LDA, QDA e Naive Bayes — aplicados aos conjuntos de dados MNIST e CIFAR-10. Utilizamos métricas essenciais como precisão (precision), recall, F1-score, suporte (support) e acurácia (accuracy) para avaliar a eficácia dos modelos em tarefas de classificação. Esses indicadores nos ajudam a entender a capacidade de cada modelo em prever corretamente as classes e lidar com diferentes níveis de complexidade dos dados. 

Os resultados completos estão organizados na pasta `res`, que contém duas subpastas: `cifar10`, com os resultados referentes ao conjunto de dados CIFAR-10, e `mnist`, para os resultados do MNIST.

### Interpretação das Tabelas

Os resultados para cada classificador são apresentados em tabelas, organizados por classe. Aqui estão os principais elementos das tabelas:

- **Precision (Precisão)**: Mede a exatidão das previsões positivas do modelo. Quanto maior a precisão, menos falsos positivos.
- **Recall (Revocação)**: Mede a capacidade do modelo de capturar as instâncias positivas reais. Um recall alto indica menos falsos negativos.
- **F1-Score**: Combina a precisão e recall em uma única métrica balanceada. Ideal para quando há um equilíbrio entre as duas métricas.
- **Support (Suporte)**: Indica o número de instâncias reais de cada classe, útil para entender a distribuição de classes no conjunto de dados.
- **Accuracy (Acurácia)**: Refere-se à porcentagem de previsões corretas feitas pelo modelo em relação ao total de instâncias.
- **Macro Avg (Média Macro)**: A média simples de precision, recall e F1-score para todas as classes, dando igual peso a cada classe.
- **Weighted Avg (Média Ponderada)**: A média ponderada das métricas, ajustada pelo suporte de cada classe. Útil quando as classes são desbalanceadas.

### Desempenho no MNIST

O conjunto de dados MNIST, que contém imagens de dígitos escritos à mão (0 a 9), mostrou um desempenho robusto com quase todos os classificadores. Vejamos alguns pontos de destaque:

- **Logistic Regression** e **LDA** atingiram uma acurácia de 96%, com F1-scores muito próximos entre si para todas as classes. As classes foram identificadas com alta precisão e recall, refletindo a natureza relativamente simples dos dígitos escritos à mão.
- **QDA** teve a melhor performance no MNIST, alcançando 98% de acurácia, com excelente precisão e recall em quase todas as classes.
- **Naive Bayes**, apesar de ter uma acurácia inferior (86%), ainda se saiu razoavelmente bem considerando a simplicidade do modelo. Notamos que sua performance é afetada principalmente nas classes com menor suporte.

### Desempenho no CIFAR-10

No caso do CIFAR-10, um conjunto de dados muito mais complexo, composto por imagens coloridas de objetos em 10 categorias, o desempenho dos classificadores foi substancialmente inferior:

- **Logistic Regression** e **LDA** tiveram uma acurácia de aproximadamente 40%. A precisão e o recall variam bastante entre as classes, refletindo a maior complexidade e variabilidade nas imagens do CIFAR-10. Isso indica que esses modelos lutam para capturar a diferença entre classes visuais tão diversas.
- **QDA** obteve a melhor acurácia entre os classificadores testados (52%), com um F1-score superior em várias classes. Ainda assim, a performance está longe de ser ideal, mostrando que até os classificadores mais avançados têm dificuldade com este dataset.
- **Naive Bayes** foi o pior classificador no CIFAR-10, com uma acurácia de apenas 31%. Isso reflete que a suposição ingênua de independência entre os pixels não é apropriada para um conjunto de dados com alta variabilidade espacial e de cores.

### Comparação Geral entre MNIST e CIFAR-10

A disparidade nos resultados entre MNIST e CIFAR-10 pode ser explicada pela diferença fundamental na natureza dos dados:

- **Simplicidade do MNIST**: As imagens são em preto e branco, com baixa variação intra-classe (dígitos manuscritos), o que facilita a identificação dos padrões pelos classificadores. Isso se reflete nas acurácias altas, acima de 90% para quase todos os métodos.
- **Complexidade do CIFAR-10**: As imagens têm cores, maior resolução e contêm objetos muito variados. Isso torna a tarefa de classificação mais difícil, especialmente para modelos mais simples como Naive Bayes e Logistic Regression, que não são capazes de capturar eficientemente as complexidades dos padrões visuais.

## Conclusão

O desempenho dos classificadores no MNIST é significativamente melhor do que no CIFAR-10 devido à simplicidade dos dados no primeiro conjunto. MNIST, com suas imagens de dígitos simples e em preto e branco, é muito mais fácil de classificar. Em contraste, CIFAR-10 apresenta desafios complexos, como imagens coloridas e de maior variação, exigindo modelos mais avançados e adequados para a classificação de imagens complexas. 

Modelos lineares, como Logistic Regression e LDA, funcionam bem no MNIST, mas falham no CIFAR-10, indicando que conjuntos de dados mais complexos precisam de classificadores mais sofisticados e de técnicas que capturam a estrutura não-linear dos dados.