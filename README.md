# Projeto da Disciplina de Inteligência Artificial

Este é um projeto da disciplina de Inteligência Artificial (IA) que aborda a tarefa de recomendação de medicamentos com base em dados de pacientes e doenças. O objetivo é criar um modelo de machine learning capaz de recomendar o medicamento mais adequado para um determinado paciente com base nas características fornecidas.


## Notebooks

- [`project_ia.ipynb`]([https://github.com/Eduardo-J-S/IA_Project/blob/main/Notebooks/project_ia.ipynb](https://colab.research.google.com/drive/1V8nXwPMmXoyq3OBQmW8Gk2qNS6VNdn4A?usp=sharing)): Este notebook contém todo o código utilizado para a análise exploratória dos dados, preparação dos dados, modelagem e avaliação de cinco técnicas de machine learning diferentes.

## Conjunto de Dados

Os dados utilizados neste projeto foram coletados do Kaggle e incluem informações sobre pacientes e medicamentos recomendados. O conjunto de dados está disponível em [data/dataset.csv](https://github.com/Eduardo-J-S/IA_Project/blob/main/data/dataset.csv) e também pode ser encontrado no seguinte link do Kaggle: [Drug Classification Dataset](https://www.kaggle.com/datasets/prathamtripathi/drug-classification).

## Técnicas Utilizadas

As seguintes técnicas de machine learning foram aplicadas ao problema:

1. Perceptron
2. Regressão Logística
3. KNN(K — Nearest Neighbors)
4. SVM (Support Vector Machine)
5. Árvore de Decisão

## Métricas de Avaliação

Foram utilizadas as seguintes métricas para avaliar o desempenho dos modelos:

1. Acurácia: medida geral de precisão do modelo.
2. Matriz de Confusão: fornece informações detalhadas sobre o desempenho do modelo em cada classe.

## Resultados

Aqui estão os resultados obtidos para as cinco técnicas de machine learning aplicadas ao conjunto de dados:

- Técnica 1 (Perceptron):
  - Acurácia de Treinamento: 0.8071428571428572
  - Matriz de Confusão de Treinamento:
    [[55  0  3  0  3]
     [ 0 10  8  0  0]
     [ 0  0 13  0  0]
     [ 2  1  0  0  9]
     [ 0  0  1  0 35]]
  - Acurácia de Testes: 0.8166666666666667
  - Matriz de Confusão de Testes:
    [[26  0  3  0  1]
     [ 0  2  3  0  0]
     [ 0  0  3  0  0]
     [ 1  2  0  0  1]
     [ 0  0  0  0 18]]

- Técnica 2 (Regressão Logística):
  - Acurácia de Treinamento: 0.8142857142857143
  - Matriz de Confusão de Treinamento:
    [[56  0  0  0  5]
     [ 2 16  0  0  0]
     [ 4  1  8  0  0]
     [11  0  0  0  1]
     [ 2  0  0  0 34]]
  - Acurácia de Testes: 0.8166666666666667
  - Matriz de Confusão de Testes: 
    [[27  0  1  0  2]
     [ 1  4  0  0  0]
     [ 2  0  1  0  0]
     [ 2  2  0  0  0]
     [ 1  0  0  0 17]]

- Técnica 3 (KNN(K — Nearest Neighbors)):
  - Acurácia de Treinamento: 0.8428571428571429
  - Matriz de Confusão de Treinamento:
      [[50  2  1  3  5]
       [ 1 17  0  0  0]
       [ 1  2 10  0  0]
       [ 3  0  0  9  0]
       [ 4  0  0  0 32]]
  - Acurácia de Testes: 0.7166666666666667
  - Matriz de Confusão de Testes:
    [[17  2  2  1  8]
     [ 2  3  0  0  0]
     [ 0  0  3  0  0]
     [ 0  0  0  4  0]
     [ 2  0  0  0 16]]

- Técnica 4 (SVM (Support Vector Machine)):
  - Acurácia de Treinamento: 0.8714285714285714
  - Matriz de Confusão de Treinamento:
    [[53  1  1  1  5]
     [ 1 17  0  0  0]
     [ 1  2 10  0  0]
     [ 6  0  0  6  0]
     [ 0  0  0  0 36]]
  - Acurácia de Testes: 0.8
  - Matriz de Confusão de Testes:
    [[21  1  1  0  7]
     [ 2  3  0  0  0]
     [ 0  0  3  0  0]
     [ 1  0  0  3  0]
     [ 0  0  0  0 18]]

- Técnica 5 (Árvore de Decisão):
  - Acurácia de Treinamento: 1.0
  - Matriz de Confusão de Treinamento:
    [[61  0  0  0  0]
     [ 0 18  0  0  0]
     [ 0  0 13  0  0]
     [ 0  0  0 12  0]
     [ 0  0  0  0 36]]
  - Acurácia de Testes: 1.0
  - Matriz de Confusão de Testes:
    [[30  0  0  0  0]
     [ 0  5  0  0  0]
     [ 0  0  3  0  0]
     [ 0  0  0  4  0]
     [ 0  0  0  0 18]]

## Validação Cruzada

Para avaliar o desempenho dos modelos de aprendizagem da Árvore de Decisão e confirmar que não estava acontecendo overfitting, foi realizada a validação cruzada utilizando a técnica de 5-fold cross-validation. Nesta técnica, o conjunto de dados é dividido em 5 partes (folds) de tamanhos iguais. O modelo é treinado em 4 desses folds e avaliado no quinto fold. Esse processo é repetido 5 vezes, alternando os folds de treinamento e teste em cada repetição.

### Resultados da Validação Cruzada

Aqui estão os resultados obtidos com a validação cruzada para o modelo da Árvore de Decisão:

- Acurácia Média da Validação Cruzada: 0.9800000000000001
- Matriz de Confusão da Validação Cruzada:
  [[91  0  0  0  0]
   [ 0 22  1  0  0]
   [ 0  2 14  0  0]
   [ 0  0  0 16  0]
   [ 1  0  0  0 53]]

A acurácia média da validação cruzada fornece uma medida geral do desempenho do modelo em diferentes divisões dos dados de treinamento e teste. A matriz de confusão da validação cruzada mostra o número de previsões corretas e incorretas do modelo para cada classe em todas as dobras.

Essa análise usando a validação cruzada é importante para verificar a capacidade de generalização do modelo em diferentes situações e garantir que ele não esteja superajustando aos dados de treinamento.

## Conclusão

Com base nos resultados obtidos, a técnica mais adequada para o problema de recomendação de medicamentos é a Árvore de Decisão. O resultado da validação cruzada mostra uma acurácia média de aproximadamente 98.00%. Isso significa que o modelo de Árvore de Decisão teve uma alta taxa de acerto em média em todas as dobras utilizadas na validação cruzada, o que é excelente.

Em resumo, os resultados da validação cruzada sugerem que o modelo de Árvore de Decisão possui uma capacidade de generalização muito boa e é capaz de fazer previsões precisas para recomendação de medicamentos com base na doença e no tipo de paciente.

## Como Executar

Para executar o notebook `project_ia.ipynb`, siga as etapas abaixo:

**Observação**: Para garantir que o notebook funcione corretamente, é importante configurar o caminho para o conjunto de dados `drug200.csv`. Caso esteja utilizando o Google Colab, você pode seguir as etapas abaixo para configurar o caminho:

1. Se você estiver usando o Google Colab, certifique-se de ter o conjunto de dados `drug200.csv` disponível em sua conta do Google Drive.

2. Montar o Google Drive no Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')

## Contribuições

Este projeto foi realizado em grupo como parte de um trabalho acadêmico na disciplina de Inteligência Artificial. Os membros do grupo são os seguintes:

- Eduardo José
- Adonai Ermínio
- Arthur Gabriel
  
Todos os membros do grupo trabalharam juntos na análise dos resultados, discussão das abordagens utilizadas e nas decisões finais relacionadas ao projeto.


