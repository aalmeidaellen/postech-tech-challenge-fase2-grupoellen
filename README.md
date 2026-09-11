# Tech Challenge — Fase 2 | POSTECH Data Analytics

> **INSTRUÇÕES:** este README é um template. Substitua **todos** os blocos marcados com
> `<!-- PREENCHER -->` e apague as linhas de instrução antes de submeter.
> O README vale **3 pontos** na Dimensão 1 da rúbrica.

---

## 1. Identificação

| Campo | Valor |
|---|---|
| Turma | 2DTATBB  - Data & Analytics e IA|
| Grupo | INDIVIDUAL |
| Data de entrega | ATÉ 10/10/26 |

### Integrantes

| Nome completo | RM | E-mail |
|---|---|---|
| ELLEN DE ALMEIDA ARAÚJO | RM377818 | |


---

## 2. Links da entrega

| Item | Link |
|---|---|
| Repositório | https://github.com/aalmeidaellen/postech-tech-challenge-fase2-grupoellen |
| Vídeo executivo (≤ 5 min) | <!-- PREENCHER: YouTube não listado / Drive com acesso liberado --> |
| Apresentação | <!-- PREENCHER: link do arquivo em `docs/` ou Drive --> |

> ⚠️ Repositório privado ou inacessível **zera** toda a Dimensão 1 da rúbrica.
> Confira o acesso em uma janela anônima antes de enviar.

---

## 3. O problema

A análise tem como objetivo desenvolver um modelo de aprendizado de máquina, utilizando técnicas de 
classificação supervisionadas, capaz de prever se um pedido de cartão de crédito deve ser aprovado 
com base em informações cadastrais e financeiras dos solicitantes combinadas com seu histórico de 
crédito. O projeto aborda as etapas de pré-processamento de dados, análise exploratória, seleção de 
atributos, modelagem e avaliação de desempenho, utilizando diferentes algoritmos. 

O problema é relevante para instituições financeiras porque uma decisão de crédito precisa 
equilibrar dois objetivos:

-identificar clientes potencialmente inadimplentes;
-evitar a classificação indevida de clientes de menor risco.



### Variável alvo

Foi criada uma variável binária denominada TARGET.

A classe TARGET = 1 representa clientes classificados como de maior risco, de acordo com 
a seguinte regra de negócio, considerando um período de 61 meses:

PELO MENOS 2 MESES COM STATUS = 1: de 30 a 59 dias de atraso (recorrência de atraso média) OU

PELO MENOS 1 MES COM STATUS >= 2, 60 dias ou mais de atraso o(gravidade do atraso) OU

PELO MENOS 6 MESES COM STATUS = 0, de 1 a 29 dias de atraso (recorrência de atraso moderada)

Caso nenhuma dessas condições seja atendida, o cliente recebe TARGET = 0.

Os valores C (quitado) e X(Sem informação) do histórico de crédito foram convertidos para -1, 
para tratá-los como número e desconsiderá-lo na regra.

A base final utilizada na modelagem apresentou:

36.457 clientes
51 variáveis preditoras
57,74% de clientes na classe 0
42,25% de clientes na classe 1

### Dataset

| Campo | Valor |
|---|---|
| Fonte | Kaggle — Credit Card Approval Prediction - https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction  |
|Arquivos|application_record.csv e credit_record.csv |
| Linhas × colunas | application_record.csv 438.557 linhas × 18 colunas |
| Linhas × colunas | credit_record.csv     1.048.575 linhas × 3 colunas |
| Período / versão | Dataset disponibilizado no Kaggle|
| Licença de uso | CC0 — Public Domain |

Descrição das variáveis:

| Variável              | Tipo             | Descrição                                           |
|AGE                	|Numérica          |Idade calculada a partir de DAYS_BIRTH|
|AMT_INCOME_TOTAL_LOG 	|Numérica          |Renda anual transformada por log1p|
|YEARS_EMPLOYED         |Numérica          |Tempo de emprego em anos|
|CNT_CHILDREN           |Numérica          |Quantidade de filhos|
|FLAG_OWN_CAR           |Binária           |Indica posse de veículo|
|FLAG_OWN_REALTY        |Binária           |Indica posse de imóvel|
|FLAG_PHONE             |Binária           |Indica existência de telefone|
|FLAG_WORK_PHONE        |Binária           |Indica existência de telefone profissional|
|FLAG_EMAIL	            |Binária           |Indica existência de e-mail|
|CODE_GENDER            |Categórica        |Gênero|
|NAME_INCOME_TYPE       |Categórica        |Tipo de renda|
|NAME_EDUCATION_TYPE    |Categórica        |Escolaridade|
|NAME_FAMILY_STATUS     |Categórica        |Estado civil|
|NAME_HOUSING_TYPE      |Categórica        |Tipo de moradia|
|OCCUPATION_TYPE        |Categórica        |Tipo de ocupação|
|TARGET|                |Binária           |Indica risco de crédito|
---

## 4. Como reproduzir

```Clonar o repositório git clone
 https://github.com/aalmeidaellen/postech-tech-challenge-fase2-grupoellen


No Windows:

py -m venv .venv
.venv\Scripts\activate
nstalar as dependências
py -m pip install -r requirements.txt
Executar o pipeline

Os dados brutos devem estar em:

data/raw/
├── application_record.csv
└── credit_record.csv

O pipeline pode ser executado diretamente com:

py src/pipeline.py

O arquivo src/pipeline.py realiza:

carregamento dos dados;
criação da variável TARGET;
tratamento das variáveis;
criação de AGE e YEARS_EMPLOYED;
tratamento da variável OCCUPATION_TYPE;
transformação das variáveis categóricas;
transformação logarítmica da renda;
remoção das variáveis não utilizadas;
divisão entre treino e teste;
treinamento do Random Forest;
avaliação do modelo;
cálculo da importância das variáveis.
```

Baixe o dataset e coloque o arquivo bruto em `data/raw/` (os dados **não** são versionados —
veja `data/README.md`).

Depois execute os notebooks nesta ordem (ou o executar o pipeline descrito acima):

| # | Notebook | O que faz |
|---|---|---|
| 1 | `notebooks/01_eda.ipynb` | Análise exploratória |
| 2 | `notebooks/02_preprocessamento.ipynb` | Limpeza, escala e feature engineering |
| 3 | `notebooks/03_modelagem.ipynb` | Treino e comparação dos modelos |
| 4 | `notebooks/04_avaliacao.ipynb` | Métricas, importância de variáveis e conclusões |

**Semente fixa:** `RANDOM_STATE = 42`, declarada na primeira célula de cada notebook.
Rodar os notebooks na ordem acima, a partir de um ambiente limpo, deve reproduzir
exatamente os números da seção 5.

---

## 5. Resultados

| Modelo              | Acurácia | Precisão | Recall | F1  | AUC-ROC | PR-ROC |
|---------------------|----------|----------|--------|-----|---------|--------|
| Logistic Regression | 0.53     | 0.59     | 0.54   | 0.57| 0.54    | 0.60   |
| Decision Tree       | 0.56     | 0.59     | 0.75   | 0.66| 0.55    | 0.60   |
| Random Forest       | 0.69     | 0.73     | 0.72   | 0.72| 0.74    | 0.77   |
| HistGradientBoosting| 0.61     | 0.60     | 0.74   | 0.73| 0.62    | 0.67   |
**Modelo escolhido:** Considerando o desbalanceamento da variável TARGET, a escolha do modelo 
não foi baseada apenas na acurácia. Foram considerados principalmente Precision, Recall, 
F1-score, ROC-AUC e PR-AUC para avaliar a capacidade de identificação dos clientes classificados
como de maior risco.
Entre os modelos avaliados, o Random Forest apresentou o melhor desempenho geral, com 
ROC-AUC de 0,74 e PR-AUC de 0,77, além de Precision de 0,73, Recall de 0,72 e F1-score de 0,72.

**Métricas priorizadas:** 
maior AUC-ROC;
maior PR-AUC;
melhor F1-score entre os modelos avaliados;
Recall relevante para a identificação de clientes de maior risco.

O modelo utilizado no pipeline possui:

n_estimators = 300
class_weight = balanced
random_state = 42
n_jobs = -1

Threshold

Embora o threshold padrão seja 0,50, foi adotado o threshold de 0,10 para a decisão final.

Essa escolha foi feita porque o objetivo de negócio prioriza a identificação de uma parcela maior dos clientes classificados como de maior risco.
---

## 6. Principais conclusões

1 - Modelo Final: O Random Forest foi selecionado como modelo final, apresentando o melhor equilíbrio 
entre discriminação, precisão e desempenho sobre a classe minoritária, com destaque para a PR-AUC.

2 - Threshold Final: O threshold de 0,10 foi adotado para priorizar a identificação de clientes de maior 
risco, alcançando Recall de 96,5%, mesmo com maior ocorrência de falsos positivos.

3 - A escolha do threshold reflete uma decisão de negócio, na qual é preferível realizar 
análises adicionais de clientes classificados preventivamente como risco do que deixar 
passar clientes potencialmente inadimplentes.

4 - As variáveis mais relevantes foram idade, renda e tempo de emprego, que juntas representam 
aproximadamente 55% da importância preditiva do modelo, indicando maior influência do perfil socioeconômico 
e da estabilidade do solicitante.


### Limitações e próximos passos

A variável TARGET foi construída a partir de uma regra de negócio definida neste projeto e 
pode ser aprimorada em estudos futuros.
Os registros C e X foram convertidos para STATUS_NUM = -1, para viabilizar o tratamento numérico, 
sendo considerado redutor no percentual do status >1 mas apenas do teste da regra de negócio 1
A utilização de 60 para YEARS_EMPLOYED representa uma decisão de tratamento dos registros associados 
a pensionistas e pode ser refinada com uma variável específica para essa condição.
O modelo Random Forest possui menor interpretabilidade que modelos lineares, o que pode ser relevante 
em aplicações reais de crédito.
Como próximos passos, recomenda-se avaliar calibração das probabilidades, diferentes estratégias 
de threshold, técnicas de explicabilidade e validação temporal do modelo.

---

## 7. Estrutura do repositório

```
├── data/ 
│		├── raw/ 
		│ │ ├── application_record.csv 
		│ │ └── credit_record.csv 
		│ └── processed/ 
		│ 
├── docs/ 
│ 	└── apresentação executiva 
│ 
├── notebooks/ 
│		 ├── 00 credit_card_PosTech_fase2_grupoEllen.ipynb 
│ 		 ├── 01_eda.ipynb 
│ 		 ├── 02_preprocessamento.ipynb 
│		 ├── 03_modelagem.ipynb │ 
		 ├── 04_avaliacao.ipynb 
		 │ └── README.md 
│ 
├── results/ 
│ 
├── src/ 
│ 	└── pipeline.py 
│ 
├── submissao/ 
│ 
├── .gitignore 
├── CHECKLIST.md
 ├── ESTRUTURA.md 
 ├── LICENSE 
 ├── README.md 
 └── requirements.txt
```

Detalhes e convenções em [`ESTRUTURA.md`](ESTRUTURA.md).
Antes de enviar, percorra o [`CHECKLIST.md`](CHECKLIST.md).

---

## 8. Tecnologias

Python 3.11
NumPy 2.4.6
Pandas 3.0.5
Scikit-learn 1.9.0
Random Forest
Logistic Regression
Decision Tree
HistGradientBoosting
Jupyter Notebook
Git e GitHub
