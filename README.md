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

PELO MENOS 30 MESES COM STATUS = 0, de 1 a 29 dias de atraso (recorrência de atraso moderada)

Caso nenhuma dessas condições seja atendida, o cliente recebe TARGET = 0.

Os valores C (quitado) e X(Sem informação) do histórico de crédito foram convertidos para 0, 
juntamente com o STATUS = 0. Portanto, a condição MESES_STATUS_0 >= 30 considera os registros 
convertidos para STATUS_NUM = 0.

A base final utilizada na modelagem apresentou:

36.457 clientes
51 variáveis preditoras
65,81% de clientes na classe 0
34,19% de clientes na classe 1

### Dataset

| Campo | Valor |
|---|---|
| Fonte | <!-- PREENCHER: URL --> |
| Linhas × colunas | <!-- PREENCHER --> |
| Período / versão | <!-- PREENCHER --> |
| Licença de uso | <!-- PREENCHER --> |

Descrição das variáveis:

| Variável | Tipo | Descrição |
|---|---|---|
| | | |

---

## 4. Como reproduzir

```bash
git clone <URL_DO_REPOSITORIO>
cd <NOME_DO_REPOSITORIO>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook
```

Baixe o dataset e coloque o arquivo bruto em `data/raw/` (os dados **não** são versionados —
veja `data/README.md`).

Depois execute os notebooks nesta ordem:

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

| Modelo | Acurácia | Precisão | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| <!-- PREENCHER --> | | | | | |
| | | | | | |

**Modelo escolhido:** <!-- PREENCHER --> — <!-- PREENCHER: por quê. -->

**Métricas priorizadas:** <!-- PREENCHER: justifique a escolha considerando o
     desbalanceamento de classes e o custo de cada tipo de erro no contexto do negócio. -->

---

## 6. Principais conclusões

<!-- PREENCHER: 3 a 5 conclusões em linguagem de negócio.
     Inclua quais variáveis mais influenciam o resultado e o que isso significa
     na prática para quem vai usar o modelo. -->

1.
2.
3.

### Limitações e próximos passos

<!-- PREENCHER -->

---

## 7. Estrutura do repositório

```
.
├── data/          dados brutos (raw) e tratados (processed) — não versionados
├── notebooks/     análise em ordem numerada
└── docs/          apresentação executiva
```

Detalhes e convenções em [`ESTRUTURA.md`](ESTRUTURA.md).
Antes de enviar, percorra o [`CHECKLIST.md`](CHECKLIST.md).

---

## 8. Tecnologias

<!-- PREENCHER: Python 3.11, pandas, scikit-learn, ... -->
