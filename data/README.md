# data/

**Nada aqui é versionado.** O `.gitignore` bloqueia o conteúdo destas pastas de propósito:
datasets em Git incham o repositório e frequentemente violam a licença da fonte.

| Pasta | Conteúdo |
|---|---|
| `raw/` | arquivo original, exatamente como baixado da fonte — nunca editado |
| `processed/` | saída dos notebooks de pré-processamento (`.parquet` ou `.csv`) |

Documente abaixo como obter os dados brutos, para que qualquer pessoa consiga reproduzir o projeto.

## Como obter

1. Baixe em: https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction
2. Salve como: `data/raw/application_record.csv e data/raw/credit_record.csv`
3. Checksum (opcional, recomendado): `shasum -a 256 data/raw/<arquivo>`

### Checksum

Os arquivos brutos foram obtidos a partir do dataset disponível no Kaggle e tiveram sua integridade verificada por meio do algoritmo SHA-256.

| Arquivo | SHA-256 |
|---|---|
| `data/raw/application_record.csv` | `4833F502D02AD94295DE3FFE74F665E726A4B04342D2E94F8CEC41DCE951925B` |
| `data/raw/credit_record.csv` | `BA0006A4734F74422D68B0A7132AD591850BE0A6AFFB535EB1042D207FE4B27E` |