# Projeto de Iniciação Científica — Redes Neurais com Dados Tabulares

Este projeto faz parte da Iniciação Científica em Engenharia de Software. O objetivo é **aprender e documentar o processo de treinamento de redes neurais com dados tabulares**, utilizando o dataset do Titanic como estudo de caso.

---

## Estrutura do Projeto

O projeto está organizado na seguinte estrutura de pastas para manter o código, dados e documentação separados e organizados.

```
Projeto/
│
├── .venv/                # Ambiente virtual Python
├── data/                 # Arquivos de dados brutos, intermediários e finais (.csv)
├── docs/                 # Documentação de apoio (guias, diário de bordo)
├── models/               # Modelos de machine learning treinados e salvos
├── notebooks/            # Jupyter Notebooks para análise exploratória e prototipagem
├── src/                  # Código fonte principal dos scripts de processamento (.py)
│
├── README.md             # Este arquivo
└── requirements.txt      # Lista de dependências Python do projeto
```

---

## Pipeline de Pré-processamento de Dados

O pré-processamento dos dados é executado em uma sequência de scripts localizados na pasta `src/`. Cada script é responsável por uma etapa da transformação, gerando um arquivo intermediário na pasta `data/`.

### 1. `src/Normalize.py`
-   **Entrada:** `data/train.csv`
-   **Ação:** Cria a feature `HasCabin`, que recebe o valor `1` se o passageiro tinha uma cabine registrada e `0` caso contrário.
-   **Saída:** `data/train_processed.csv`

### 2. `src/Normalize_train_processed.py`
-   **Entrada:** `data/train_processed.csv`
-   **Ação:** Realiza a limpeza avançada de dados nulos:
    -   **Age:** Preenche valores nulos com a mediana da idade correspondente ao título do passageiro (ex: Mr., Mrs., Miss).
    -   **Embarked:** Preenche valores nulos com base no porto de embarque de familiares (mesmo sobrenome). Se não houver, usa o porto mais comum (moda).
    -   **Cabin:** Tenta preencher cabines nulas com base na cabine de familiares e atualiza a coluna `HasCabin`. A coluna `Cabin` original é removida se a taxa de preenchimento for inferior a 50%.
-   **Saída:** `data/train_cleaned_advanced.csv`

### 3. `src/prepare_for_model.py`
-   **Entrada:** `data/train_cleaned_advanced.csv`
-   **Ação:** Finaliza a preparação do dataset para o modelo de machine learning:
    -   Remove colunas não informativas (`Name`, `Ticket`, `PassengerId`).
    -   Normaliza as colunas numéricas `Age` e `Fare` para o intervalo [0, 1].
    -   Converte colunas categóricas (`Sex`, `Pclass`, `Embarked`, `SibSp`) para formato numérico usando mapeamento e One-Hot Encoding.
-   **Saída:** `data/train_final.csv`

---

## Como Executar

1.  **Configurar o Ambiente:**
    -   Certifique-se de ter o Python instalado.
    -   Na pasta raiz do projeto, crie e ative um ambiente virtual:
      ```powershell
      # Criar ambiente (apenas uma vez)
      python -m venv .venv
      # Ativar ambiente (toda vez que for trabalhar no projeto)
      .\.venv\Scripts\Activate.ps1
      ```

2.  **Instalar Dependências:**
    -   Com o ambiente ativado, instale as bibliotecas necessárias:
      ```powershell
      pip install -r requirements.txt
      ```

3.  **Executar o Pipeline:**
    -   Execute os scripts de pré-processamento em ordem a partir da pasta `src/`:
      ```powershell
      cd src
      python Normalize.py
      python Normalize_train_processed.py
      python prepare_for_model.py
      ```
    -   Ao final, o arquivo `data/train_final.csv` estará pronto para ser usado no treinamento do modelo.