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

O pré-processamento agora está concentrado em um único script unificado localizado em `src/normalize.py`.

### `src/normalize.py` (pipeline unificado)

**Entrada:** qualquer CSV estilo Titanic (ex.: `data/train.csv` ou `data/test.csv`)

**Ação (resumo):**

- Cria `HasCabin` a partir de `Cabin` (1/0).
- Imputa `Age` usando a mediana por `Title` extraído de `Name`.
- Imputa `Embarked` por família (sobrenome) quando possível, senão por moda global.
- Tenta preencher `Cabin` por familiares/mesmo ticket e remove a coluna `Cabin` se a taxa de preenchimento for inferior a um limiar (padrão 50%).
- Limpa `Fare` (remove separadores de milhar), imputa por mediana de `Pclass` quando aplicável e normaliza para [0,1] (MinMax).
- Normaliza `Age` (dividindo por 100).
- Converte `Sex` para 0/1 e aplica One-Hot Encoding em colunas categóricas presentes (`Pclass`, `Embarked`, `SibSp`, `Parch`).
- Remove colunas auxiliares/identificadoras por padrão (`Name`, `Ticket`, `PassengerId`, `Title`). A remoção de identificadores é opcional e documentada abaixo.

**Saída:** arquivo CSV processado (especifique com `--output`)

Observação importante: a remoção de colunas de identificação (`PassengerId`, `Ticket`, `Name`) é recomendada para preparar dados de treino para redes neurais, mas deixamos essa remoção como comportamento padrão do script e também documentada como opcional no README.

Nota: remover identificadores é uma etapa de preparação para TREINO da rede neural (para evitar que o modelo use IDs como sinal). Isso NÃO é uma operação matemática de normalização — ou seja, a normalização das features acontece independentemente desta remoção.

---

## Como Executar

**Configurar o Ambiente:**

- Certifique-se de ter o Python instalado.
- Na pasta raiz do projeto, crie e ative um ambiente virtual:

```powershell
# Criar ambiente (apenas uma vez)
python -m venv .venv
# Ativar ambiente (toda vez que for trabalhar no projeto)
.\.venv\Scripts\Activate.ps1
```

**Instalar Dependências:**

- Com o ambiente ativado, instale as bibliotecas necessárias:

```powershell
pip install -r requirements.txt
```

**Executar o Pipeline (exemplo):**

Execute o script unificado `src/normalize.py` informando arquivo de entrada e saída:

```powershell
# dentro da raiz do projeto (exemplo usando o Python do sistema)
python src/normalize.py -i "data/train.csv" -o "data/train_normalized.csv"

# Em Windows usando o launcher py
py -3 src/normalize.py -i "data/train.csv" -o "data/train_normalized.csv"

# Exemplos de flags opcionais:
# --drop-cabin-threshold 0.6        # altera limiar para remoção de Cabin
# --keep-cabin-if-enough           # mantém a coluna Cabin quando taxa >= threshold
```

Resultado: o CSV especificado em `--output` conterá o dataset pronto para o modelo (colunas numéricas normalizadas e features categóricas codificadas).

Nota sobre colunas identificadoras: o script remove `PassengerId`, `Ticket` e `Name` por padrão. Se quiser mantê-las, abra `src/normalize.py` e comente a linha que faz `df.drop(...)` ou peça para eu adicionar uma flag `--keep-ids`.
