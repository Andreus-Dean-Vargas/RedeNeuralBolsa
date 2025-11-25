# Projeto Titanic - Machine Learning

## Objetivo
Treinar redes neurais (MLPs e Ensembles) para prever sobrevivência no Titanic.

## Dataset
- **Fonte:** train_final.csv (preprocessado)
- **Amostras:** 891
- **Features:** 15 (já normalizadas 0-1)
- **Target:** Survived (0=morreu, 1=sobreviveu)

## Distribuição das Classes
- Morreu (0): ~62%
- Sobreviveu (1): ~38%

## Reprodutibilidade
- **SEED:** 42 (fixada em todos os scripts)
- **Split:** 80% treino / 20% validação (estratificado)
- **Índices salvos em:** `splits/split_indices.json`

## Como Reproduzir

### 1. Preparar ambiente (Semana 1)
```bash
jupyter notebook notebooks/semana1_setup.ipynb
```

Isso cria:
- Estrutura de pastas
- Split estratificado (80/20)
- Salva índices para reuso
- Datasets separados em `data/processed/`

### 2. Verificar splits
Os mesmos índices serão usados em TODAS as semanas para garantir comparabilidade!

## Estrutura de Pastas
```
.
├── data/
│   ├── raw/              # Dados originais
│   └── processed/        # X_train, X_val, y_train, y_val
├── splits/               # Índices dos splits
├── artifacts/            # Modelos treinados, pipelines
├── reports/              # Resultados, métricas
│   └── figures/          # Gráficos
├── src/                  # Scripts Python
└── notebooks/            # Jupyter notebooks
```

## Próximos Passos
- **Semana 2:** Planejar hiperparâmetros
- **Semana 3:** Treinar MLP 1
- **Semana 4:** Análise de métricas
- **Semana 5:** K-fold validation

## Arquiteturas Propostas
Ver documento: `docs/4_propostas_arquiteturas_titanic.md`
