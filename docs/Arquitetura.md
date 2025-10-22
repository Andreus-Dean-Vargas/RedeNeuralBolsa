# 🧠 Arquitetura de Redes Neurais para Classificação Binária (Dados Tabulares)

Este documento apresenta a estratégia escolhida de arquitetura de Redes Neurais (NeuralNetwork) adequada para o problema de classificação binária (Sobrevivência do Titanic) e otimizada para dados tabulares.


---

# 🧠 Arquiteturas de Redes Neurais: Estrutura Otimizada

## 3. Modo 3: Arquitetura Híbrida e Altamente Regularizada

Esta arquitetura é baseada no MLP, mas utiliza **regularização dupla** (Dropout + L2) para máxima robustez e estabilidade.

### Estrutura de Camadas:
- **Camada Oculta 1:** 64 Neurônios com Ativação **ReLU** + **L2 Penalty** (Penalização de Pesos).
- **Camada de Regularização:** **Dropout (30%)**.
- **Camada Oculta 2:** 32 Neurônios com Ativação **ReLU** + **L2 Penalty**.
- **Camada de Regularização:** **Dropout (20%)**.
- **Camada de Saída:** 1 Neurônio com Ativação **Sigmoid**.

*Obs:* O **L2 Penalty** atua nos pesos para simplificar o modelo, complementando o trabalho do Dropout nas ativações.

### Racional da Regularização Dupla
* **Ataque Duplo ao Overfitting:** O **Dropout** atua nas **ativações** (generalização), enquanto o **L2 Regularization** atua nos **pesos** (simplificação do modelo), garantindo a máxima estabilidade e generalização.