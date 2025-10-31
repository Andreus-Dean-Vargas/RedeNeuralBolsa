# 4 PROPOSTAS DE ARQUITETURAS - TITANIC DATASET

**Dataset:** 891 amostras × 15 features  
**Problema:** Classificação binária (sobreviveu: 0 ou 1)  
**Desafio:** Poucos dados = alto risco de overfitting

---

## 🎯 ESTRATÉGIA GERAL

**2 TIPOS de abordagem:**
- **TIPO A - MLP (Multi-Layer Perceptron):** Rede única com camadas densas
- **TIPO B - ENSEMBLE:** Múltiplas redes pequenas votando juntas

**Cada tipo terá 2 variações** = 4 caminhos totais

**Regra:** Com 891 amostras, limite conservador = ~300 parâmetros (cada parâmetro vê ~3 exemplos)

---

# TIPO A - MLP (Multi-Layer Perceptron)

## 📊 ARQUITETURA 1: MLP MINIMALISTA

**Estrutura:**
```
Input (15) → Dense(16, ReLU) → Output(1, Sigmoid)
```

**Parâmetros:**
- Camada 1: 15 × 16 + 16 = 256
- Camada 2: 16 × 1 + 1 = 17
- **TOTAL: 273 parâmetros**

**Justificativa:**
- ✅ Relação 891/273 = **3.26 amostras por parâmetro** (seguro!)
- ✅ Apenas 1 camada oculta = menos risco de overfitting
- ✅ 16 neurônios suficientes para capturar padrões principais (Sex, Pclass, Age, FamilySize)
- ✅ Arquitetura mais **simples e interpretável**

**Por que essa escolha?**
Com poucos dados, simplicidade é vantagem. Esta rede aprende apenas padrões essenciais sem decorar ruídos.

**Técnicas anti-overfitting:**
- Dropout: 0.2 (desliga 20% dos neurônios no treino)
- L2 Regularization: 0.01 (penaliza pesos grandes)

**Previsão de performance:** 80-82% acurácia

---

## 📊 ARQUITETURA 2: MLP MODERADA

**Estrutura:**
```
Input (15) → Dense(32, ReLU) → Dense(16, ReLU) → Output(1, Sigmoid)
```

**Parâmetros:**
- Camada 1: 15 × 32 + 32 = 512
- Camada 2: 32 × 16 + 16 = 528
- Camada 3: 16 × 1 + 1 = 17
- **TOTAL: 1057 parâmetros**

**Justificativa:**
- ⚠️ Relação 891/1057 = **0.84 amostras por parâmetro** (RISCO!)
- ✅ 2 camadas ocultas = pode capturar **interações não-lineares** entre features
- ✅ Arquitetura "afunilada" (32→16→1) = compressão progressiva de informação
- ⚠️ PRECISA de Dropout/Regularização forte para funcionar

**Por que essa escolha?**
Testa se padrões mais complexos existem nos dados. Se houver interações sutis (ex: "mulheres de 1ª classe COM família pequena"), essa rede pode capturar.

**Técnicas anti-overfitting (OBRIGATÓRIAS):**
- Dropout: 0.4 nas camadas (desliga 40% dos neurônios)
- L2 Regularization: 0.02
- Early Stopping: para treino se validação não melhorar por 20 epochs

**Previsão de performance:** 79-83% acurácia (alta variância - pode overfitar OU superar a MLP 1)

---

# TIPO B - ENSEMBLE NEURAL

**Conceito:** Ao invés de 1 rede grande, treinar **várias redes pequenas** com:
- Inicializações aleatórias diferentes
- Subsets ligeiramente diferentes dos dados (bagging)
- Arquiteturas levemente variadas

**Decisão final:** Votação por maioria ou média das probabilidades

**Por que Ensemble reduz overfitting?**
- Cada rede "decora" coisas diferentes
- Erros individuais se cancelam na votação
- Aumenta robustez e confiabilidade

---

## 📊 ARQUITETURA 3: ENSEMBLE CONSERVADOR

**Estrutura:** 3 redes idênticas com inicializações diferentes

**Cada rede:**
```
Input (15) → Dense(12, ReLU) → Output(1, Sigmoid)
```

**Parâmetros POR REDE:**
- Camada 1: 15 × 12 + 12 = 192
- Camada 2: 12 × 1 + 1 = 13
- **Por rede: 205 parâmetros**

**TOTAL ENSEMBLE: 3 × 205 = 615 parâmetros**

**Justificativa:**
- ✅ Cada rede individual é PEQUENA (205 params)
- ✅ 3 redes = equilíbrio entre diversidade e custo computacional
- ✅ Votação reduz overfitting individual de cada rede
- ✅ Mais robusto que MLP 1, mas ainda conservador

**Como funciona a votação:**
```python
Rede 1: 0.82 (sobrevive)
Rede 2: 0.45 (morre)
Rede 3: 0.78 (sobrevive)
MÉDIA: 0.68 → SOBREVIVE ✅
```

**Técnicas anti-overfitting:**
- Bagging: cada rede treina com 80% dos dados (aleatório)
- Dropout: 0.3
- L2 Regularization: 0.01

**Previsão de performance:** 81-83% acurácia (mais estável que MLPs)

---

## 📊 ARQUITETURA 4: ENSEMBLE DIVERSIFICADO

**Estrutura:** 5 redes com ARQUITETURAS VARIADAS

**Rede 1 (Linear):**
```
Input (15) → Dense(8, ReLU) → Output(1, Sigmoid)
Parâmetros: 129
```

**Rede 2 (Moderada):**
```
Input (15) → Dense(16, ReLU) → Dense(8, ReLU) → Output(1, Sigmoid)
Parâmetros: 409
```

**Rede 3 (Wide):**
```
Input (15) → Dense(20, ReLU) → Output(1, Sigmoid)
Parâmetros: 321
```

**Rede 4 (Deep Narrow):**
```
Input (15) → Dense(10, ReLU) → Dense(10, ReLU) → Dense(8, ReLU) → Output(1, Sigmoid)
Parâmetros: 339
```

**Rede 5 (Dropout Heavy):**
```
Input (15) → Dense(12, ReLU) → Dropout(0.5) → Dense(12, ReLU) → Output(1, Sigmoid)
Parâmetros: 349
```

**TOTAL ENSEMBLE: 1547 parâmetros** (mas distribuídos!)

**Justificativa:**
- ✅ **DIVERSIDADE MÁXIMA:** cada rede tem viés diferente
- ✅ Rede 1 captura padrões simples/lineares
- ✅ Rede 2 captura interações moderadas
- ✅ Rede 3 tenta memorizar mais (mas será "corrigida" pelas outras)
- ✅ Rede 4 busca padrões profundos
- ✅ Rede 5 é extremamente regularizada
- ✅ Votação de 5 é mais robusta que 3

**Por que essa escolha?**
Filosofia: "não sabemos qual padrão funciona melhor, então testamos TODOS simultaneamente". A votação escolhe o melhor coletivamente.

**Como funciona a votação:**
```python
Rede 1: 0.45 (morre)
Rede 2: 0.78 (sobrevive)
Rede 3: 0.92 (sobrevive)
Rede 4: 0.55 (sobrevive)
Rede 5: 0.41 (morre)
MÉDIA: 0.62 → SOBREVIVE ✅
```

**Técnicas anti-overfitting:**
- Bagging: cada rede treina com subset diferente
- Dropout variável (0.3 a 0.5)
- L2 Regularization: 0.01-0.02
- Early Stopping individual por rede

**Previsão de performance:** 82-84% acurácia (mais alto potencial, mas mais complexo)

---

# 📋 COMPARAÇÃO RESUMIDA

| Arquitetura | Parâmetros | Amostras/Param | Risco Overfit | Complexidade | Acurácia Esperada |
|-------------|------------|----------------|---------------|--------------|-------------------|
| **MLP 1 - Minimalista** | 273 | 3.26 | 🟢 BAIXO | Simples | 80-82% |
| **MLP 2 - Moderada** | 1057 | 0.84 | 🔴 ALTO | Média | 79-83% |
| **Ensemble 3 - Conservador** | 615 | 1.45 | 🟡 MÉDIO | Média | 81-83% |
| **Ensemble 4 - Diversificado** | 1547 | 0.58 | 🟡 MÉDIO | Alta | 82-84% |

---

# 🎯 RECOMENDAÇÃO PARA DEBATE

**ORDEM DE TESTE:**

1️⃣ **MLP 1 (Minimalista)** → Baseline simples e seguro  
2️⃣ **Ensemble 3 (Conservador)** → Teste se ensemble vale a pena  
3️⃣ **Ensemble 4 (Diversificado)** → Máximo potencial, aceita risco  
4️⃣ **MLP 2 (Moderada)** → Só testar se os outros decepcionar (alto risco)

---

# 💡 PONTOS PARA DISCUTIR COM O PROFESSOR

**1. Trade-off Bias-Variance:**
- MLP 1 = alto bias, baixa variance (underfitting leve, mas estável)
- MLP 2 = baixo bias, alta variance (pode overfitar)
- Ensembles = equilibram bias-variance naturalmente

**2. Por que 2 MLPs e 2 Ensembles?**
- MLPs: para entender se complexidade ajuda ou atrapalha
- Ensembles: para entender se diversidade é melhor que profundidade

**3. Justificativa estatística:**
- 891 amostras / 273 params = cada peso vê 3+ exemplos ✅
- 891 amostras / 1057 params = cada peso vê <1 exemplo ⚠️

**4. Porque Ensemble pode usar mais parâmetros totais?**
- Parâmetros estão **distribuídos** em redes independentes
- Cada rede individualmente é pequena
- Votação cancela overfitting individual

**5. Experimento controlado:**
- Treinar as 4 arquiteturas com MESMOS dados
- Comparar curvas de learning (train vs validation loss)
- Analisar onde cada uma overfitta

---

# 🚀 PRÓXIMOS PASSOS

Após escolher 1 das 4:

1. Implementar em Keras/TensorFlow
2. Treinar com validação cruzada (k-fold)
3. Plotar curvas de aprendizado
4. Comparar com baseline (Random Forest do chat anterior = 81-83%)
5. Escolher hiperparâmetros finais
6. Testar no conjunto de teste

---

**Documento criado para projeto de ML - Dataset Titanic**  
*Data: 31 de Outubro de 2025*
