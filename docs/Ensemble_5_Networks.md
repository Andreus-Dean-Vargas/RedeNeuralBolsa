# ENSEMBLE 4 - VISUALIZAÇÃO DETALHADA DAS 5 REDES

**Conceito:** Cada rede tem um "papel" diferente no time!

---

## 🔷 REDE 1: LINEAR (A Conservadora)

```
       [15 features]
            ↓
    ┌───────────────┐
    │   8 neurônios │  ← ReLU
    │   (pequena!)  │
    └───────────────┘
            ↓
         [OUTPUT]
         Sigmoid
```

**Parâmetros:** 129

**O QUE ELA FAZ:**
- Apenas 1 camada = aprende relações **DIRETAS** e **SIMPLES**
- Exemplo: "Mulher = sobrevive", "1ª classe = sobrevive"
- **NÃO consegue** capturar interações complexas tipo "mulher E 1ª classe E com família"

**POR QUE NO ENSEMBLE:**
Serve como **baseline simples**. Se os dados tiverem padrões lineares básicos, ela acerta!

**ANALOGIA:** É o jogador que faz o básico bem feito.

---

## 🔶 REDE 2: MODERADA (A Equilibrada)

```
       [15 features]
            ↓
    ┌───────────────┐
    │  16 neurônios │  ← ReLU (captura padrões)
    └───────────────┘
            ↓
    ┌───────────────┐
    │   8 neurônios │  ← ReLU (combina padrões)
    └───────────────┘
            ↓
         [OUTPUT]
         Sigmoid
```

**Parâmetros:** 409

**O QUE ELA FAZ:**
- **1ª camada:** Cada neurônio aprende um "mini-padrão"
  - Neurônio 1: "mulher jovem"
  - Neurônio 2: "1ª classe com família"
  - Neurônio 3: "homem com tarifa alta"
  
- **2ª camada:** Combina esses mini-padrões
  - "SE (mulher jovem) E (1ª classe com família) → SOBREVIVE"

**✅ "Captura interações moderadas"** = consegue ver 2-3 features trabalhando JUNTAS!

**ANALOGIA:** É o jogador versátil que faz jogadas intermediárias.

---

## 🟠 REDE 3: WIDE (A Memorona)

```
       [15 features]
            ↓
    ┌───────────────┐
    │  20 neurônios │  ← ReLU (MUITOS neurônios!)
    │   (WIDE!)     │
    └───────────────┘
            ↓
         [OUTPUT]
         Sigmoid
```

**Parâmetros:** 321

**O QUE ELA FAZ:**
- 20 neurônios em 1 camada = muita "memória" pra guardar detalhes
- Tenta aprender padrões **ESPECÍFICOS** do treino
- **RISCO:** Pode decorar passageiros individuais!

**✅ "Tenta memorizar mais"** = com 20 neurônios, ela tem capacidade de guardar casos específicos, mas...

**✅ "Será corrigida pelas outras"** = quando ela errar por ter decorado, as outras 4 redes votam contra ela!

**EXEMPLO:**
```
Passageiro de teste: Homem, 25 anos, 3ª classe

Rede 3: "Sobrevive!" (0.78)  ← ERRADO, decorou um caso parecido do treino
Rede 1: "Morre" (0.32)       ← padrão simples: homem 3ª = morre
Rede 2: "Morre" (0.41)
Rede 4: "Morre" (0.38)
Rede 5: "Morre" (0.29)

VOTAÇÃO FINAL: (0.78+0.32+0.41+0.38+0.29)/5 = 0.44 → MORRE ✅
```

**ANALOGIA:** É o jogador criativo que às vezes inventa jogadas geniais, mas às vezes tenta coisas malucas. O time corrige.

---

## 🔵 REDE 4: DEEP NARROW (A Profunda)

```
       [15 features]
            ↓
    ┌───────────────┐
    │  10 neurônios │  ← ReLU (1ª camada: padrões básicos)
    └───────────────┘
            ↓
    ┌───────────────┐
    │  10 neurônios │  ← ReLU (2ª camada: combina padrões)
    └───────────────┘
            ↓
    ┌───────────────┐
    │   8 neurônios │  ← ReLU (3ª camada: abstração alta)
    └───────────────┘
            ↓
         [OUTPUT]
         Sigmoid
```

**Parâmetros:** 339

**O QUE ELA FAZ:**
- **3 CAMADAS = PROFUNDA!**
- Cada camada aprende abstrações mais complexas:

**1ª camada:** Padrões simples
- "mulher"
- "tarifa alta"
- "família pequena"

**2ª camada:** Combinações
- "mulher COM tarifa alta"
- "família pequena EM 1ª classe"

**3ª camada:** Abstrações complexas
- "mulher jovem, em 1ª classe, COM família pequena, E tarifa média-alta"

**✅ "Busca padrões profundos"** = consegue ver 4-5 features interagindo de forma complexa!

**EXEMPLO DE PADRÃO PROFUNDO:**
```
SE (Sexo=Feminino) E (Pclass=1) E (FamilySize=2-3) E (Age<40) 
   ENTÃO chance alta de sobreviver
```
Rede rasa não consegue isso! Rede profunda sim!

**ANALOGIA:** É o jogador estrategista que vê jogadas que ninguém mais vê.

---

## 🟣 REDE 5: DROPOUT HEAVY (A Cautelosa)

```
       [15 features]
            ↓
    ┌───────────────┐
    │  12 neurônios │  ← ReLU
    └───────────────┘
            ↓
     ⚡ DROPOUT 0.5 ⚡  ← Desliga 50% dos neurônios!
            ↓
    ┌───────────────┐
    │  12 neurônios │  ← ReLU
    └───────────────┘
            ↓
         [OUTPUT]
         Sigmoid
```

**Parâmetros:** 349

**O QUE ELA FAZ:**
- **Dropout 0.5** = a cada iteração de treino, DESLIGA metade dos neurônios aleatoriamente!

**POR QUE FAZER ISSO?**
- Força a rede a aprender padrões **REDUNDANTES**
- Não pode depender de 1 neurônio específico
- **SUPER RESISTENTE** a overfitting!

**EXEMPLO DE TREINO:**
```
Iteração 1: neurônios [1,2,3,4,5,6] ativos, [7,8,9,10,11,12] desligados
Iteração 2: neurônios [2,4,5,7,9,11] ativos, [1,3,6,8,10,12] desligados
Iteração 3: neurônios [1,3,4,6,8,10] ativos, [2,5,7,9,11,12] desligados
...
```

Cada iteração treina uma "sub-rede" diferente!

**✅ "Extremamente regularizada"** = TÃO cautelosa que quase nunca overfitta!

**TRADE-OFF:**
- ✅ Nunca decora
- ⚠️ Pode perder alguns padrões sutis (underfitting leve)

**ANALOGIA:** É o jogador defensivo que nunca arrisca, sempre seguro.

---

# 🎯 COMO AS 5 TRABALHAM JUNTAS

## VOTAÇÃO EM AÇÃO:

**Exemplo: Passageiro mulher, 35 anos, 2ª classe, família de 3**

```
Rede 1 (Linear):        0.72  "mulher = bom sinal"
Rede 2 (Moderada):      0.81  "mulher + 2ª classe + família = sobrevive"
Rede 3 (Wide):          0.91  "lembra de caso similar no treino"
Rede 4 (Deep):          0.78  "padrão complexo detectado"
Rede 5 (Dropout):       0.65  "cautelosa, mas concorda"

MÉDIA: (0.72+0.81+0.91+0.78+0.65)/5 = 0.774

DECISÃO FINAL: 77.4% de chance → SOBREVIVE ✅
```

---

**Exemplo 2: Homem, 22 anos, 3ª classe, sozinho**

```
Rede 1 (Linear):        0.28  "homem = sinal ruim"
Rede 2 (Moderada):      0.35  "homem + 3ª + sozinho = morre"
Rede 3 (Wide):          0.62  "ERRO! Decorou um outlier"  ⚠️
Rede 4 (Deep):          0.31  "padrão claro: não sobrevive"
Rede 5 (Dropout):       0.24  "muito segura: não sobrevive"

MÉDIA: (0.28+0.35+0.62+0.31+0.24)/5 = 0.36

DECISÃO FINAL: 36% de chance → MORRE ✅
```

**Viu?** A Rede 3 errou (decorou), mas foi **corrigida pelas outras 4**!

---

# 📊 RESUMO VISUAL DOS PAPÉIS

| Rede | Papel | Força | Fraqueza | Quando Brilha |
|------|-------|-------|----------|---------------|
| **1 - Linear** | Baseline | Simples e rápida | Não vê interações | Padrões óbvios |
| **2 - Moderada** | Equilibrada | Balanceada | Nem melhor em nada | Casos gerais |
| **3 - Wide** | Memorona | Guarda detalhes | Pode decorar | Padrões raros |
| **4 - Deep** | Estrategista | Padrões complexos | Pode overfitar | Interações sutis |
| **5 - Dropout** | Cautelosa | Nunca overfitta | Pode underfitar | Dados ruidosos |

---

# 💡 POR QUE ISSO FUNCIONA?

**Princípio fundamental:** "Sabiamente errada é melhor que individualmente certinha"

- Rede 3 acerta 82% sozinha
- Rede 5 acerta 81% sozinha
- **ENSEMBLE acerta 84%** porque erros se cancelam!

**É tipo perguntar pra 5 especialistas diferentes:**
- 1 generalista
- 1 equilibrado
- 1 que conhece casos raros
- 1 estrategista
- 1 super cauteloso

A maioria normalmente acerta! 🎯

---

# 🚀 IMPLEMENTAÇÃO SIMPLIFICADA

```python
# Rede 1: Linear
modelo1 = Sequential([
    Dense(8, activation='relu', input_shape=(15,)),
    Dense(1, activation='sigmoid')
])

# Rede 2: Moderada  
modelo2 = Sequential([
    Dense(16, activation='relu', input_shape=(15,)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Rede 3: Wide
modelo3 = Sequential([
    Dense(20, activation='relu', input_shape=(15,)),
    Dense(1, activation='sigmoid')
])

# Rede 4: Deep
modelo4 = Sequential([
    Dense(10, activation='relu', input_shape=(15,)),
    Dense(10, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Rede 5: Dropout Heavy
modelo5 = Sequential([
    Dense(12, activation='relu', input_shape=(15,)),
    Dropout(0.5),
    Dense(12, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Votação
previsoes = [m1.predict(X), m2.predict(X), m3.predict(X), 
             m4.predict(X), m5.predict(X)]
resultado_final = np.mean(previsoes, axis=0)
```

---

**Agora você ENXERGA como cada rede pensa diferente e como elas se complementam! 🎓**