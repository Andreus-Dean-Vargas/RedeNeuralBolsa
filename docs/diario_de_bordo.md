# 🚢 Diário de Bordo: Estratégias de Pré-Processamento do Dataset Titanic (Revisão Final)

Este documento detalha as decisões críticas de engenharia de *features*, justificativas e as alternativas rejeitadas para cada etapa de pré-processamento. Descrevendo de forma mais objetiva os pensamentos e decisões tomadas para deixar o dataset apto a treinamento de rede neural.

---

## 1. Tratamento da Coluna `Cabin` (Cabine)

### Decisão:
Criar a *feature* **`HasCabin`** (0/1) e **excluir** `Cabin` original (abaixo do limite, decidido por mim mesmo, de 51% de preenchimento).

### Justificativa:
A alta taxa de dados faltantes tornava a imputação do valor exato inviável. A extração da *feature* `HasCabin` preserva a informação mais valiosa (tem cabine explicita ou não) de forma robusta e binária.

### Alternativa Rejeitada:
Tentar **imputar** a cabine OU **remover** `Cabin` sem criar `HasCabin`.

### Por Que Rejeitar?
Imputar seria especulação excessiva e introduziria ruído. Remover sem extrair `HasCabin` descartaria uma *feature* valiosa que tem correlação comprovada com a sobrevivência.

---

## 2. Tratamento da Coluna `Age` (Idade)

### Decisão:
**Imputar** usando a **mediana** por **Título de Cortesia** ('Mr.', 'Mrs.', etc.) extraído do `Name`.

### Justificativa:
Títulos são fortes *proxies* para idade, permitindo segmentação em grupos etários mais homogêneos. Esta abordagem fornece uma estimativa de idade **mais precisa** do que a média geral, mantendo a distribuição de idade próxima do real.

### Alternativa Rejeitada:
**Imputar** com a **média/mediana geral** do navio OU **remover** as linhas nulas.

### Por Que Rejeitar?
A média geral seria muito imprecisa. Remover as linhas descartaria cerca de 20% do *dataset*, prejudicando o treinamento da Rede Neural.

---

## 3. Tratamento da Coluna `Embarked` (Porto de Embarque)

### Decisão:
**Primeiro** buscar imputar com o porto de embarque de **familiares**; se falhar, usar a **Moda** geral (porto mais utilizado).

### Justificativa:
A lógica familiar é contextualizada, aproveitando a alta probabilidade de familiares embarcarem no mesmo porto. A Moda é usada apenas como último recurso para garantir a ausência de nulos.

### Alternativa Rejeitada:
**Imputar imediatamente** com a **Moda geral** OU **deixar** os nulos.

### Por Que Rejeitar?
A imputação imediata com a moda é um preenchimento "cego". Deixar nulos faria a rede neural falhar.

---

## 4. Tratamento e Padronização da Coluna `Fare` (Tarifa)

### Decisão:
**Padronizar o formato** (remover separadores de milhares), **imputar** o nulo com a **mediana do `Pclass`** e **escalonar** com `MinMaxScaler`.

### Justificativa:
A padronização inicial foi crucial para corrigir a interpretação de strings (o problema dos números gigantes) causada por separadores de milhares no CSV. A imputação por `Pclass` é mais precisa. O escalonamento é essencial para o modelo.

### Alternativa Rejeitada:
Usar o valor de `Fare` **diretamente** (sem padronização) OU aplicar **Transformação Logarítmica**.

### Por Que Rejeitar?
Não padronizar causaria erro/valores massivos na rede neural. A logarítmica foi rejeitada para manter a escala original do dado após a correção de formato.

---

## 5. Normalização de Dados Contínuos 

### Decisão:
Aplicar **escalonamento Min-Max** nas *features* contínuas: `Fare`.
Simples **divisão por 100** para encontrar `Age` entre 0 e 1, já que no dataset nao existe ninguém com mais de 99 anos.

### Justificativa:
É essencial para Redes Neurais que as *features* estejam na mesma escala (entre 0 e 1). Isso evita que *features* com valores maiores dominem o processo de otimização (gradiente).

### Alternativa Rejeitada:
Usar **Z-Score** (StandardScaler) OU **Não normalizar** nenhuma *feature*.

### Por Que Rejeitar?
O Z-Score é mais indicado para dados com distribuição normal. Não normalizar arruinaria a convergência do modelo.

---

## 6. Codificação de Categóricos

### Decisão:
Usar **One-Hot Encoding** para variáveis categóricas (`Pclass`, `Embarked`, `SibSp`) e **mapeamento binário** para `Sex`.

### Justificativa:
O One-Hot Encoding evita que o modelo infira uma **ordem numérica artificial** em categorias sem hierarquia (como o Porto de Embarque), garantindo a correta interpretação dos dados pela rede.

### Alternativa Rejeitada:
Usar **Label Encoding** (atribuir números sequenciais) em todas as variáveis categóricas.

### Por Que Rejeitar?
O Label Encoding criaria uma **ordem artificial/falsa** para categorias sem ordem, levando a conclusões incorretas no treinamento.

