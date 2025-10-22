import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

# --- Configuração dos Caminhos ---
data_path = os.path.join('..', 'data')
input_csv_path = os.path.join(data_path, 'train_cleaned_advanced.csv')
output_csv_path = os.path.join(data_path, 'train_final.csv')

try:
    # --- Carregamento dos Dados ---
    df = pd.read_csv(input_csv_path)
    print("Arquivo limpo carregado com sucesso.")
    print("-" * 40)

    # --- 1. Seleção e Remoção de Colunas ---
    cols_to_drop = ['Name', 'Ticket', 'PassengerId']
    if 'Cabin' in df.columns:
        cols_to_drop.append('Cabin')
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    print(f"Colunas removidas: {cols_to_drop}")

    # --- 2. Normalização de Features Contínuas ---

    # Normalizar 'Age' para o intervalo [0, 1]
    df['Age'] = df['Age'] / 100
    print("Coluna 'Age' normalizada (dividida por 100).")
    
    # ----------------------------------------------------
    # Tratamento de Formato, Imputação e Normalização de 'Fare'
    # ----------------------------------------------------
    
    # Passo 1: Correção e Limpeza do Formato (PADRONIZAÇÃO)
    # Este passo corrige o problema dos números gigantes, removendo pontos 
    # que estão sendo interpretados erroneamente como separadores de milhares.
    df['Fare'] = df['Fare'].astype(str).str.replace('.', '', regex=False)
    # Converte a coluna para o tipo numérico, transformando erros de conversão em NaN (nulo).
    df['Fare'] = pd.to_numeric(df['Fare'], errors='coerce') 
    print("Correção de formato da coluna 'Fare' realizada (removendo separadores de milhares e convertendo para numérico).")
    
    
    # Passo 2: Imputação do Valor Faltante (Usando Pclass como critério, como definido)
    if df['Fare'].isnull().any():
        # Calcula a mediana de Fare agrupando pela Classe do Passageiro (Pclass)
        median_fare_by_pclass = df.groupby('Pclass')['Fare'].median()
        df['Fare'] = df.apply(
            # Preenche nulos (NaN) usando a mediana da respectiva Pclass
            lambda row: median_fare_by_pclass[row['Pclass']] if pd.isnull(row['Fare']) else row['Fare'],
            axis=1
        )
        print("Valor nulo em 'Fare' imputado com a mediana de sua respectiva 'Pclass'.")
    
    
    # Passo 3: Normalização Final (Escalonamento Min-Max para o intervalo [0, 1])
    # Mantendo o escalonamento Min-Max padrão, sem transformação logarítmica.
    scaler = MinMaxScaler()
    df['Fare'] = scaler.fit_transform(df[['Fare']])
    print("Coluna 'Fare' normalizada para o intervalo [0, 1] usando MinMaxScaler.")
    
    # ----------------------------------------------------
    # Fim do Tratamento de 'Fare'
    # ----------------------------------------------------

    # --- 3. Codificação de Atributos Categóricos ---
    # Mapeamento da coluna 'Sex'
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).astype(int)
    print("Coluna 'Sex' convertida para formato numérico (0/1).")

    # One-Hot Encoding das colunas 'Pclass', 'Embarked' e 'SibSp'
    # dtype=int garante que o resultado seja 1 e 0.
    categorical_cols = ['Pclass', 'Embarked', 'SibSp']
    df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols, dtype=int)
    print(f"Colunas {categorical_cols} convertidas usando One-Hot Encoding.")
    print("-" * 40)

    # --- Verificação Final e Salvamento ---
    print("Dataset final pronto para o modelo.")
    print("Colunas finais:", df.columns.tolist())
    print("\nVisualização das 5 primeiras linhas do dataset final:")
    print(df.head())

    df.to_csv(output_csv_path, index=False)
    print(f"\nDataset final salvo com sucesso em: {output_csv_path}")

except FileNotFoundError:
    print(f"Erro: O arquivo {input_csv_path} não foi encontrado. Verifique o caminho.")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")