import pandas as pd
import os
import re # Importa a biblioteca de expressões regulares para extrair os títulos

# --- Configuração dos Caminhos ---
data_path = os.path.join('..', 'data')
input_csv_path = os.path.join(data_path, 'train_processed.csv')
output_csv_path = os.path.join(data_path, 'train_cleaned_advanced.csv')

try:
    # --- Carregamento dos Dados ---
    df = pd.read_csv(input_csv_path)
    print("Arquivo de entrada carregado com sucesso.")
    print("Valores nulos ANTES do tratamento:")
    print(df.isnull().sum())
    print("-" * 40)

    # --- 1. Tratamento avançado de 'Age' usando Títulos ---
    print("Iniciando tratamento da coluna 'Age'...")
    # Extrai o título do nome usando uma expressão regular
    df['Title'] = df['Name'].apply(lambda name: re.search(' ([A-Za-z]+)\.', name).group(1))
    # Calcula a mediana de idade para cada título
    median_age_by_title = df.groupby('Title')['Age'].median()
    print("Mediana de idade por título:")
    print(median_age_by_title)
    # Preenche as idades nulas usando a mediana do título correspondente
    df['Age'] = df.apply(
        lambda row: median_age_by_title[row['Title']] if pd.isnull(row['Age']) else row['Age'],
        axis=1
    )
    print("'Age' preenchida com base nos títulos.")
    print("-" * 40)

    # --- 2. Tratamento avançado de 'Embarked' usando Família ---
    print("Iniciando tratamento da coluna 'Embarked'...")
    df['LastName'] = df['Name'].apply(lambda name: name.split(',')[0])
    missing_embarked_indices = df[df['Embarked'].isnull()].index

    for index in missing_embarked_indices:
        lastname = df.loc[index, 'LastName']
        # Procura por familiares (mesmo sobrenome) que não sejam a própria pessoa
        family_df = df[(df['LastName'] == lastname) & (df.index != index)]
        
        if not family_df.empty and family_df['Embarked'].notna().any():
            # Se encontrar, usa a moda do porto de embarque da família
            embarked_mode = family_df['Embarked'].mode()[0]
            df.loc[index, 'Embarked'] = embarked_mode
            print(f"Passageiro {df.loc[index, 'PassengerId']}: 'Embarked' preenchido com base na família -> '{embarked_mode}'")
        else:
            # Se não encontrar família ou a família também não tiver dados, usa a moda geral
            overall_mode = df['Embarked'].mode()[0]
            df.loc[index, 'Embarked'] = overall_mode
            print(f"Passageiro {df.loc[index, 'PassengerId']}: 'Embarked' preenchido com moda geral -> '{overall_mode}'")
    print("'Embarked' preenchido.")
    print("-" * 40)

    # --- 3. Tratamento avançado de 'Cabin' e 'HasCabin' usando Família ---
    print("Iniciando tratamento da coluna 'Cabin'...")
    missing_cabin_indices = df[df['Cabin'].isnull()].index

    for index in missing_cabin_indices:
        lastname = df.loc[index, 'LastName']
        ticket = df.loc[index, 'Ticket']
        # Procura por familiares (mesmo sobrenome OU mesmo ticket) que tenham cabine
        family_df = df[((df['LastName'] == lastname) | (df['Ticket'] == ticket)) & (df.index != index) & (df['Cabin'].notna())]

        if not family_df.empty:
            # Pega a primeira cabine encontrada na família
            family_cabin = family_df['Cabin'].iloc[0]
            df.loc[index, 'Cabin'] = family_cabin
            df.loc[index, 'HasCabin'] = 1 # Atualiza o HasCabin
            print(f"Passageiro {df.loc[index, 'PassengerId']}: Cabine '{family_cabin}' atribuída com base na família.")

    # Verificação final da coluna 'Cabin'
    cabin_fill_rate = df['Cabin'].notna().sum() / len(df)
    print(f"\nTaxa de preenchimento da coluna 'Cabin' após tratamento: {cabin_fill_rate:.2%}")

    if cabin_fill_rate < 0.50:
        df.drop('Cabin', axis=1, inplace=True)
        print("Taxa de preenchimento < 50%. Coluna 'Cabin' removida, 'HasCabin' mantida.")
    else:
        print("Taxa de preenchimento >= 50%. Coluna 'Cabin' mantida para análise futura.")
    print("-" * 40)

    # --- Limpeza Final e Salvamento ---
    # Remove colunas auxiliares que criamos
    df.drop(['Title', 'LastName'], axis=1, inplace=True)
    
    print("Valores nulos DEPOIS do tratamento:")
    print(df.isnull().sum())
    print("-" * 40)

    df.to_csv(output_csv_path, index=False)
    print(f"Arquivo limpo (com tratamento avançado) salvo com sucesso em: {output_csv_path}")

except FileNotFoundError:
    print(f"Erro: O arquivo {input_csv_path} não foi encontrado. Verifique o caminho.")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")