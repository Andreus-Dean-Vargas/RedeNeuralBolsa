import pandas as pd
import os

# --- Configuração dos Caminhos ---
data_path = os.path.join('..', 'data')
test_input_path = os.path.join(data_path, 'test.csv')
gender_submission_path = os.path.join(data_path, 'gender_submission.csv')
output_csv_path = os.path.join(data_path, 'test_with_labels.csv')

try:
    # --- Carregamento dos Dados ---
    df_test = pd.read_csv(test_input_path)
    df_gender = pd.read_csv(gender_submission_path)
    print("Arquivos 'test.csv' and 'gender_submission.csv' carregados com sucesso.")

    # --- União dos DataFrames ---
    # Unir os dois dataframes com base na coluna 'PassengerId'
    df_merged = pd.merge(df_test, df_gender, on='PassengerId')
    print("DataFrames unidos com sucesso pela coluna 'PassengerId'.")
    
    # Reordenar colunas para ficar semelhante ao 'train.csv' (com 'Survived' na segunda posição)
    cols = df_merged.columns.tolist()
    # Mover 'Survived' para a segunda posição
    survived_col = cols.pop(cols.index('Survived'))
    cols.insert(1, survived_col)
    df_merged = df_merged[cols]


    # --- Verificação e Salvamento ---
    print("\nVisualização das 5 primeiras linhas do dataset unido:")
    print(df_merged.head())

    df_merged.to_csv(output_csv_path, index=False)
    print(f"\nDataset de teste unido salvo com sucesso em: {output_csv_path}")

except FileNotFoundError as e:
    print(f"Erro: Arquivo não encontrado. Verifique o caminho: {e.filename}")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")
