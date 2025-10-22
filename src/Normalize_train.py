import pandas as pd
import os

# Define o caminho para os arquivos de dados
# O '..' significa "voltar uma pasta", saindo de 'src' para a pasta principal do projeto.
data_path = os.path.join('..', 'data')
train_csv_path = os.path.join(data_path, 'train.csv')
output_csv_path = os.path.join(data_path, 'train_processed.csv')

print(f"Lendo o arquivo: {train_csv_path}")

# Carrega a planilha de treino para um DataFrame do pandas
try:
    df = pd.read_csv(train_csv_path)

    # --- Passo 1: Criar a coluna 'HasCabin' ---
    # A coluna 'Cabin' tem valores em branco (NaN) para quem não tinha cabine registrada.
    # O método .notna() retorna True para quem tem um valor e False para quem não tem.
    # Em seguida, convertemos True/False para 1/0, que é o formato que a rede neural entende.
    df['HasCabin'] = df['Cabin'].notna().astype(int)

    print("Coluna 'HasCabin' criada com sucesso.")
    print("Visualização das primeiras linhas com a nova coluna:")
    print(df[['PassengerId', 'Cabin', 'HasCabin']].head())

    # Salva o DataFrame modificado em um novo arquivo CSV
    df.to_csv(output_csv_path, index=False)
    print(f"\nArquivo processado salvo com sucesso em: {output_csv_path}")

except FileNotFoundError:
    print(f"Erro: O arquivo {train_csv_path} não foi encontrado. Verifique o caminho.")

except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")