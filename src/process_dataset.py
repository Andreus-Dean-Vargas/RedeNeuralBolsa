import pandas as pd
import os
import re
from sklearn.preprocessing import MinMaxScaler

def select_path_interactively(prompt_message, select_file=True):
    """
    Permite ao usuário navegar pelo sistema de arquivos e selecionar um arquivo ou diretório.
    """
    # Começa no diretório pai de 'src', que é a raiz do projeto.
    current_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    while True:
        print("\n" + "="*60) print(f"Diretório atual: {current_path}")
        print(prompt_message)
        print("="*60)
        
        try:
            items = os.listdir(current_path)
        except OSError as e:
            print(f"Erro ao acessar o diretório: {e}")
            current_path = os.path.dirname(current_path) # Sobe um nível em caso de erro
            continue

        # Separa pastas e arquivos
        dirs = sorted([d for d in items if os.path.isdir(os.path.join(current_path, d))])
        files = sorted([f for f in items if os.path.isfile(os.path.join(current_path, f))])
        
        options = {0: os.path.dirname(current_path)} # Opção 0 sempre será para voltar
        i = 1
        
        print("0. .. (Voltar)")
        
        # Lista as pastas
        for d in dirs:
            options[i] = os.path.join(current_path, d)
            print(f"{i}. [PASTA] {d}")
            i += 1
            
        # Lista os arquivos apenas se o objetivo for selecionar um arquivo
        if select_file:
            for f in files:
                if f.endswith('.csv'): # Mostra apenas arquivos CSV para facilitar
                    options[i] = os.path.join(current_path, f)
                    print(f"{i}. [ARQUIVO] {f}")
                    i += 1
        
        if not select_file:
             print("\n(Navegue até a pasta onde quer salvar e digite 's' para confirmar)")

        try:
            choice = input("Sua escolha: ")
            if not select_file and choice.lower() == 's':
                return current_path

            choice_num = int(choice)
            selected_path = options.get(choice_num)

            if selected_path is None:
                print("Opção inválida. Tente novamente.")
                continue

            if os.path.isdir(selected_path):
                current_path = selected_path
            elif select_file and os.path.isfile(selected_path):
                return selected_path # Retorna o caminho do arquivo selecionado
            else:
                print("Seleção inválida.")

        except (ValueError, KeyError):
            print("Entrada inválida. Por favor, digite um número da lista ou 's' quando aplicável.")
        except Exception as e:
            print(f"Ocorreu um erro: {e}")
            return None


def get_title(name):
    """Extrai o título de um nome usando regex de forma segura."""
    match = re.search(r' ([A-Za-z]+)\.', str(name))
    if match:
        return match.group(1)
    return "Unknown"

def process_dataset(input_path, output_path):
    """
    Executa um pipeline completo de pré-processamento em um dataset estilo Titanic.
    Combina as etapas de normalização, limpeza avançada e preparação para o modelo.
    """
    try:
        # --- 1. Carregamento dos Dados ---
        print(f"\nCarregando dataset de: {input_path}")
        df = pd.read_csv(input_path)
        print("Dataset carregado com sucesso.")
        print("-" * 40)

        # --- ETAPA A: Normalização Inicial ---
        print("Iniciando Etapa A: Normalização Inicial...")
        if 'Cabin' in df.columns:
            df['HasCabin'] = df['Cabin'].notna().astype(int)
            print("Coluna 'HasCabin' criada.")
        else:
            df['HasCabin'] = 0
            print("Coluna 'Cabin' não encontrada. 'HasCabin' criada com valor 0.")
        print("-" * 40)

        # --- ETAPA B: Limpeza Avançada de Nulos ---
        print("Iniciando Etapa B: Limpeza Avançada de Nulos...")
        if 'Name' in df.columns:
            df['Title'] = df['Name'].apply(get_title)
            median_age_by_title = df.groupby('Title')['Age'].median()
            df['Age'] = df.apply(
                lambda row: median_age_by_title.get(row['Title'], df['Age'].median()) if pd.isnull(row['Age']) else row['Age'],
                axis=1
            )
            print("Coluna 'Age' tratada com base no título.")

        if 'Embarked' in df.columns and df['Embarked'].isnull().any():
            embarked_mode = df['Embarked'].mode()[0]
            df['Embarked'].fillna(embarked_mode, inplace=True)
            print(f"Coluna 'Embarked' tratada (preenchida com '{embarked_mode}').")
        
        if 'Fare' in df.columns and df['Fare'].isnull().any():
            fare_median = df['Fare'].median()
            df['Fare'].fillna(fare_median, inplace=True)
            print(f"Coluna 'Fare' tratada (preenchida com a mediana {fare_median}).")

        if 'Cabin' in df.columns:
            if df['Cabin'].notna().sum() / len(df) < 0.50:
                df.drop('Cabin', axis=1, inplace=True)
                print("Coluna 'Cabin' removida por baixa taxa de preenchimento.")
        print("-" * 40)

        # --- ETAPA C: Preparação para o Modelo ---
        print("Iniciando Etapa C: Preparação Final para o Modelo...")
        cols_to_drop = ['Name', 'Ticket', 'PassengerId', 'Title']
        df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True, errors='ignore')
        print(f"Colunas de identificação removidas.")

        # Normalização de colunas numéricas (APÓS preencher nulos)
        if 'Age' in df.columns:
            df['Age'] = df['Age'] / 100
            print("Coluna 'Age' normalizada (dividida por 100).")
        if 'Fare' in df.columns:
            scaler = MinMaxScaler()
            df['Fare'] = scaler.fit_transform(df[['Fare']])
            print("Coluna 'Fare' normalizada para o intervalo [0, 1].")

        # Codificação de colunas categóricas
        if 'Sex' in df.columns:
            df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).astype(int)
            print("Coluna 'Sex' codificada.")
        
        categorical_cols = [col for col in ['Pclass', 'Embarked', 'SibSp', 'Parch'] if col in df.columns]
        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols, dtype=int)
            print(f"Colunas {categorical_cols} codificadas com One-Hot Encoding.")
        print("-" * 40)

        # --- Finalização ---
        print("Pipeline de processamento concluído.")
        print("Colunas finais:", df.columns.tolist())
        
        df.to_csv(output_path, index=False)
        print(f"\nDataset final salvo com sucesso em: {output_path}")

    except FileNotFoundError:
        print(f"Erro: O arquivo de entrada {input_path} não foi encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado durante o processamento: {e}")

if __name__ == "__main__":
    input_file = select_path_interactively("Selecione o arquivo CSV de ENTRADA:")
    
    if input_file:
        output_dir = select_path_interactively("Selecione a pasta de SAÍDA (e pressione 's'):", select_file=False)
        if output_dir:
            output_filename = input("Digite o nome para o arquivo de saída (ex: 'meu_arquivo_final.csv'): ")
            if output_filename:
                output_file = os.path.join(output_dir, output_filename)
                process_dataset(input_file, output_file)

    print("\nScript finalizado.")
