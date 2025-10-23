import os
import re
import argparse
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def get_title(name):
    match = re.search(r' ([A-Za-z]+)\.', str(name))
    if match:
        return match.group(1)
    return 'Unknown'


def process_pipeline(df, drop_cabin_threshold=0.5, keep_cabin_if_enough=True):
    """Applies a consolidated preprocessing pipeline to a Titanic-like DataFrame.

    Steps merged from the project's existing scripts:
    - create HasCabin
    - advanced Age imputation by Title
    - Embarked imputation by family or overall mode
    - Cabin family-based fill and optional drop based on threshold
    - Fare cleaning, imputation by Pclass, and MinMax scaling
    - Age scaling (divide by 100)
    - Sex mapping and one-hot for categorical cols
    - drop identification columns
    """

    # --- HasCabin ---
    if 'Cabin' in df.columns:
        df['HasCabin'] = df['Cabin'].notna().astype(int)
    else:
        df['HasCabin'] = 0

    # --- Advanced Age by Title ---
    if 'Name' in df.columns and 'Age' in df.columns:
        df['Title'] = df['Name'].apply(get_title)
        median_age_by_title = df.groupby('Title')['Age'].median()
        df['Age'] = df.apply(
            lambda row: median_age_by_title[row['Title']] if pd.isnull(row['Age']) and row['Title'] in median_age_by_title.index else row['Age'],
            axis=1
        )

    # --- Embarked imputation using family when possible, else mode ---
    if 'Embarked' in df.columns and df['Embarked'].isnull().any():
        if 'Name' in df.columns:
            df['LastName'] = df['Name'].apply(lambda name: str(name).split(',')[0])
            for idx in df[df['Embarked'].isnull()].index:
                lastname = df.loc[idx, 'LastName']
                family_df = df[(df['LastName'] == lastname) & (df.index != idx)]
                if not family_df.empty and family_df['Embarked'].notna().any():
                    df.loc[idx, 'Embarked'] = family_df['Embarked'].mode()[0]
                else:
                    df.loc[idx, 'Embarked'] = df['Embarked'].mode()[0]
            df.drop(columns=['LastName'], inplace=True, errors='ignore')
        else:
            df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # --- Cabin family-based fill and optional drop ---
    if 'Cabin' in df.columns:
        missing_cabin_indices = df[df['Cabin'].isnull()].index
        for idx in missing_cabin_indices:
            lastname = None
            ticket = None
            if 'Name' in df.columns:
                lastname = df.loc[idx, 'Name'].split(',')[0]
            if 'Ticket' in df.columns:
                ticket = df.loc[idx, 'Ticket']

            family_df = df[
                ((df['Name'].apply(lambda x: str(x).split(',')[0]) == lastname) if lastname is not None else False)
                | ((df['Ticket'] == ticket) if ticket is not None else False)
            ]
            family_df = family_df[(family_df.index != idx) & (family_df['Cabin'].notna())]
            if not family_df.empty:
                df.loc[idx, 'Cabin'] = family_df['Cabin'].iloc[0]
                df.loc[idx, 'HasCabin'] = 1

        cabin_fill_rate = df['Cabin'].notna().sum() / len(df)
        if not keep_cabin_if_enough or cabin_fill_rate < drop_cabin_threshold:
            # drop Cabin but keep HasCabin
            if 'Cabin' in df.columns:
                df.drop('Cabin', axis=1, inplace=True)

    # --- Fare cleaning / imputation / scaling ---
    if 'Fare' in df.columns:
        # remove thousands separators if present (some files use '.' as thousand sep)
        df['Fare'] = df['Fare'].astype(str).str.replace('.', '', regex=False)
        df['Fare'] = pd.to_numeric(df['Fare'], errors='coerce')

        if df['Fare'].isnull().any() and 'Pclass' in df.columns:
            median_fare_by_pclass = df.groupby('Pclass')['Fare'].median()
            df['Fare'] = df.apply(
                lambda row: median_fare_by_pclass[row['Pclass']] if pd.isnull(row['Fare']) else row['Fare'],
                axis=1
            )
        elif df['Fare'].isnull().any():
            df['Fare'].fillna(df['Fare'].median(), inplace=True)

        scaler = MinMaxScaler()
        df['Fare'] = scaler.fit_transform(df[['Fare']])

    # --- Age scaling ---
    if 'Age' in df.columns:
        df['Age'] = df['Age'] / 100

    # --- Sex mapping ---
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).astype(int)

    # --- One-hot encoding for selected categorical columns ---
    # --- FamilySize feature (novo) ---
    # FamilySize = SibSp + Parch + 1 (inclui a própria pessoa)
    # Usamos df.get para garantir que uma Series exista mesmo se as colunas não
    # estiverem presentes; assim FamilySize será ao menos 1.
    sibsp = df.get('SibSp', pd.Series(0, index=df.index))
    parch = df.get('Parch', pd.Series(0, index=df.index))
    df['FamilySize'] = sibsp.fillna(0).astype(int) + parch.fillna(0).astype(int) + 1

    # Após criar FamilySize, removemos SibSp e Parch do dataset (apenas FamilySize é mantido)
    if 'SibSp' in df.columns:
        df.drop(columns=['SibSp'], inplace=True, errors='ignore')
    if 'Parch' in df.columns:
        df.drop(columns=['Parch'], inplace=True, errors='ignore')

    # Agrupar FamilySize >= 5 em '5+' (reduz dimensionalidade)
    df['FamilySize_cat'] = df['FamilySize'].apply(lambda x: '5+' if x >= 5 else str(x))

    # One-hot encoding para Pclass e Embarked (se presentes)
    cat_pe = [col for col in ['Pclass', 'Embarked'] if col in df.columns]
    if cat_pe:
        df = pd.get_dummies(df, columns=cat_pe, prefix=cat_pe, dtype=int)

    # One-hot para FamilySize_cat com prefixo 'FamilySize' (agrupa '5+' como '5+')
    if 'FamilySize_cat' in df.columns:
        fam_dummies = pd.get_dummies(df['FamilySize_cat'], prefix='FamilySize', dtype=int)
        # rename FamilySize_5+ to FamilySize_5plus if exists
        if 'FamilySize_5+' in fam_dummies.columns:
            fam_dummies.rename(columns={'FamilySize_5+': 'FamilySize_5plus'}, inplace=True)
        # concat e remover a coluna categórica original
        df = pd.concat([df, fam_dummies], axis=1)
        df.drop(columns=['FamilySize_cat'], inplace=True, errors='ignore')

    # Remover a coluna numérica original FamilySize (mantemos apenas as one-hot)
    if 'FamilySize' in df.columns:
        df.drop(columns=['FamilySize'], inplace=True, errors='ignore')

    # --- Drop identification/auxiliary columns ---
    # Observação: a remoção de colunas de identificação (ex: PassengerId, Ticket, Name)
    # é considerada um requisito para preparar o dataset para TREINO de uma rede neural
    # (evita que o modelo aprenda a partir de identificadores únicos). Isto é uma
    # etapa de preparação para treinamento, não estritamente parte da normalização
    # matemática das features. Mantemos a remoção por padrão aqui, mas pode ser
    # ajustada caso queira preservar IDs para análises fora do treinamento.
    drop_cols = [c for c in ['Name', 'Ticket', 'PassengerId', 'Title'] if c in df.columns]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True, errors='ignore')

    return df


def main():
    parser = argparse.ArgumentParser(description='Unified normalization pipeline for Titanic-like CSVs')
    parser.add_argument('--input', '-i', required=True, help='Caminho para o CSV de entrada')
    parser.add_argument('--output', '-o', required=True, help='Caminho para o CSV de saída')
    parser.add_argument('--drop-cabin-threshold', type=float, default=0.5,
                        help='Se a taxa de preenchimento de Cabin for menor que esse valor, a coluna Cabin será removida (padrão: 0.5)')
    parser.add_argument('--keep-cabin-if-enough', action='store_true',
                        help='Se setado, mantém a coluna Cabin quando a taxa de preenchimento for >= threshold')
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    if not os.path.isfile(input_path):
        print(f"Erro: arquivo de entrada não encontrado: {input_path}")
        return

    try:
        df = pd.read_csv(input_path)
        processed = process_pipeline(df, drop_cabin_threshold=args.drop_cabin_threshold,
                                     keep_cabin_if_enough=args.keep_cabin_if_enough)
        processed.to_csv(output_path, index=False)
        print(f"Arquivo processado salvo em: {output_path}")
    except Exception as e:
        print(f"Erro durante o processamento: {e}")


if __name__ == '__main__':
    main()
