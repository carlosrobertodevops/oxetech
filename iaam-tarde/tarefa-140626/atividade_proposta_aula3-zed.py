# In[1]:
# %% Imports
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

print("Carregando o dataset Auto MPG da UCI...")
df = sns.load_dataset('mpg').drop(columns=['name'])

print(f"Dataset carregado com {df.shape[0]} linhas e {df.shape[1]} colunas.")
df.head()

# In[2]:
# %% Código

# --- SEU CÓDIGO AQUI ---
# 1. Separe as features (X) do target (y)
X = df.drop(columns=['mpg'])
y = df['mpg']

# 2. Use a função train_test_split (use test_size=0.2 e random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- FIM DO CÓDIGO ---

# In[3]:
# %% Código
# --- CHECKPOINT 1 ---
try:
    assert X_train is not None, "Você esqueceu de fazer o train_test_split!"
    assert X_train.shape[0] == 318, "O tamanho do conjunto de treino está incorreto. Verifique o test_size."
    print("✅ CHECKPOINT 1 PASSOU! Divisão feita com sucesso.")
except Exception as e:
    print(f"❌ ERRO NO CHECKPOINT 1: {e}")

# In[4]:
# %% Código

print("CONTAGEM DE NULOS (ANTES DO TRATAMENTO):")
print(X_train.isnull().sum())

# --- SEU CÓDIGO AQUI ---

# 1. Instancie o SimpleImputer
imputer = SimpleImputer(strategy='median')


# --- FIM DO CÓDIGO ---
# In[5]:
# %% Código
assert imputer is not None, "Instancie o SimpleImputer antes de continuar!"

# --- SEU CÓDIGO AQUI ---

# 2. Treine (fit) o imputador e transforme (transform) a coluna no X_train
X_train[['horsepower']] = imputer.fit_transform(X_train[['horsepower']])

# 3. Apenas aplique (transform) no X_test
X_test[['horsepower']] = imputer.transform(X_test[['horsepower']])

# --- FIM DO CÓDIGO ---
# In[6]:
# %% Código
# --- CHECKPOINT 2 ---
try:
    assert X_train['horsepower'].isnull().sum() == 0, "Ainda existem valores nulos no X_train!"
    assert X_test['horsepower'].isnull().sum() == 0, "Ainda existem valores nulos no X_test!"
    print("✅ CHECKPOINT 2 PASSOU! Problema resolvido. Veja o resultado abaixo:")
    print(X_train[['horsepower']].isnull().sum())
except Exception as e:
    print(f"❌ ERRO NO CHECKPOINT 2: {e}")

print("VALORES NA COLUNA 'ORIGIN' (TEXTO):")
print(X_train['origin'].value_counts())

# In[7]:
# %% Código

# --- SEU CÓDIGO AQUI ---

# 1. Instancie o OneHotEncoder
try:
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
except TypeError:
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

# --- FIM DO CÓDIGO ---
# In[8]:
# %% Código
assert ohe is not None, "Instancie o OneHotEncoder antes de continuar!"

# --- SEU CÓDIGO AQUI ---

# 2. Treine e transforme a coluna 'origin' do X_train
ohe_cols_train = ohe.fit_transform(X_train[['origin']])
ohe_df_train = pd.DataFrame(ohe_cols_train, columns=ohe.get_feature_names_out(), index=X_train.index)

# 3. Transforme a coluna 'origin' do X_test
ohe_cols_test = ohe.transform(X_test[['origin']])
ohe_df_test = pd.DataFrame(ohe_cols_test, columns=ohe.get_feature_names_out(), index=X_test.index)

# --- FIM DO CÓDIGO ---

# 4. Concatenando os dataframes e removendo a original
X_train = pd.concat([X_train.drop(columns=['origin']), ohe_df_train], axis=1)
X_test = pd.concat([X_test.drop(columns=['origin']), ohe_df_test], axis=1)

# --- CHECKPOINT 3 ---
try:
    assert 'origin' not in X_train.columns, "A coluna 'origin' original ainda está no dataset!"
    assert any('origin_' in col for col in X_train.columns), "As novas colunas não foram criadas."
    print("✅ CHECKPOINT 3 PASSOU! Texto transformado em binários. Veja o resultado:")
    X_train.head(3)
except Exception as e:
    print(f"❌ ERRO NO CHECKPOINT 3: {e}")

# In[9]:
# %% Código

cols_para_escalar = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']

print("ESTATÍSTICAS ANTES DO ESCALONAMENTO:")
X_train[cols_para_escalar].describe().round(2).loc[['mean', 'std', 'max']]

# --- SEU CÓDIGO AQUI ---

# 1. Instancie o StandardScaler
scaler = StandardScaler()
assert scaler is not None, "Instancie o StandardScaler antes de continuar!"

# 2. Treine (fit) e transforme (transform) as colunas do X_train
X_train[cols_para_escalar] = scaler.fit_transform(X_train[cols_para_escalar])

# 3. Apenas transforme o X_test
X_test[cols_para_escalar] = scaler.transform(X_test[cols_para_escalar])

# --- FIM DO CÓDIGO ---

# --- CHECKPOINT 4 ---
try:
    assert abs(X_train['weight'].mean()) < 0.01, "A média não está próxima de 0."
    assert abs(X_train['weight'].std() - 1.0) < 0.01, "O desvio padrão não é 1."
    print("✅ CHECKPOINT 4 PASSOU! Dados padronizados. Veja como as escalas agora são iguais:")
    X_train[cols_para_escalar].describe().round(2).loc[['mean', 'std', 'max']]
    print("\n🎉 PARABÉNS! Você concluiu a preparação de dados com sucesso!")
except Exception as e:
    print(f"❌ ERRO NO CHECKPOINT 4: {e}")
