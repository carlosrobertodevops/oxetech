# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] id="alhdBU1f6Y34"
# # 💻 Laboratório Prático: Diagnóstico de Diabetes — AV1-05
#
# **OxeTech Academy** | Aula 05 | Prof. Me. Derek Nielsen Araújo Alves
#
# ---
#
# ## 📋 Sobre esta Atividade
#
# | Item | Descrição |
# |---|---|
# | **Objetivo** | Construir e avaliar um classificador de Regressão Logística para diagnóstico de diabetes |
# | **Dataset** | Pima Indians Diabetes Database (768 pacientes, 8 atributos clínicos) |
# | **Dinâmica** | Resolva individualmente. Prazo: 7 dias após a aula. |
# | **Avaliação** | Automática via checkpoints — nota gerada no Relatório Final |
#
# ---
#
# ## ⚙️ Como Funciona
#
# | Símbolo | Significado |
# |---|---|
# | 🔍 **Diagnóstico** | Célula pronta — execute e leia a saída |
# | 👨‍💻 **Sua Vez** | Substitua `None` pelo seu código |
# | 🛑 **Checkpoint** | Validação automática — leia o feedback |
#
# ---
#
# ## 💡 Contexto do Problema
#
# O dataset *Pima Indians Diabetes* contém dados de pacientes do programa de saúde do povo Pima (Arizona/EUA). O objetivo é prever se um paciente **desenvolverá diabetes** com base em medições clínicas.
#
# **Atenção ética:** este dataset foi historicamente criticado por sub-representação de populações rurais. Um modelo treinado nele pode ter desempenho diferente em populações como comunidades quilombolas de Alagoas — cujos dados raramente aparecem em datasets públicos. Tenha isso em mente ao interpretar seus resultados.
#
# | Atributo | Descrição |
# |----------|-----------|
# | `pregnancies` | Número de gestações |
# | `glucose` | Concentração de glicose (mg/dL) |
# | `blood_pressure` | Pressão arterial diastólica (mmHg) |
# | `skin_thickness` | Espessura da dobra cutânea (mm) |
# | `insulin` | Insulina sérica (μU/mL) |
# | `bmi` | Índice de Massa Corporal (kg/m²) |
# | `diabetes_pedigree` | Histórico familiar de diabetes |
# | `age` | Idade (anos) |
# | **`outcome`** | **1 = diabetes, 0 = sem diabetes (TARGET)** |
#
# ---
#

# %% colab={"base_uri": "https://localhost:8080/"} id="e-bn5IfV6Y34" outputId="cb555ad2-3b9e-4c4c-cbb1-0e7892cd55de"
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

print('✅ Imports OK — ambiente pronto.')


# %% colab={"base_uri": "https://localhost:8080/", "height": 259} id="_QEqCkDf6Y35" outputId="517517f7-8cfd-4d32-dba9-14a3296128d7"
# ─── Carregamento inline do dataset Pima Indians Diabetes ───────────────────
# O dataset está disponível publicamente no UCI ML Repository.
# Para garantir que funcione offline, vamos carregá-lo via URL raw do GitHub.

URL = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
COLUNAS = ['pregnancies','glucose','blood_pressure','skin_thickness',
           'insulin','bmi','diabetes_pedigree','age','outcome']

df = pd.read_csv(URL, header=None, names=COLUNAS)
print('✅ Dataset carregado via URL.')
print(f'Shape: {df.shape}')
print(f'Colunas: {df.columns.tolist()}')
df.head()


# %% [markdown] id="408sLdaw6Y35"
# ---
# ## 🔍 Diagnóstico 1 — Explorando o Dataset
#
# Execute a célula abaixo e leia com atenção os resultados. **Você vai precisar destas informações para o Checkpoint 1.**
#

# %% colab={"base_uri": "https://localhost:8080/"} id="cDRRxZDY6Y36" outputId="e1130427-36a6-4ba5-c347-9b1d60c11bff"
# ─── NÃO MODIFIQUE — apenas execute ────────────────────────────────────────
print('═' * 60)
print('DIAGNÓSTICO INICIAL DO DATASET')
print('═' * 60)

print(f'\n1. Shape: {df.shape}')
print(f'   Linhas (pacientes): {df.shape[0]}')
print(f'   Colunas (atributos + target): {df.shape[1]}')

print(f'\n2. Valores ausentes por coluna:')
print(df.isnull().sum().to_string())

print(f'\n3. Distribuição da variável-alvo (outcome):')
vc = df['outcome'].value_counts().sort_index()
for cls, cnt in vc.items():
    label = 'diabetes' if cls == 1 else 'sem diabetes'
    print(f'   Classe {cls} ({label:12s}): {cnt:3d} ({cnt/len(df)*100:.1f}%)')

print(f'\n4. Estatísticas descritivas:')
print(df.describe().round(2).to_string())
print('═' * 60)


# %% [markdown] id="VQe5fmkr6Y36"
# ---
# ## 👨‍💻 Checkpoint 1 — Preparar Features e Target
#
# **Tarefa:** separar o dataset em:
# - `X` → todas as colunas **exceto** `outcome`
# - `y` → apenas a coluna `outcome`
#
# Em seguida, verifique:
# - Quantas features tem `X`?
# - Quantas amostras tem `y`?
# - O dataset está desbalanceado?
#

# %% colab={"base_uri": "https://localhost:8080/"} id="9KICxwNY6Y36" outputId="0cfeb598-1daf-401f-f5b5-dcb4128eb574"
# --- SEU CÓDIGO AQUI ---
X = df.drop(columns='outcome')
y = df['outcome']
# --- FIM DO CÓDIGO ---

# --- CHECKPOINT 1 ---
try:
    assert X is not None, "X não foi definido — substitua None pelo código correto."
    assert y is not None, "y não foi definido — substitua None pelo código correto."
    assert isinstance(X, pd.DataFrame), "X deve ser um DataFrame."
    assert isinstance(y, pd.Series), "y deve ser uma Series."
    assert X.shape[1] == 8, f"X deve ter 8 features, mas tem {X.shape[1]}."
    assert 'outcome' not in X.columns, "A coluna 'outcome' não deve estar em X."
    assert set(y.unique()).issubset({0, 1}), "y deve conter apenas 0 e 1."
    assert len(X) == len(y) == 768, f"Esperava 768 amostras, encontrou {len(X)}."

    print('✅ CHECKPOINT 1 PASSOU!')
    print(f'   X: {X.shape[0]} pacientes × {X.shape[1]} features')
    print(f'   y: {len(y)} valores | classes: {sorted(y.unique())}')
    taxa_pos = y.mean() * 100
    print(f'   Taxa de diabetes (classe 1): {taxa_pos:.1f}%')
    print(f'\n💡 Com {taxa_pos:.0f}% de positivos, o dataset está moderadamente desbalanceado.')
    print('   Por isso, nas próximas etapas usaremos stratify= no split e F1 como métrica.')
except Exception as e:
    print(f'❌ ERRO NO CHECKPOINT 1: {e}')


# %% [markdown] id="kOtWePt36Y36"
# ---
# ## 👨‍💻 Checkpoint 2 — Split Estratificado Treino/Teste
#
# **Tarefa:** usar `train_test_split` para criar:
# - `X_train, X_test, y_train, y_test`
# - Defina a porcentagem para teste
# - `random_state=42` para reprodutibilidade
# - Usar stratify?
#
#

# %% id="4rhWkRZh6Y36"
# --- SEU CÓDIGO AQUI ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
# --- FIM DO CÓDIGO ---

# --- CHECKPOINT 2 ---
try:
    assert X_train is not None, "X_train não foi definido."
    assert X_test is not None, "X_test não foi definido."
    assert y_train is not None, "y_train não foi definido."
    assert y_test is not None, "y_test não foi definido."
    assert len(X_train) + len(X_test) == len(X), "A soma treino+teste deve ser igual ao total."
    assert abs(len(X_test) / len(X) - 0.2) < 0.02, "Verifique test_size=0.2."
    # Verificar estratificação: proporção de positivos deve ser similar nos dois splits
    taxa_train = y_train.mean()
    taxa_test  = y_test.mean()
    assert abs(taxa_train - taxa_test) < 0.05, (
        f"Taxa de positivos muito diferente entre treino ({taxa_train:.2f}) e teste ({taxa_test:.2f}). "
        "Verifique stratify=y.")
    print('✅ CHECKPOINT 2 PASSOU!')
    print(f'   Treino: {len(X_train)} amostras | Taxa diabetes: {taxa_train:.1%}')
    print(f'   Teste:  {len(X_test)} amostras  | Taxa diabetes: {taxa_test:.1%}')
    print(f'\n💡 A proporção de diabetes ficou semelhante nos dois splits — isso é estratificação funcionando.')
except Exception as e:
    print(f'❌ ERRO NO CHECKPOINT 2: {e}')


# %% [markdown] id="s9E8zW4o6Y37"
# ---
# ## 👨‍💻 Checkpoint 3 — Pipeline com StandardScaler + LogisticRegression
#
# **Tarefa:** criar um `Pipeline` chamado `pipe` com dois passos:
# 1. `('scaler', StandardScaler())` — padronizar as features
# 2. `('logreg', LogisticRegression(max_iter=1000, random_state=42))` — classificador
#
# Depois, **treinar o pipeline** com `X_train` e `y_train`.
#
# ⚠️ *Por que Pipeline e não fazer separado?* Com Pipeline, o `StandardScaler` aprende a média e desvio **somente do treino**. Aplicar separado antes do split cause **data leakage** — como vimos na Aula 04.
#

# %% id="gaw2XRYV6Y37"
# --- SEU CÓDIGO AQUI ---
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=1000, random_state=42)),
])
pipe.fit(X_train, y_train)
# --- FIM DO CÓDIGO ---

# --- CHECKPOINT 3 ---
try:
    assert pipe is not None, "O pipeline não foi criado — substitua None."
    assert hasattr(pipe, 'predict'), "pipe deve ser um Pipeline treinado (com .predict)."
    assert 'scaler' in pipe.named_steps, "O pipeline deve ter um passo chamado 'scaler'."
    assert 'logreg' in pipe.named_steps, "O pipeline deve ter um passo chamado 'logreg'."
    assert hasattr(pipe.named_steps['logreg'], 'coef_'), (
        "O modelo ainda não foi treinado. Certifique-se de chamar pipe.fit(X_train, y_train).")
    y_pred_test = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred_test)
    assert acc > 0.70, f"Acurácia muito baixa ({acc:.2%}). Verifique o pipeline."
    print('✅ CHECKPOINT 3 PASSOU!')
    print(f'   Pipeline treinado com {len(X_train)} amostras')
    print(f'   Passos: {list(pipe.named_steps.keys())}')
    print(f'   Acurácia no teste: {acc:.2%}')
    print(f'\n💡 O pipeline garante que o scaler nunca "vê" os dados de teste — zero data leakage.')
except Exception as e:
    print(f'❌ ERRO NO CHECKPOINT 3: {e}')


# %% [markdown] id="ITE2FN2R6Y38"
# ---
# ## 👨‍💻 Checkpoint 4 — Validação Cruzada com F1
#
# **Tarefa:** usar `cross_val_score` para avaliar o pipeline com **validação cruzada estratificada de 5 folds**.
#
# - Use `scoring='f1'` (não accuracy — o dataset é desbalanceado!)
# - Use `cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
# - Salve os resultados em `scores_cv` (array de 5 valores)
# - Salve a média em `media_f1` e o desvio padrão em `std_f1`
#

# %% id="zt8ZKCh26Y38"
# --- SEU CÓDIGO AQUI ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_cv = cross_val_score(pipe, X, y, cv=cv, scoring='f1')
media_f1  = scores_cv.mean()
std_f1    = scores_cv.std()
# --- FIM DO CÓDIGO ---

# --- CHECKPOINT 4 ---
try:
    assert scores_cv is not None, "scores_cv não foi calculado."
    assert media_f1 is not None, "media_f1 não foi calculado."
    assert std_f1 is not None, "std_f1 não foi calculado."
    assert len(scores_cv) == 5, f"Esperava 5 scores (um por fold), encontrou {len(scores_cv)}."
    assert 0.0 < media_f1 < 1.0, "media_f1 deve estar entre 0 e 1."
    assert media_f1 > 0.55, f"F1 médio muito baixo ({media_f1:.3f}). Verifique scoring='f1'."
    assert abs(media_f1 - np.mean(scores_cv)) < 0.001, "media_f1 deve ser a média de scores_cv."
    print('✅ CHECKPOINT 4 PASSOU!')
    print(f'\n   Scores por fold: {[f"{s:.3f}" for s in scores_cv]}')
    print(f'   F1 médio:  {media_f1:.4f}')
    print(f'   Desvio padrão: {std_f1:.4f}')
    print(f'   Intervalo: [{media_f1-std_f1:.3f}, {media_f1+std_f1:.3f}]')
    print(f'\n💡 Um desvio padrão baixo indica que o modelo é estável entre os folds.')
    print(f'   Um F1 médio alto indica bom equilíbrio entre precision e recall.')
except Exception as e:
    print(f'❌ ERRO NO CHECKPOINT 4: {e}')


# %% [markdown] id="u0eDd6I96Y38"
# ---
# ## 👨‍💻 Checkpoint 5 — Interpretação dos Coeficientes
#
# **Tarefa:** extrair e analisar os coeficientes da Regressão Logística treinada no pipeline.
#
# 1. Extraia os coeficientes com `pipe.named_steps['logreg'].coef_[0]`
# 2. Crie um DataFrame `df_coef` com colunas `atributo` e `coeficiente`
# 3. Ordene por **valor absoluto** decrescente
# 4. Salve em `top3` os 3 atributos com maior magnitude absoluta
#
# 💡 *Por que só podemos interpretar a magnitude após o StandardScaler?* Porque os coeficientes agora estão na mesma escala — podemos comparar glicose com IMC diretamente.
#

# %% id="uiGos43C6Y38"
# --- SEU CÓDIGO AQUI ---
coefs = pipe.named_steps['logreg'].coef_[0]
df_coef = pd.DataFrame({'atributo': X.columns, 'coeficiente': coefs})
df_coef = df_coef.reindex(
    df_coef['coeficiente'].abs().sort_values(ascending=False).index
).reset_index(drop=True)
top3 = df_coef['atributo'].head(3).tolist()
# --- FIM DO CÓDIGO ---

# --- CHECKPOINT 5 ---
try:
    assert df_coef is not None, "df_coef não foi criado."
    assert top3 is not None, "top3 não foi criado."
    assert isinstance(df_coef, pd.DataFrame), "df_coef deve ser um DataFrame."
    assert 'atributo' in df_coef.columns, "df_coef deve ter coluna 'atributo'."
    assert 'coeficiente' in df_coef.columns, "df_coef deve ter coluna 'coeficiente'."
    assert len(df_coef) == 8, f"df_coef deve ter 8 linhas (uma por feature), encontrou {len(df_coef)}."
    assert len(top3) == 3, f"top3 deve ter exatamente 3 itens, encontrou {len(top3)}."
    assert 'glucose' in list(top3), "'glucose' deve estar entre os 3 mais influentes (é a feature mais preditiva do diabetes)."
    print('✅ CHECKPOINT 5 PASSOU!')
    print('\n   Coeficientes ordenados por magnitude:')
    print('   ' + '-'*50)
    for _, row in df_coef.head(8).iterrows():
        direcao = '↑ aumenta P(diabetes)' if row['coeficiente'] > 0 else '↓ diminui P(diabetes)'
        print(f"   {row['atributo']:20s} | β={row['coeficiente']:+.3f} | {direcao}")
    print('   ' + '-'*50)
    print(f'\n   Top 3 mais influentes: {list(top3)}')
    print(f'\n💡 REFLEXÃO ÉTICA: o modelo usa "pregnancies" como preditor de diabetes.')
    print('   Isso faz sentido clínico? Ou é uma proxy de outros fatores socioeconômicos?')
    print('   Em populações rurais alagoanas, esse coeficiente poderia ser diferente?')
except Exception as e:
    print(f'❌ ERRO NO CHECKPOINT 5: {e}')


# %% [markdown] id="jYgrm2oM6Y38"
# ---
# ## 📊 Relatório Final — AV1-05
#
# Execute a célula abaixo para gerar sua nota automaticamente. (Não edite a célula)
#
#

# %% id="jLJ94h4J6Y38"
# ═══════════════════════════════════════════════════════════
# RELATÓRIO FINAL — AV1-05
# ═══════════════════════════════════════════════════════════

checkpoints_total = 5

checks = {
    'Checkpoint 1 — Features e Target': [
        X is not None and isinstance(X, pd.DataFrame) and X.shape[1] == 8,
        y is not None and isinstance(y, pd.Series) and set(y.unique()).issubset({0,1})
    ],
    'Checkpoint 2 — Split Estratificado': [
        X_train is not None and X_test is not None,
        abs(len(X_test)/len(X) - 0.2) < 0.02,
        abs(y_train.mean() - y_test.mean()) < 0.05
    ],
    'Checkpoint 3 — Pipeline Treinado': [
        pipe is not None and hasattr(pipe, 'predict'),
        'scaler' in pipe.named_steps and 'logreg' in pipe.named_steps,
        accuracy_score(y_test, pipe.predict(X_test)) > 0.70
    ],
    'Checkpoint 4 — Cross-Validation F1': [
        scores_cv is not None and len(scores_cv) == 5,
        media_f1 is not None and 0 < media_f1 < 1 and media_f1 > 0.55
    ],
    'Checkpoint 5 — Coeficientes': [
        df_coef is not None and isinstance(df_coef, pd.DataFrame) and len(df_coef) == 8,
        top3 is not None and len(top3) == 3 and 'glucose' in list(top3)
    ],
}

resultados = {}
checkpoints_passed = 0
for nome, condicoes in checks.items():
    try:
        passou = all(condicoes)
    except Exception:
        passou = False
    resultados[nome] = passou
    if passou:
        checkpoints_passed += 1

nota = round((checkpoints_passed / checkpoints_total) * 10, 1)

print('=' * 56)
print(f'         RELATÓRIO FINAL — AV1-05')
print('=' * 56)
for nome, passou in resultados.items():
    print(f"  {'✅' if passou else '❌'}  {nome}")
print('-' * 56)
print(f'  Checkpoints aprovados: {checkpoints_passed} / {checkpoints_total}')
print(f'  Nota final:            {nota} / 10.0')
print('=' * 56)
if nota == 10.0:
    print('\n🏆 Parabéns! Todos os checkpoints passaram.')
elif nota >= 7.0:
    print(f'\n✅ Aprovado! Revise os checkpoints que não passaram.')
else:
    print(f'\n⚠️  Abaixo da média. Revise os conceitos e tente novamente.')
    print('   Dica: releia as mensagens de erro de cada checkpoint com atenção.')
