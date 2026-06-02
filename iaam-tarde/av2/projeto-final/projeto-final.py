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

# %% [markdown]
# # Projeto Final Integrador (AV2) — Predicao da Natureza de Ocorrencias Criminais em Alagoas
#
# **Programa:** OxeTech Academy &nbsp;|&nbsp; Curso Avancado de Inteligencia Artificial e Aprendizado de Maquina
# **Instrutor:** Prof. Me. Derek Nielsen Araujo Alves
#
# ---
#
# **Problema.** Classificacao **multiclasse** da natureza da ocorrencia criminal
# (`ARMA`, `CVLI`, `DROGA`) registrada pelo NEAC em Alagoas (2022), a partir de atributos
# **espaco-temporais**. Aplicacao operacional: subsidiar a alocacao de recursos da SSP-AL
# por padrao espaco-temporal, sem perfilamento individual.
#
# **Metrica.** `macro-F1` — robusta ao desbalanceamento entre as tres classes.
#
# **Fonte.** Tres datasets NEAC unificados:
# `NEAC_ARMA_2022.csv` (apreensao de armas), `NEAC_CVLI_2022.csv` (crimes violentos letais
# intencionais, com dados de vitima) e `NEAC_DROGA_2022.csv` (apreensao de drogas).
#
# **Cobertura da rubrica AV2.** Pipeline reprodutivel sem vazamento (Sec. 3-5), comparacao
# baseline vs. modelo final (Sec. 5), interpretabilidade SHAP/importancia (Sec. 6),
# auditoria de vieses e impacto social (Sec. 7), deploy FastAPI + Docker (Sec. 8).
#
# **Como roda.** O notebook usa um coletor de validacoes **nao-interruptivo**: cada secao
# termina com chamadas `verificar(...)` que registram `[OK]`/`[X]` sem abortar a execucao.
# A ultima celula emite o relatorio de desempenho consolidado.

# %% [markdown]
# ## Secao 0 — Preparacao do Ambiente
#
# Imports globais, semente de reprodutibilidade e o coletor de validacoes.

# %%
import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)

try:
    import tensorflow as tf
    from tensorflow import keras
    _TF_OK = True
except Exception as _e:
    _TF_OK = False
    print("TensorFlow indisponivel — a MLP da Secao 5 sera ignorada:", _e)

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
if _TF_OK:
    tf.random.set_seed(SEED)

# Coletor de resultados: registra cada verificacao sem interromper o notebook.
_resultados = []


def verificar(secao, descricao, teste):
    """Avalia 'teste' (funcao sem argumentos) de forma segura: excecoes ou valores
    invalidos sao registrados como falha, nunca propagados."""
    try:
        ok = bool(teste())
    except Exception:
        ok = False
    _resultados.append({"secao": secao, "descricao": descricao, "ok": ok})
    print(f"[{'OK ' if ok else ' X '}] Secao {secao} — {descricao}")
    return ok


print("Ambiente pronto. Coletor de validacoes inicializado.")
verificar("0", "numpy/pandas/sklearn importados",
          lambda: all(m is not None for m in (np.__version__, pd.__version__, sklearn.__version__)))
verificar("0", "semente global definida", lambda: SEED == 42)

# %% [markdown]
# ## Secao 1 — Carga e Unificacao dos Dados
#
# Carrega os tres CSVs, padroniza tokens de ausencia (`NI`, `SEM INFORMACAO`, ...),
# concatena mantendo as colunas comuns, adiciona o rotulo `tipo_ocorrencia` e converte
# coordenadas (virgula decimal) e a data do fato. As colunas de vitima existem apenas no
# CVLI e ficam `NaN` nas demais classes — sao reservadas para a auditoria de equidade da Secao 7.

# %%
def _em_colab():
    """True se o notebook roda no Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def _tem_csv(caminho):
    """True se `caminho` e um diretorio que contem ao menos um arquivo .csv."""
    try:
        return os.path.isdir(caminho) and any(f.endswith(".csv")
                                              for f in os.listdir(caminho))
    except Exception:
        return False


def resolver_dados_dir():
    """Resolve o diretorio dos CSVs em qualquer ambiente.

    Ordem de busca:
      1. `./dados` local (e variacoes) ou o caminho em `DADOS_DRIVE`, se definido —
         verificacao SEM efeito colateral (nao monta Drive se os dados ja estao locais);
      2. so se nao achar e estiver no Colab: monta o Google Drive e tenta os caminhos
         padrao no MyDrive.
    Retorna o primeiro diretorio que contem arquivos `.csv`.
    """
    extra = globals().get("DADOS_DRIVE")
    locais = ([extra] if extra else []) + \
             ["dados", "./dados", os.path.join(os.getcwd(), "dados")]

    # 1) verifica local / caminho informado primeiro (sem montar o Drive).
    for c in locais:
        if _tem_csv(c):
            print("Diretorio de dados (local):", c)
            return c

    # 2) Colab: monta o Drive apenas se os dados nao foram encontrados localmente.
    if _em_colab():
        try:
            from google.colab import drive
            drive.mount("/content/drive", force_remount=False)
        except Exception as e:
            print("Falha ao montar o Google Drive:", e)
        for c in ["/content/drive/MyDrive/oxetech/projeto-final/dados",
                  "/content/drive/MyDrive/projeto-final/dados",
                  "/content/dados"]:
            if _tem_csv(c):
                print("Diretorio de dados (Google Drive):", c)
                return c

    print("AVISO: CSVs nao encontrados em ./dados nem no Drive; usando 'dados'.")
    return "dados"


DADOS_DIR = resolver_dados_dir()

COLUNAS_COMUNS = [
    "QUALIFICACAO", "AMBIENTE", "DATA_HORA_FATO", "ANO_FATO", "MES_FATO",
    "MES_FATO_TEXTO", "DIA_FATO", "DIA_SEMANA_FATO", "HORA_FATO", "TURNO",
    "CIDADE_FATO", "BAIRRO_FATO", "ENDERECO_FATO", "PT_REF_FATO", "RISP",
    "AISP", "LONGITUDE", "LATITUDE",
]
COLUNAS_VITIMA = [
    "IDADE_VITIMA", "COR_RACA_VITIMA", "SEXO_VITIMA", "ESCOLARIDADE_VITIMA",
    "INSTRUMENTO_UTILIZADO", "SUBJETIVIDADE",
]
TOKENS_AUSENCIA = ["NI", "SEM INFORMACAO", "SEM INFORMAÇÃO", "NAO INFORMADO",
                   "NÃO INFORMADO", ""]


def carregar(nome):
    """Le um CSV NEAC com fallback de encoding e normaliza tokens de ausencia."""
    caminho = os.path.join(DADOS_DIR, nome)
    try:
        dados = pd.read_csv(caminho, sep=",", encoding="utf-8")
    except Exception:
        dados = pd.read_csv(caminho, sep=",", encoding="latin-1")
    return dados.replace(TOKENS_AUSENCIA, np.nan)


def coords_para_float(serie):
    """Converte '-36,669077' (string com virgula decimal) em float."""
    return pd.to_numeric(serie.astype(str).str.replace(",", ".", regex=False),
                         errors="coerce")


df_arma = carregar("NEAC_ARMA_2022.csv")
df_cvli = carregar("NEAC_CVLI_2022.csv")
df_droga = carregar("NEAC_DROGA_2022.csv")

# Mantem as colunas comuns em todas as fontes; o CVLI carrega tambem as colunas de vitima.
df_arma = df_arma.reindex(columns=COLUNAS_COMUNS)
df_droga = df_droga.reindex(columns=COLUNAS_COMUNS)
df_cvli = df_cvli.reindex(columns=COLUNAS_COMUNS + COLUNAS_VITIMA)

df_arma["tipo_ocorrencia"] = "ARMA"
df_cvli["tipo_ocorrencia"] = "CVLI"
df_droga["tipo_ocorrencia"] = "DROGA"

df = pd.concat([df_arma, df_cvli, df_droga], ignore_index=True, sort=False)

# Coordenadas e data do fato.
df["LONGITUDE"] = coords_para_float(df["LONGITUDE"])
df["LATITUDE"] = coords_para_float(df["LATITUDE"])
df["dt"] = pd.to_datetime(df["DATA_HORA_FATO"], dayfirst=True, errors="coerce")

# Garante tipos numericos nos campos temporais usados como atributo.
for col in ["MES_FATO", "DIA_FATO", "HORA_FATO"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("DataFrame unificado:", df.shape)
print("\nDistribuicao de tipo_ocorrencia:")
print(df["tipo_ocorrencia"].value_counts())
df.head()

# %%
verificar("1", "diretorio de dados contem os 3 CSVs NEAC",
          lambda: all(os.path.exists(os.path.join(DADOS_DIR, f"NEAC_{t}_2022.csv"))
                      for t in ("ARMA", "CVLI", "DROGA")))
verificar("1", "tres fontes unificadas em df",
          lambda: set(df["tipo_ocorrencia"].unique()) == {"ARMA", "CVLI", "DROGA"})
verificar("1", "coordenadas numericas",
          lambda: pd.api.types.is_numeric_dtype(df["LONGITUDE"])
                  and pd.api.types.is_numeric_dtype(df["LATITUDE"]))
verificar("1", "datetime parseado em >80% das linhas",
          lambda: df["dt"].notna().mean() > 0.8)

# %% [markdown]
# ## Secao 2 — Analise Exploratoria (EDA)
#
# Objetivo: entender a distribuicao do alvo, padroes temporais e geograficos, e a
# qualidade dos dados (ausentes) — informacoes que orientam a modelagem e a discussao de vieses.

# %%
print("Shape:", df.shape)
ausentes = (df.isna().mean() * 100).round(2).sort_values(ascending=False)
print("\n% de valores ausentes por coluna (top 15):")
print(ausentes.head(15).to_string())

# %%
# Distribuicao do alvo — evidencia o desbalanceamento que motiva o macro-F1.
contagem = df["tipo_ocorrencia"].value_counts()
print(contagem)
print("\nProporcao:\n", (contagem / len(df) * 100).round(1).astype(str) + " %")
try:
    cores = {"ARMA": "#FF9900", "CVLI": "#0078D4", "DROGA": "#34A853"}
    fig, ax = plt.subplots(figsize=(8, 5))
    contagem.plot(kind="bar", ax=ax, color=[cores[c] for c in contagem.index],
                  edgecolor="black")
    ax.set_title("Distribuicao do alvo (tipo_ocorrencia)")
    ax.set_ylabel("Frequencia")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Falha no grafico do alvo:", e)

# %% [markdown]
# Classes desbalanceadas: `DROGA` e `ARMA` predominam sobre `CVLI`. Isso justifica usar
# `macro-F1` (media nao ponderada por classe) e `class_weight="balanced"` no modelo final.

# %%
# Padrao temporal por mes e por turno.
try:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    (df.groupby(["MES_FATO", "tipo_ocorrencia"]).size().unstack(fill_value=0)
        .plot(kind="line", marker="o", ax=axes[0]))
    axes[0].set_title("Ocorrencias por mes")
    axes[0].set_xlabel("Mes")
    axes[0].set_ylabel("Frequencia")
    (df.groupby(["TURNO", "tipo_ocorrencia"]).size().unstack(fill_value=0)
        .plot(kind="bar", ax=axes[1]))
    axes[1].set_title("Ocorrencias por turno")
    axes[1].set_xlabel("Turno")
    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Falha nos graficos temporais:", e)

# %%
# Top 10 cidades por volume.
try:
    fig, ax = plt.subplots(figsize=(9, 5))
    df["CIDADE_FATO"].value_counts().head(10).plot(kind="barh", ax=ax, color="#4285F4",
                                                   edgecolor="black")
    ax.set_title("Top 10 cidades por volume de ocorrencias")
    ax.set_xlabel("Frequencia")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Falha no grafico de cidades:", e)

# %%
# Dispersao geografica colorida por tipo.
try:
    geo = df[(df["LONGITUDE"].between(-39, -34)) & (df["LATITUDE"].between(-11, -8))]
    fig, ax = plt.subplots(figsize=(10, 8))
    for tipo, cor in {"ARMA": "#FF9900", "CVLI": "#0078D4", "DROGA": "#34A853"}.items():
        sub = geo[geo["tipo_ocorrencia"] == tipo]
        ax.scatter(sub["LONGITUDE"], sub["LATITUDE"], s=12, alpha=0.5, c=cor, label=tipo)
    ax.set_title("Dispersao geografica das ocorrencias (Alagoas)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(title="tipo_ocorrencia")
    plt.tight_layout()
    plt.show()
    print(f"Coordenadas plotadas: {len(geo)} de {len(df)} registros.")
except Exception as e:
    print("Falha no grafico geografico:", e)

# %% [markdown]
# **Implicacoes.** Ha padrao temporal (mes/turno) e forte concentracao geografica em poucas
# cidades — sinal de que atributos espaco-temporais serao preditivos, mas tambem alerta de
# **vies geografico** (a geolocalizacao e proxy de raca/renda; ver Secao 7).

# %%
verificar("2", "EDA executada sobre df nao vazio", lambda: len(df) > 0)
verificar("2", "alvo com tres classes presentes", lambda: df["tipo_ocorrencia"].nunique() == 3)

# %% [markdown]
# ## Secao 3 — Pre-processamento sem Vazamento
#
# A estrategia anti-vazamento e central na rubrica. Todo ajuste (imputacao, escala,
# codificacao) e encapsulado num `ColumnTransformer` que so e **fitado dentro de um
# `Pipeline`**, junto do estimador — portanto, **apenas sobre o conjunto de treino**.
#
# **Selecao de atributos.** Usamos somente variaveis espaco-temporais. Excluimos de proposito
# `QUALIFICACAO`, `INSTRUMENTO_UTILIZADO`, `SUBJETIVIDADE` e demais campos que revelariam a
# classe (vazamento), alem dos atributos sensiveis de vitima (raca/sexo), reservados a auditoria.

# %%
num_cols = ["MES_FATO", "DIA_FATO", "HORA_FATO", "LONGITUDE", "LATITUDE"]
cat_cols = ["DIA_SEMANA_FATO", "TURNO", "CIDADE_FATO", "RISP", "AISP", "AMBIENTE"]
target = "tipo_ocorrencia"

X = df[num_cols + cat_cols].copy()
y = df[target].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
    ]
)


def avaliar(nome, modelo, X_te, y_te):
    """Avalia um modelo no teste: imprime relatorio e devolve acuracia e macro-F1."""
    pred = modelo.predict(X_te)
    acc = accuracy_score(y_te, pred)
    f1m = f1_score(y_te, pred, average="macro")
    print(f"\n=== {nome} ===")
    print(f"Acuracia: {acc:.4f} | macro-F1: {f1m:.4f}")
    print(classification_report(y_te, pred, zero_division=0))
    return {"acuracia": acc, "f1_macro": f1m}


print("Treino:", X_train.shape, "| Teste:", X_test.shape)
print("\nProporcao de classes (treino vs teste):")
print(pd.DataFrame({
    "treino": y_train.value_counts(normalize=True).round(3),
    "teste": y_test.value_counts(normalize=True).round(3),
}))

# %%
verificar("3", "split estratificado sem sobreposicao de indices",
          lambda: set(X_train.index).isdisjoint(set(X_test.index))
                  and len(X_train) + len(X_test) == len(X))
verificar("3", "proporcao de classes preservada (estratificacao)",
          lambda: (y_train.value_counts(normalize=True)
                   - y_test.value_counts(normalize=True)).abs().max() < 0.03)
verificar("3", "preprocess e um ColumnTransformer",
          lambda: hasattr(preprocess, "transformers"))

# %% [markdown]
# ## Secao 4 — Modelo Baseline
#
# O baseline estabelece o piso de desempenho. Comparamos um classificador trivial
# (`most_frequent`) com uma `LogisticRegression`. O ganho do modelo final (Secao 5) so e
# legitimo se superar esse piso em `macro-F1`.

# %%
clf_dummy = Pipeline([("prep", preprocess),
                      ("clf", DummyClassifier(strategy="most_frequent"))])
clf_logreg = Pipeline([("prep", preprocess),
                       ("clf", LogisticRegression(max_iter=1000, random_state=SEED))])

clf_dummy.fit(X_train, y_train)
clf_logreg.fit(X_train, y_train)

f1_dummy = avaliar("Baseline — DummyClassifier", clf_dummy, X_test, y_test)["f1_macro"]
f1_lr = avaliar("Baseline — LogisticRegression", clf_logreg, X_test, y_test)["f1_macro"]
f1_base = max(f1_dummy, f1_lr)
print(f"\nMelhor baseline (macro-F1): {f1_base:.4f}")

# %%
verificar("4", "baseline dummy treinado",
          lambda: hasattr(clf_dummy.named_steps["clf"], "classes_"))
verificar("4", "logistic regression supera o dummy em macro-F1",
          lambda: f1_lr > f1_dummy)
verificar("4", "f1_base definido em [0,1]",
          lambda: 0.0 <= float(f1_base) <= 1.0)

# %% [markdown]
# ## Secao 5 — Modelo Final e Comparacao
#
# Modelo final principal: **RandomForest** (`class_weight="balanced"` para o desbalanceamento).
# Como a disciplina trata de redes neurais, treinamos tambem uma **MLP (Keras)** como modelo
# neural comparativo. Reunimos baseline, RF e MLP numa tabela de `macro-F1`.

# %%
modelo_rf = Pipeline([
    ("prep", preprocess),
    ("rf", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                  random_state=SEED, n_jobs=-1)),
])
modelo_rf.fit(X_train, y_train)
modelo_final = modelo_rf  # artefato escolhido para o deploy (Secao 8)

res_rf = avaliar("Modelo final — RandomForest", modelo_rf, X_test, y_test)
f1_rf = res_rf["f1_macro"]
acc_rf = res_rf["acuracia"]

# %%
# MLP Keras sobre a matriz pre-processada (fit do pre-processador apenas no treino).
modelo_mlp = None
f1_mlp = np.nan
acc_mlp = np.nan
if _TF_OK:
    try:
        prep_mlp = clone(preprocess)
        Xtr = prep_mlp.fit_transform(X_train)
        Xte = prep_mlp.transform(X_test)
        if hasattr(Xtr, "toarray"):
            Xtr = Xtr.toarray()
            Xte = Xte.toarray()
        le = LabelEncoder().fit(y_train)
        ytr = le.transform(y_train)
        yte = le.transform(y_test)

        modelo_mlp = keras.Sequential([
            keras.layers.Input(shape=(Xtr.shape[1],)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(len(le.classes_), activation="softmax"),
        ])
        modelo_mlp.compile(optimizer="adam",
                           loss="sparse_categorical_crossentropy",
                           metrics=["accuracy"])
        modelo_mlp.fit(Xtr, ytr, validation_split=0.2, epochs=30, batch_size=32, verbose=0)

        pred_mlp = modelo_mlp.predict(Xte, verbose=0).argmax(axis=1)
        f1_mlp = f1_score(yte, pred_mlp, average="macro")
        acc_mlp = accuracy_score(yte, pred_mlp)
        print(f"MLP — Acuracia: {acc_mlp:.4f} | macro-F1: {f1_mlp:.4f}")
    except Exception as e:
        print("Falha ao treinar a MLP:", e)
else:
    print("TensorFlow indisponivel — MLP ignorada.")

# %% [markdown]
# ### Arvore de Decisao (modelo interpretavel)
#
# Alem do RandomForest (ensemble) e da MLP (rede neural), treinamos uma **arvore de decisao**
# unica e rasa (`max_depth=4`). Sozinha ela e menos precisa, mas **totalmente interpretavel**:
# cada predicao e uma sequencia de regras espaco-temporais legiveis — base para a discussao de
# interpretabilidade (Secao 6) e de vieses (Secao 7).

# %%
modelo_dt = Pipeline([
    ("prep", preprocess),
    ("dt", DecisionTreeClassifier(max_depth=4, class_weight="balanced",
                                  random_state=SEED)),
])
modelo_dt.fit(X_train, y_train)
res_dt = avaliar("Arvore de Decisao (max_depth=4)", modelo_dt, X_test, y_test)
f1_dt = res_dt["f1_macro"]
acc_dt = res_dt["acuracia"]

# %%
# Visualizacao da arvore: as regras de decisao mais altas (3 niveis) em forma legivel.
try:
    nomes_dt = modelo_dt.named_steps["prep"].get_feature_names_out()
    fig, ax = plt.subplots(figsize=(22, 11))
    plot_tree(modelo_dt.named_steps["dt"], feature_names=list(nomes_dt),
              class_names=list(modelo_dt.named_steps["dt"].classes_),
              filled=True, rounded=True, fontsize=8, max_depth=3, ax=ax)
    ax.set_title("Arvore de Decisao — regras espaco-temporais (3 niveis superiores)")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Falha ao plotar a arvore de decisao:", e)

# %%
tabela_comp = pd.DataFrame([
    {"Modelo": "Baseline (melhor)", "Acuracia": np.nan, "F1_macro": f1_base},
    {"Modelo": "Arvore de Decisao", "Acuracia": acc_dt, "F1_macro": f1_dt},
    {"Modelo": "RandomForest", "Acuracia": acc_rf, "F1_macro": f1_rf},
    {"Modelo": "MLP (Keras)", "Acuracia": acc_mlp, "F1_macro": f1_mlp},
])
print(tabela_comp.to_string(index=False))

try:
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(modelo_rf, X_test, y_test, ax=ax,
                                          cmap="Blues", colorbar=False)
    ax.set_title("Matriz de confusao — RandomForest")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Falha na matriz de confusao:", e)

# %%
verificar("5", "modelo final (RF) treinado",
          lambda: hasattr(modelo_rf.named_steps["rf"], "classes_"))
verificar("5", "MLP Keras treinada com validacao",
          lambda: modelo_mlp is not None and "val_loss" in modelo_mlp.history.history)
verificar("5", "modelo final supera o baseline em macro-F1",
          lambda: float(f1_rf) > float(f1_base))
verificar("5", "arvore de decisao treinada",
          lambda: hasattr(modelo_dt.named_steps["dt"], "tree_"))
verificar("5", "tabela comparativa com 4 modelos",
          lambda: tabela_comp.shape[0] >= 4 and "F1_macro" in tabela_comp.columns)

# %% [markdown]
# ## Secao 6 — Interpretabilidade (SHAP / Importancia)
#
# A rubrica de etica (20%) exige interpretabilidade. Extraimos a importancia das variaveis do
# RandomForest e, quando `shap` estiver disponivel, o `TreeExplainer` sobre uma amostra do teste.

# %%
nomes = modelo_rf.named_steps["prep"].get_feature_names_out()
importancias = modelo_rf.named_steps["rf"].feature_importances_

imp = (pd.Series(importancias, index=nomes).sort_values(ascending=False))
print("Top 15 atributos por importancia (RandomForest):")
print(imp.head(15).to_string())

try:
    fig, ax = plt.subplots(figsize=(9, 6))
    imp.head(15).iloc[::-1].plot(kind="barh", ax=ax, color="#0078D4", edgecolor="black")
    ax.set_title("Top 15 importancias — RandomForest")
    ax.set_xlabel("Importancia")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Falha no grafico de importancias:", e)

# %%
# SHAP (opcional — nao aborta se ausente).
try:
    import shap
    Xte_t = modelo_rf.named_steps["prep"].transform(X_test)
    if hasattr(Xte_t, "toarray"):
        Xte_t = Xte_t.toarray()
    amostra = Xte_t[:200]
    explainer = shap.TreeExplainer(modelo_rf.named_steps["rf"])
    shap_values = explainer.shap_values(amostra)
    shap.summary_plot(shap_values, amostra, feature_names=list(nomes),
                      plot_type="bar", show=True)
except Exception as e:
    print("SHAP indisponivel ou falhou (usando importancia do RF como fallback):", e)

# %% [markdown]
# **Leitura.** As variaveis mais influentes tendem a ser **geograficas** (LONGITUDE/LATITUDE,
# CIDADE) e **temporais** (HORA/TURNO). Isso confirma que o sinal preditivo vem do *onde* e
# *quando* — exatamente o que torna o modelo util para alocacao de recursos, mas tambem o que
# exige a auditoria de viés geografico da proxima secao.

# %%
verificar("6", "importancias calculadas",
          lambda: len(importancias) > 0)

# %% [markdown]
# ## Secao 7 — Auditoria de Equidade e Vieses
#
# A acuracia global oculta disparidades entre subgrupos. Auditamos o desempenho por `TURNO` e
# por cidade, e analisamos o perfil das vitimas de CVLI (impacto social).
#
# **Discriminacao indireta.** As features sao espaco-temporais, mas `CIDADE`/geolocalizacao
# funcionam como **proxy de raca e renda** em Alagoas. Mesmo sem usar raca explicitamente, o
# modelo pode reproduzir padroes historicos de policiamento (feedback loop). Por isso os
# atributos sensiveis de vitima (raca, sexo) foram **deliberadamente excluidos do treino** —
# eles servem apenas para medir impacto, nunca para prever.

# %%
# Desempenho por subgrupo (TURNO e Top-5 cidades).
aud = pd.DataFrame()
disparidade = 0.0
try:
    y_pred = modelo_rf.predict(X_test)
    y_true = y_test if isinstance(y_test, pd.Series) else pd.Series(y_test, index=X_test.index)
    turno_te = df.loc[X_test.index, "TURNO"].fillna("Desconhecido")
    cidade_te = df.loc[X_test.index, "CIDADE_FATO"].fillna("Desconhecida")
    pred_s = pd.Series(y_pred, index=X_test.index)

    linhas = []
    for turno in sorted(turno_te.unique()):
        m = turno_te == turno
        if m.sum() > 0:
            linhas.append({"subgrupo": f"TURNO::{turno}", "n": int(m.sum()),
                           "acuracia": accuracy_score(y_true[m], pred_s[m]),
                           "f1_macro": f1_score(y_true[m], pred_s[m], average="macro",
                                                zero_division=0)})
    for cidade in cidade_te.value_counts().head(5).index:
        m = cidade_te == cidade
        if m.sum() > 0:
            linhas.append({"subgrupo": f"CIDADE::{cidade}", "n": int(m.sum()),
                           "acuracia": accuracy_score(y_true[m], pred_s[m]),
                           "f1_macro": f1_score(y_true[m], pred_s[m], average="macro",
                                                zero_division=0)})
    aud = pd.DataFrame(linhas)
    disparidade = float(aud["f1_macro"].max() - aud["f1_macro"].min())
    print(aud.to_string(index=False))
    print(f"\nDisparidade (max-min macro-F1 entre subgrupos): {disparidade:.4f}")
except Exception as e:
    print("Falha na auditoria por subgrupo:", e)

# %%
try:
    fig, ax = plt.subplots(figsize=(9, 5))
    a = aud.sort_values("f1_macro")
    ax.barh(a["subgrupo"], a["f1_macro"], color="#0078D4", edgecolor="black")
    ax.axvline(aud["f1_macro"].mean(), color="red", linestyle="--", label="media")
    ax.set_title("macro-F1 por subgrupo")
    ax.set_xlabel("macro-F1")
    ax.legend()
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Falha no grafico de subgrupos:", e)

# %% [markdown]
# ### Impacto social — perfil das vitimas de CVLI
#
# Quem sofre a violencia letal em Alagoas? O recorte abaixo (apenas CVLI) qualifica o debate
# entre **paridade demografica** e **igualdade de oportunidades**.

# %%
try:
    cvli = df[df["tipo_ocorrencia"] == "CVLI"]
    raca = cvli["COR_RACA_VITIMA"].value_counts(dropna=True)
    sexo = cvli["SEXO_VITIMA"].value_counts(dropna=True)
    print("Vitimas de CVLI por cor/raca:\n", raca.to_string())
    print("\nVitimas de CVLI por sexo:\n", sexo.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    raca.plot(kind="bar", ax=axes[0], color="#0078D4", edgecolor="black")
    axes[0].set_title("Vitimas de CVLI por cor/raca")
    axes[0].tick_params(axis="x", rotation=30)
    sexo.plot(kind="bar", ax=axes[1], color="#FF9900", edgecolor="black")
    axes[1].set_title("Vitimas de CVLI por sexo")
    axes[1].tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Falha na analise demografica de CVLI:", e)

# %% [markdown]
# **Conclusao etica.** (1) Os dados refletem *registro policial*, nao criminalidade real —
# subnotificacao e viés de vigilancia. (2) `CIDADE` e proxy de atributos sensiveis: usar o
# modelo para **perfilamento individual** e inaceitavel. (3) Uso responsavel = **agregado**,
# para alocar recursos de prevencao, com monitoramento continuo da disparidade entre subgrupos
# (alerta se > 0,10) para evitar feedback loops.

# %%
verificar("7", "auditoria por subgrupo calculada",
          lambda: aud.shape[0] >= 2 and "f1_macro" in aud.columns)
verificar("7", "disparidade entre subgrupos quantificada",
          lambda: disparidade >= 0)

# %% [markdown]
# ## Secao 8 — Serializacao e Deploy (FastAPI + Docker)
#
# Persistimos o pipeline final com `joblib` e geramos os artefatos de deploy em `deploy/`:
# API `FastAPI` (com OpenAPI automatica), `requirements.txt`, `Dockerfile` e `README.md`.
# O pipeline ja embute o pre-processamento, entao a API recebe os campos crus.

# %%
os.makedirs("deploy", exist_ok=True)
joblib_ok = True
try:
    import joblib
    joblib.dump(modelo_final, "deploy/modelo.joblib")
    metadata = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "classes": sorted(modelo_final.classes_.tolist()),
        "metrica": "macro-F1",
    }
    with open("deploy/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print("Modelo e metadados serializados em deploy/.")
except Exception as e:
    joblib_ok = False
    print("Falha na serializacao:", e)

# %%
APP_CODE = '''"""API de inferencia — classificacao da natureza de ocorrencia criminal (ARMA/CVLI/DROGA)."""
import json
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="NEAC-AL — Classificador de Ocorrencias",
    description="Preve a natureza da ocorrencia (ARMA, CVLI, DROGA) a partir de atributos espaco-temporais.",
    version="1.0.0",
)

modelo = joblib.load("modelo.joblib")
with open("metadata.json", encoding="utf-8") as f:
    META = json.load(f)


class Ocorrencia(BaseModel):
    MES_FATO: int = Field(..., ge=1, le=12)
    DIA_FATO: int = Field(..., ge=1, le=31)
    HORA_FATO: int = Field(..., ge=0, le=23)
    LONGITUDE: float
    LATITUDE: float
    DIA_SEMANA_FATO: str
    TURNO: str
    CIDADE_FATO: str
    RISP: str
    AISP: str
    AMBIENTE: str


@app.get("/health")
def health():
    return {"status": "ok", "classes": META["classes"]}


@app.post("/prever")
def prever(o: Ocorrencia):
    linha = pd.DataFrame([o.dict()])
    classe = modelo.predict(linha)[0]
    probs = modelo.predict_proba(linha)[0]
    return {
        "classe_prevista": str(classe),
        "confianca": float(max(probs)),
        "probabilidades": {c: float(p) for c, p in zip(modelo.classes_, probs)},
    }
'''
with open("deploy/app.py", "w", encoding="utf-8") as f:
    f.write(APP_CODE)
print("deploy/app.py escrito.")

# %%
REQUIREMENTS = """fastapi>=0.110
uvicorn[standard]>=0.29
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
joblib>=1.3
"""
with open("deploy/requirements.txt", "w", encoding="utf-8") as f:
    f.write(REQUIREMENTS)

DOCKERFILE = """FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py modelo.joblib metadata.json ./
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
"""
with open("deploy/Dockerfile", "w", encoding="utf-8") as f:
    f.write(DOCKERFILE)

README = """# Deploy — Classificador de Ocorrencias NEAC-AL

API FastAPI que serve o pipeline treinado (ARMA / CVLI / DROGA).

## Build e execucao

    docker build -t neac-api:1.0 .
    docker run -p 8000:8000 neac-api:1.0

Docs OpenAPI: http://localhost:8000/docs

## Exemplo

    curl -X POST http://localhost:8000/prever -H "Content-Type: application/json" -d '{
      "MES_FATO": 6, "DIA_FATO": 15, "HORA_FATO": 20,
      "LONGITUDE": -35.73, "LATITUDE": -9.66,
      "DIA_SEMANA_FATO": "Sex", "TURNO": "Noite",
      "CIDADE_FATO": "Maceio", "RISP": "1a RISP", "AISP": "1a AISP",
      "AMBIENTE": "Via Publica"
    }'
"""
with open("deploy/README.md", "w", encoding="utf-8") as f:
    f.write(README)
print("requirements.txt, Dockerfile e README.md escritos.")

# %%
# Teste local: recarrega o artefato e preve um exemplo sintetico.
pred_exemplo = None
try:
    import joblib
    modelo_carregado = joblib.load("deploy/modelo.joblib")
    exemplo = pd.DataFrame([{
        "MES_FATO": 6, "DIA_FATO": 15, "HORA_FATO": 20,
        "LONGITUDE": -35.73, "LATITUDE": -9.66,
        "DIA_SEMANA_FATO": "Sex", "TURNO": "Noite",
        "CIDADE_FATO": "Maceio", "RISP": "1ª RISP", "AISP": "1ª AISP",
        "AMBIENTE": "Via Pública",
    }])
    classe = modelo_carregado.predict(exemplo)[0]
    probs = modelo_carregado.predict_proba(exemplo)[0]
    pred_exemplo = {"classe": str(classe),
                    "probabilidades": dict(zip(modelo_carregado.classes_, probs.round(3)))}
    print("Predicao do exemplo:", pred_exemplo)
except Exception as e:
    print("Falha ao recarregar/prever:", e)

# %%
verificar("8", "modelo serializado em disco",
          lambda: os.path.exists("deploy/modelo.joblib"))
verificar("8", "artefatos de deploy gerados (app, Dockerfile, requirements)",
          lambda: all(os.path.exists(f"deploy/{f}")
                      for f in ["app.py", "Dockerfile", "requirements.txt", "metadata.json"]))
verificar("8", "modelo recarregado preve exemplo",
          lambda: pred_exemplo is not None)

# %% [markdown]
# ## Encerramento — Relatorio de Desempenho
#
# Consolida todas as verificacoes da sessao: total de asserts aprovados, percentual e
# detalhamento por secao.

# %%
rel = pd.DataFrame(_resultados)
print("=" * 56)
print("  RELATORIO DE DESEMPENHO — PROJETO FINAL (AV2)")
print("=" * 56)
if len(rel) == 0:
    print("Nenhuma verificacao registrada. Execute as celulas em ordem.")
else:
    total = len(rel); aprovados = int(rel["ok"].sum())
    print(f"  Asserts aprovados: {aprovados} / {total}  ({aprovados/total*100:.1f}%)")
    print("-" * 56)
    por_secao = rel.groupby("secao")["ok"].agg(["sum", "count"])
    por_secao = por_secao.reindex(sorted(por_secao.index, key=lambda s: int(s)))
    for sec, linha in por_secao.iterrows():
        print(f"  Secao {sec:>2}: {int(linha['sum'])}/{int(linha['count'])} aprovados")
    pend = rel[~rel["ok"]]
    if len(pend) > 0:
        print("-" * 56); print("  Itens pendentes:")
        for _, r in pend.iterrows():
            print(f"    [X] Secao {r['secao']} — {r['descricao']}")
print("=" * 56)
