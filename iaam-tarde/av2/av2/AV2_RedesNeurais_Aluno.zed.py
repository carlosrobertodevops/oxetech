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
# # Notebook Avaliativo — Redes Neurais Artificiais
# ### Do Perceptron a Rede Multicamada (MLP) com TensorFlow/Keras — Modulo 4, Aula 07
#
# **Programa:** OxeTech Academy &nbsp;|&nbsp; **Disciplina:** Inteligencia Artificial e Aprendizado de Maquina (Avancado)
# **Instrutor:** Prof. Me. Derek Nielsen Araujo Alves
#
# ---
#
# **Escopo:** a atividade trata exclusivamente de **redes neurais** — os modelos de maior complexidade
# estudados na ultima aula. Percorre-se a progressao do **neuronio unico (perceptron)** ate a **rede
# multicamada (MLP)**, exercitando funcoes de ativacao, otimizadores e a estrategia correta de
# particionamento (treino / validacao / teste).
#
# **Como resolver:**
# - Preencha apenas os trechos delimitados por `# Seu codigo aqui` e `# Fim do codigo`.
# - Os marcadores trazem variaveis inicializadas em `None`; substitua-as pela sua solucao.
# - Execute as celulas em ordem, de cima para baixo.
# - Cada secao e seguida por uma celula de **validacao**, que **registra** o resultado (`[OK]`/`[X]`) e
#   **nao interrompe** a execucao — o notebook segue ate o fim mesmo com erros.
# - A ultima celula emite o **relatorio de desempenho** com a contagem total e o detalhamento por secao.
#
# **O que os asserts verificam:** dados e alvo bem-formados, particao treino/teste **sem vazamento**,
# presenca de **conjunto de validacao**, **modelo treinado** (gera previsoes validas no teste),
# **generalizacao** (supera piso e baseline) e o calculo da **auditoria de equidade** por subgrupo.
#

# %% [markdown]
# ## Secao 0 — Preparacao do Ambiente

# %%
# Execucao fora do Colab pode exigir: !pip install -q scikit-learn pandas numpy tensorflow

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
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


# %% [markdown]
# ## Secao 1 — Carga dos Dados e Definicao do Alvo
#
# Utiliza-se o conjunto **Adult / Census Income**: prever se a renda anual ultrapassa 50 mil dolares.
# Os atributos sensiveis `sex` e `race` sao reservados para a auditoria de equidade da Secao 7.
# A carga tenta o OpenML e possui rota de contingencia para execucao offline.

# %%
def carregar_dados():
    # Rota principal: CSV unico hospedado no GitHub (rapido e estavel, ~0.7s).
    url = "https://raw.githubusercontent.com/pooja2512/Adult-Census-Income/master/adult.csv"
    try:
        dados = pd.read_csv(url)
    except Exception:
        # Contingencia: OpenML com parser pandas (evita o parser ARFF lento).
        from sklearn.datasets import fetch_openml
        dados = fetch_openml("adult", version=2, as_frame=True, parser="pandas").frame
    # Normaliza o nome da coluna-alvo para 'class'.
    if "income" in dados.columns:
        dados = dados.rename(columns={"income": "class"})
    return dados

df = carregar_dados()
print("Dimensoes:", df.shape)
print(df["class"].value_counts())
df.head()

# %%
# Seu codigo aqui
y = df["class"].astype(str).str.contains(">50K").astype(int)
# Fim do codigo

# %%
# Matriz de atributos codificada + copia dos atributos sensiveis (para auditoria).
_dados = df.replace("?", np.nan)
X_proc = pd.get_dummies(_dados.drop(columns=["class"]), drop_first=True).astype(float).fillna(0.0)
X_proc = X_proc.reset_index(drop=True)
sensiveis = _dados[["sex", "race"]].astype(str).fillna("desconhecido").reset_index(drop=True)
print("Matriz de atributos:", X_proc.shape)

# %%
verificar("1", "alvo binario definido corretamente",
          lambda: set(pd.unique(y)) <= {0, 1} and len(y) == len(df))
verificar("1", "matriz de atributos construida",
          lambda: X_proc.shape[0] == len(df) and X_proc.shape[1] > 10)

# %% [markdown]
# ## Secao 2 — Particao Treino/Teste sem Vazamento
#
# Separe treino e teste com `test_size=0.2`, `stratify=y` e `random_state=SEED`. Particione **em
# conjunto** os atributos sensiveis e o vetor de indices `indices_full`, para permitir tanto o
# alinhamento da auditoria quanto a verificacao explicita de **nao sobreposicao** (anti-vazamento)
# entre treino e teste.

# %%
indices_full = np.arange(len(X_proc))

# %%
# Seu codigo aqui
X_train, X_test, y_train, y_test, sens_train, sens_test, idx_train, idx_test = train_test_split(
    X_proc, y, sensiveis, indices_full, test_size=0.2, stratify=y, random_state=SEED
)
# Fim do codigo

# %%
verificar("2", "particao treino/teste valida",
          lambda: len(X_train) > 0 and len(X_test) > 0)
verificar("2", "estratificacao preservada",
          lambda: abs(float(np.mean(y_train)) - float(np.mean(y_test))) < 0.03)
verificar("2", "treino e teste disjuntos (sem vazamento)",
          lambda: set(idx_train).isdisjoint(set(idx_test))
                  and len(idx_train) + len(idx_test) == len(X_proc))
verificar("2", "atributos sensiveis alinhados ao teste",
          lambda: len(sens_test) == len(X_test))

# %% [markdown]
# ## Secao 3 — Padronizacao e Conjunto de Validacao
#
# Redes neurais ajustam pesos por descida de gradiente e sao sensiveis a escala. A padronizacao abaixo
# e ajustada **apenas no treino** (`StandardScaler.fit(X_train)`), evitando vazamento.
#
# Em seguida, separe do treino um **conjunto de validacao** (`X_val_s`), usado para acompanhar a
# generalizacao durante o treinamento — distinto do conjunto de teste, que so e tocado ao final.

# %%
escala = StandardScaler().fit(X_train)
X_train_s = escala.transform(X_train)
X_test_s = escala.transform(X_test)
print("Treino padronizado:", X_train_s.shape, "| Teste padronizado:", X_test_s.shape)

# %%
# Seu codigo aqui
# Separe o treino padronizado em treino efetivo (X_tr_s) e validacao (X_val_s),
# com test_size=0.2, stratify=y_train e random_state=SEED.
X_tr_s, X_val_s, y_tr, y_val = train_test_split(
    X_train_s, y_train, test_size=0.2, stratify=y_train, random_state=SEED
)
# Fim do codigo

# %%
verificar("3", "conjunto de validacao criado e nao vazio",
          lambda: len(X_val_s) > 0)
verificar("3", "soma treino-efetivo + validacao = treino original",
          lambda: len(X_tr_s) + len(X_val_s) == len(X_train_s))

# %% [markdown]
# ## Secao 4 — Perceptron: o Neuronio Artificial Unico
#
# O perceptron e a unidade fundamental: uma unica camada densa com ativacao sigmoide, treinada por
# gradiente. Equivale a uma fronteira de decisao linear. Construa esse modelo em Keras, **compile**
# com funcao de perda de classificacao binaria, e **treine** usando `validation_data=(X_val_s, y_val)`
# para registrar a evolucao da validacao.

# %%
# Seu codigo aqui
perceptron = keras.Sequential([keras.layers.Dense(1, activation='sigmoid', input_shape=(X_tr_s.shape[1],))])
perceptron.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
perceptron.fit(X_tr_s, y_tr, epochs=15, batch_size=32, validation_data=(X_val_s, y_val), verbose=0)
proba_perc = perceptron.predict(X_test_s).flatten()
# Fim do codigo

# %%
verificar("4", "perceptron treinado e prevendo no teste",
          lambda: proba_perc is not None and len(proba_perc) == len(X_test_s))
verificar("4", "uso de validacao registrado no historico",
          lambda: "val_loss" in perceptron.history.history)

# %% [markdown]
# ## Secao 5 — Rede Multicamada (MLP): Profundidade e Nao Linearidade
#
# Empilhe **camadas ocultas** com funcoes de ativacao nao lineares (ex.: ReLU) para que a rede capture
# interacoes que o neuronio unico nao alcanca. Escolha um **otimizador** (ex.: Adam), compile com perda
# de classificacao binaria e treine com `validation_data`. A validacao confere estrutura, otimizador,
# generalizacao acima de um piso tolerante e superacao do baseline da classe majoritaria.

# %%
# Seu codigo aqui
rede = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_tr_s.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
rede.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
rede.fit(X_tr_s, y_tr, epochs=20, batch_size=32, validation_data=(X_val_s, y_val), verbose=0)
proba_rede = rede.predict(X_test_s).flatten()
acc_rede = accuracy_score(y_test, (proba_rede >= 0.5).astype(int))
# Fim do codigo

# %%
verificar("5", "rede com camada(s) oculta(s)",
          lambda: len(rede.layers) >= 2)
verificar("5", "perda de classificacao binaria",
          lambda: "binary_crossentropy" in str(rede.loss).lower().replace(" ", ""))
verificar("5", "otimizador configurado",
          lambda: rede.optimizer is not None)
verificar("5", "rede prevendo no teste",
          lambda: proba_rede is not None and len(proba_rede) == len(X_test_s))
verificar("5", "acuracia acima do piso (0.78)",
          lambda: float(acc_rede) > 0.78)
verificar("5", "supera o baseline da classe majoritaria",
          lambda: float(acc_rede) > max(float(np.mean(y_test)), 1 - float(np.mean(y_test))))

# %% [markdown]
# ## Secao 6 — Comparacao: Perceptron vs. MLP
#
# Reuna a acuracia dos dois modelos sobre o teste. A comparacao evidencia o ganho (ou nao) obtido ao
# adicionar profundidade e nao linearidade ao neuronio unico.

# %%
# Seu codigo aqui
pred_perc = (proba_perc >= 0.5).astype(int)
acc_perc = accuracy_score(y_test, pred_perc)
tabela = pd.DataFrame({
    "Modelo": ["Perceptron", "Rede Multicamada (MLP)"],
    "Acuracia": [acc_perc, acc_rede]
})
# Fim do codigo

# %%
verificar("6", "tabela comparativa com os dois modelos",
          lambda: tabela.shape[0] == 2 and "Acuracia" in tabela.columns)

# %% [markdown]
# ## Secao 7 — Auditoria de Equidade entre Subgrupos
#
# A acuracia global oculta disparidades entre grupos. Usando as previsoes da **MLP**, calcule por
# subgrupo de `sex` a **taxa de predicao positiva** e a **taxa de verdadeiro positivo** (sensibilidade).
# A validacao confere apenas a correcao mecanica; a interpretacao — paridade demografica versus
# igualdade de oportunidades — compoe a entrega escrita avaliada por rubrica.

# %%
# Seu codigo aqui
df_resultado = pd.DataFrame({
    'sex': sens_test['sex'].values,
    'y_true': np.array(y_test),
    'y_pred': (proba_rede >= 0.5).astype(int)
})
metricas_grupo = df_resultado.groupby('sex').apply(lambda x: pd.Series({
    'taxa_predicao_positiva': x['y_pred'].mean(),
    'taxa_verdadeiro_positivo': x[x['y_true'] == 1]['y_pred'].mean()
}))
# Fim do codigo

# %%
verificar("7", "metricas calculadas por subgrupo de sexo",
          lambda: metricas_grupo.shape[0] >= 2)
verificar("7", "colunas de equidade presentes",
          lambda: "taxa_predicao_positiva" in metricas_grupo.columns
                  and "taxa_verdadeiro_positivo" in metricas_grupo.columns)

# %% [markdown]
# ## Encerramento — Relatorio de Desempenho
#
# Consolida todas as verificacoes da sessao: total de asserts aprovados, percentual e detalhamento por secao.

# %%
rel = pd.DataFrame(_resultados)
print("=" * 52)
print("  RELATORIO DE DESEMPENHO — NOTEBOOK AVALIATIVO")
print("=" * 52)
if len(rel) == 0:
    print("Nenhuma verificacao registrada. Execute as celulas em ordem.")
else:
    total = len(rel); aprovados = int(rel["ok"].sum())
    print(f"  Asserts aprovados: {aprovados} / {total}  ({aprovados/total*100:.1f}%)")
    print("-" * 52)
    por_secao = rel.groupby("secao")["ok"].agg(["sum", "count"])
    por_secao = por_secao.reindex(sorted(por_secao.index, key=lambda s: int(s)))
    for sec, linha in por_secao.iterrows():
        print(f"  Secao {sec:>2}: {int(linha['sum'])}/{int(linha['count'])} aprovados")
    pendentes = rel[~rel["ok"]]
    if len(pendentes) > 0:
        print("-" * 52); print("  Itens pendentes:")
        for _, r in pendentes.iterrows():
            print(f"    [X] Secao {r['secao']} — {r['descricao']}")
print("=" * 52)
