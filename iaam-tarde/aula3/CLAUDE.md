# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Laboratório prático guiado da **ÔxeTech Academy** (curso IAAM — turma da tarde, aula 3): engenharia de dados para Machine Learning usando o dataset **Auto MPG**. Não é uma aplicação — é um caderno de exercícios onde o aluno preenche blocos `# --- SEU CÓDIGO AQUI ---` e valida cada passo com um checkpoint automático. Treinamento livre (não avaliativo); aquecimento para as AVs.

## Status: resolvido

Os 4 blocos `# --- SEU CÓDIGO AQUI ---` estão preenchidos e os 4 checkpoints passam (✅). Verificado por execução end-to-end (consenso de 3 subagentes). Soluções aplicadas:

- **PASSO 2 (split):** `train_test_split(X, y, test_size=0.2, random_state=42)` — `random_state=42` é obrigatório, senão CHECKPOINT 1 (`X_train.shape[0] == 318`) falha de forma intermitente.
- **PASSO 3 (imputação):** `SimpleImputer(strategy='median')` em `horsepower`.
- **PASSO 4 (encoding):** `OneHotEncoder(sparse_output=False, drop='first')` em `origin` (3 categorias → 2 colunas `origin_*`).
- **PASSO 5 (scaling):** `StandardScaler()` nas 6 colunas de `cols_para_escalar`.

Não existe seção "Relatório Final — AV1-05" neste arquivo; o notebook termina no CHECKPOINT 4 (PASSO 5).

## Files NÃO são pareados automaticamente (jupytext)

`atividade_proposta_aula3.ipynb` e `atividade_proposta_aula3.py` são o mesmo notebook ([jupytext](https://jupytext.readthedocs.io), formato `percent`, células `# %%`), porém **sem metadados de pairing** — `jupytext --sync` avisa "not a paired notebook" e não faz nada. Após editar o `.py`, regenerar o `.ipynb` explicitamente:

```bash
jupytext --to notebook --output atividade_proposta_aula3.ipynb atividade_proposta_aula3.py
```

(`--update` no lugar de `--output` preserva outputs/ids das células existentes.) Editar um e esquecer o outro deixa os dois divergentes.

## Estrutura pedagógica (não quebrar)

Dinâmica fixa a preservar ao editar:

- **DIAGNÓSTICO (🔍):** células que só mostram o problema (nulos, escala, texto). Não alterar — são didáticas.
- **SEU CÓDIGO AQUI:** lacunas que o aluno preenche. Manter o esqueleto e os comentários numerados (`# 1.`, `# 2.`) que guiam a resposta.
- **CHECKPOINT N:** validador automático após cada exercício; deve falhar enquanto o código estiver incompleto.

Sequência dos 5 passos, cada um dependente do anterior: leitura → split → imputer (`horsepower`) → onehot (`origin`) → scaler.

## Regra de ouro do conteúdo: sem data leakage

O lab inteiro existe para ensinar a evitar vazamento de dados. Em todo pré-processamento: `fit`/`fit_transform` **apenas no `X_train`**, somente `transform` no `X_test`. Qualquer código que faça `fit` no teste contradiz o objetivo da aula.

## Stack

`pandas`, `numpy`, `scikit-learn` (`train_test_split`, `SimpleImputer`, `OneHotEncoder`, `StandardScaler`), `seaborn` (dataset via `sns.load_dataset('mpg')`). Sem `requirements.txt` — instalar via pip conforme os imports do topo.
