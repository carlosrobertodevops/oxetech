# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## O que é esta pasta

Atividade avaliativa **AV1 — Aula 05** da **OxeTech Academy** (disciplina IA e Aprendizado de Máquina, turma tarde), Prof. Me. Derek Nielsen Araújo Alves. Conteúdo em português do Brasil.

**Tema**: Laboratório Prático de **Diagnóstico de Diabetes** — construir e avaliar um classificador de **Regressão Logística** (classificação binária).

- **Dataset**: Pima Indians Diabetes (768 pacientes, 8 atributos clínicos), alvo `outcome` (1 = diabetes, 0 = sem).
- Carregado **via URL raw do GitHub** (`jbrownlee/Datasets/.../pima-indians-diabetes.data.csv`, `header=None`, `names=COLUNAS`)
- offline-safe, sem download manual nem mount de Drive.
- **Métrica**: **F1** (não accuracy — classes desbalanceadas), com **validação cruzada estratificada de 5 folds**.
- **Nota ética** (no enunciado): dataset Pima é criticado por sub-representação; modelo pode não generalizar para populações como quilombolas de Alagoas. Manter ao interpretar.

## Arquivos e pareamento jupytext

Os `.py` são espelhos **jupytext formato `percent`** dos `.ipynb` (células `# %%` e `# %% [markdown]`). Editar o `.py` é mais barato que o JSON do `.ipynb`.

- `Aula05_AV1.ipynb` / `.py` — **RESOLVIDO (gabarito)**: os 5 blocos `# --- SEU CÓDIGO AQUI ---` … `# --- FIM DO CÓDIGO ---` estão preenchidos. Execução real do dataset → **Relatório Final 5/5, nota 10.0**. Editar soluções **dentro desses marcadores**, no `.py`, e sincronizar o `.ipynb`.
- `Aula05_AV1_colab.ipynb` / `.py` — **variante Colab**: mesmo corpo, imports reagrupados; já carrega via URL, então não monta Drive. Os `.ipynb` carregam metadados/outputs de execução do Colab. (Ainda em formato template — propagar o gabarito do `.py` principal se necessário.)

### Gabarito dos checkpoints (referência)

- **CP1** `X = df.drop(columns='outcome')` · `y = df['outcome']`
- **CP2** `train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)`
- **CP3** `Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(max_iter=1000, random_state=42))])` + `pipe.fit(X_train, y_train)`
- **CP4** `cross_val_score(pipe, X, y, cv=cv, scoring='f1')` → `.mean()` / `.std()` (`cv` já definido no bloco)
- **CP5** `coef_[0]` → `df_coef('atributo','coeficiente')` ordenado por `|β|` desc → `top3 = df_coef['atributo'].head(3).tolist()`. Resultado real: `top3 = ['glucose','bmi','pregnancies']`.

## Executar

```bash
pip install -q scikit-learn pandas numpy jupytext
jupyter nbconvert --to notebook --execute --inplace Aula05_AV1.ipynb   # roda tudo
jupytext --to py:percent *.ipynb
jupytext --sync Aula05_AV1.py                                          # .py <-> .ipynb
python -m py_compile Aula05_AV1.py Aula05_AV1_colab.py                 # validar sintaxe
```

Não há build, lint nem suíte de testes externa — a verificação é interna ao notebook (ver abaixo).

## Padrão de validação (Checkpoints com `assert` + nota automática)

Cada Checkpoint é seguido por uma célula de validação que usa **`assert` que aborta** se a solução estiver errada (ex.: `assert media_f1 > 0.55, "..."`). A célula final **`📊 Relatório Final — AV1-05`** (não editar) reavalia as condições, conta `checkpoints_passed / checkpoints_total` e calcula:

```python
nota = round((checkpoints_passed / checkpoints_total) * 10, 1)
```

Toda solução nova deve fazer os `assert` do seu Checkpoint passarem para somar à nota.

## Estrutura (5 Checkpoints)

- **Diagnóstico 1** — exploração do dataset (célula pronta, só executar).
- **Checkpoint 1** — preparar features `X` e target `y` (`outcome`).
- **Checkpoint 2** — `train_test_split` **estratificado** treino/teste (`random_state=42`).
- **Checkpoint 3** — `Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(max_iter=1000, random_state=42))])`; gate `accuracy > 0.70` no teste.
- **Checkpoint 4** — `cross_val_score` com `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, `scoring='f1'` → `scores_cv` (5 valores), `media_f1`, `std_f1`; gate `media_f1 > 0.55`.
- **Checkpoint 5** — interpretação dos coeficientes da Regressão Logística.

## Convenções

- Idioma: PT-BR; identificadores sem acento, comentários/markdown em português.
- Reprodutibilidade: sempre `random_state=42`.
- Anti-vazamento: `StandardScaler` dentro do `Pipeline` (fit só no treino via `cross_val`/`fit`); teste tocado apenas para o gate final.
