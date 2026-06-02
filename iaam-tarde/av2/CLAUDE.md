# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## O que é este repositório

Atividade avaliativa (AV2) da **OxeTech Academy** — disciplina de IA e Aprendizado de Máquina (Avançado), Módulo 4 Aula 07. Conteúdo em português do Brasil. Dois entregáveis:

1. **Notebook de redes neurais** (raiz) — progressão Perceptron → MLP com TensorFlow/Keras sobre o dataset Adult/Census Income, com auditoria de equidade.
2. **`projeto-final/`** — projeto final sobre dados de criminalidade NEAC (CSVs em `projeto-final/dados/`), especificação em `Especificacao_Projeto_Final_AV2.docx.pdf`.

## Arquivos e o pareamento jupytext

Os `.py` são espelhos **jupytext em formato `percent`** dos `.ipynb` (células delimitadas por `# %%` e `# %% [markdown]`). Editar o `.py` e sincronizar é mais barato que mexer no JSON do `.ipynb`.

- `AV2_RedesNeurais_Aluno.ipynb` / `.py` — **template do aluno**: trechos a preencher ficam entre `# Seu codigo aqui` e `# Fim do codigo`, com variáveis inicializadas em `None`.
- `AV2_RedesNeurais_Carlos-Roberto.ipynb` — versão **resolvida** (entrega).
- `*.zed.ipynb` / `*.zed.py` — variantes de trabalho no editor Zed.
- `patch.py` / `patch2.py` — geradores que **injetam soluções** no template: leem `AV2_RedesNeurais_Aluno.py`, substituem cada bloco `# Seu codigo aqui ... # Fim do codigo` por regex (`re.sub(..., flags=re.DOTALL)`) e reescrevem o `.py`.

Ao preencher soluções, a abordagem do projeto é **editar via patch script** (regex sobre os marcadores), não editar o notebook célula a célula.

## Executar

```bash
pip install -q scikit-learn pandas numpy tensorflow jupytext
jupyter nbconvert --to notebook --execute --inplace AV2_RedesNeurais_Aluno.ipynb   # roda tudo
jupytext --sync AV2_RedesNeurais_Aluno.py                                          # .py <-> .ipynb
python patch.py                                                                    # regenera o .py com soluções
```

Não há build, lint nem suíte de testes — a verificação é interna ao notebook (ver abaixo).

## Padrão de validação não-interruptiva (central)

O notebook **nunca usa `assert` que aborta**. A Seção 0 define um coletor:

```python
def verificar(secao, descricao, teste):  # 'teste' é lambda sem argumentos
    try: ok = bool(teste())
    except Exception: ok = False         # exceção = falha registrada, nunca propagada
    _resultados.append({"secao": secao, "descricao": descricao, "ok": ok})
    print(f"[{'OK ' if ok else ' X '}] Secao {secao} — {descricao}")
```

Cada seção é seguida por células `verificar(...)`; a célula final (`Relatorio de Desempenho`) agrega `_resultados` em DataFrame e imprime aprovados/total por seção. **O notebook roda até o fim mesmo com células incompletas** — uma solução errada vira `[ X ]`, não um traceback. Toda solução nova deve fazer os `verificar(...)` da sua seção passarem.

## Estrutura do notebook (Seções 0–7)

- **0** Ambiente: imports, `SEED=42` em numpy+tf, coletor `verificar`.
- **1** Carga: `carregar_dados()` puxa Adult CSV de URL do GitHub, com fallback `fetch_openml("adult", version=2, parser="pandas")`; alvo binário `y` (renda > 50K → 1). `sex` e `race` reservados para a auditoria.
- **2** Partição treino/teste **sem vazamento**: `train_test_split(..., test_size=0.2, stratify=y, random_state=SEED)`, particionando junto `sensiveis` e `indices_full` para checar disjunção `idx_train`/`idx_test`.
- **3** `StandardScaler.fit` **só no treino** + separar conjunto de validação (`X_tr_s`/`X_val_s`).
- **4** Perceptron: 1 camada `Dense(1, sigmoid)`, `binary_crossentropy`, treino com `validation_data=(X_val_s, y_val)`.
- **5** MLP: camadas ocultas (ReLU) + otimizador; pisos de qualidade — `acc > 0.78` e acima do baseline da classe majoritária.
- **6** Tabela comparativa Perceptron vs MLP (precisa de 2 linhas e coluna `Acuracia`).
- **7** Auditoria de equidade por subgrupo de `sex`: `taxa_predicao_positiva` e `taxa_verdadeiro_positivo`.

## `projeto-final/` — Projeto Final Integrador (AV2)

Entregável de peso 50% (spec: `Especificacao_Projeto_Final_AV2.docx.pdf`). Solução completa de IA sobre criminalidade de Alagoas, gerada em `projeto-final.ipynb` / `projeto-final.py` (mesmo pareamento jupytext `percent` do notebook do aluno).

- **Problema**: classificação **multiclasse** da natureza da ocorrência `tipo_ocorrencia ∈ {ARMA, CVLI, DROGA}`, unificando os 3 CSVs `dados/NEAC_{ARMA,CVLI,DROGA}_2022.csv`. Métrica: **macro-F1** (classes desbalanceadas).
- **Quirks dos CSVs** (já tratados em `carregar()`/`coords_para_float()`): `sep=','`, encoding utf-8 com fallback latin-1; `LONGITUDE`/`LATITUDE` vêm como string com vírgula decimal entre aspas (`"-36,669077"`) → `str.replace(',','.')` + `pd.to_numeric`; `DATA_HORA_FATO` é `dd/mm/yyyy HH:MM:SS` → `to_datetime(dayfirst=True)`; tokens `NI`/`SEM INFORMAÇÃO`/`NÃO INFORMADO` → NaN. Colunas de vítima (`COR_RACA_VITIMA`, `SEXO_VITIMA`, ...) só existem no CVLI.
- **Anti-vazamento**: features só espaço-temporais (`num_cols`/`cat_cols`); `QUALIFICACAO`, `INSTRUMENTO_UTILIZADO`, `SUBJETIVIDADE` e atributos sensíveis de vítima são **excluídos do treino** (vazariam o alvo / discriminação). `ColumnTransformer` (`preprocess`) só é fitado **dentro do `Pipeline`**, sobre o treino.
- **Seções** (mesmo coletor `verificar()` não-interruptivo do notebook do aluno): 0 ambiente · 1 carga/unificação · 2 EDA · 3 pré-processamento · 4 baseline (Dummy+LogReg, `f1_base`) · 5 modelo final RandomForest + Árvore de Decisão (`plot_tree`, interpretável) + MLP Keras + `tabela_comp` (4 modelos) · 6 interpretabilidade (importância + SHAP opcional) · 7 auditoria de equidade/vieses (proxy de raça via cidade; perfil de vítimas CVLI) · 8 deploy.
- **Deploy** (Seção 8 escreve em `projeto-final/deploy/` ao rodar): `modelo.joblib`, `metadata.json`, `app.py` (FastAPI `/health` + `/prever`), `requirements.txt`, `Dockerfile`, `README.md`. `modelo_final = modelo_rf` (Pipeline recebe campos crus).
- **Dependências extras** vs notebook do aluno: `joblib`, `fastapi`, `uvicorn` (deploy), `shap` (opcional — Seção 6 cai em fallback se ausente). TensorFlow opcional: MLP é ignorada se `import tensorflow` falhar (`_TF_OK`).
- **Carga multi-ambiente** (`resolver_dados_dir()` + `_tem_csv()` na Seção 1): verifica **primeiro** `./dados` local (e o caminho opcional `DADOS_DRIVE`) **sem efeito colateral** — não monta o Drive se os CSVs já estão locais. Só se não achar **e** estiver no Colab (`import google.colab`) é que monta o Google Drive e tenta `MyDrive/oxetech/projeto-final/dados`. Retorna o 1º diretório com `.csv`. Um `verificar("1", ...)` confere que os 3 `NEAC_*_2022.csv` existem no diretório resolvido. Mesma lógica roda local e no Colab — `projeto-final.ipynb` é único para os dois ambientes.
- **Variante Colab** `projeto_final_colab.ipynb` / `.py`: é o mesmo corpo de `projeto-final.py` com 3 células de prelúdio injetadas antes da Seção 1 (markdown de instruções, `pip install` de extras, célula de config `DADOS_DRIVE`). Gerada por injeção do prelúdio + `jupytext`, não editada à mão.
- **Regenerar**: `jupytext --to ipynb projeto-final.py -o projeto-final.ipynb` (e idem para `projeto_final_colab.py`). Validar sintaxe: `python -m py_compile projeto-final.py projeto_final_colab.py`. Execução local valida 24/25 asserts (o assert da MLP Keras exige TensorFlow, ausente em Python 3.13 local; passa no Colab → 25/25).

## Convenções

- Idioma: PT-BR sem acentos no código/identificadores (`secao`, `acuracia`, `verificar`); comentários e markdown em português.
- Reprodutibilidade: sempre `random_state=SEED` (=42) e `np.random.seed`/`tf.random.set_seed`.
- Anti-vazamento é requisito avaliado: `fit` de scaler e split só no treino; teste tocado apenas no fim.
