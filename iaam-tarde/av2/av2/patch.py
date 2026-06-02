import re

with open("AV2_RedesNeurais_Aluno.py", "r") as f:
    content = f.read()

s1 = """# Seu codigo aqui
y = df["class"].astype(str).str.contains(">50K").astype(int)
# Fim do codigo"""
content = re.sub(r"# Seu codigo aqui\n# Defina 'y'.*?\ny = None\n# Fim do codigo", s1, content, flags=re.DOTALL)

s2 = """# Seu codigo aqui
X_train, X_test, y_train, y_test, sens_train, sens_test, idx_train, idx_test = train_test_split(
    X_proc, y, sensiveis, indices_full, test_size=0.2, stratify=y, random_state=SEED
)
# Fim do codigo"""
content = re.sub(r"# Seu codigo aqui\nX_train = None.*?\nidx_test = None\n# Fim do codigo", s2, content, flags=re.DOTALL)

s3 = """# Seu codigo aqui
# Separe o treino padronizado em treino efetivo (X_tr_s) e validacao (X_val_s),
# com test_size=0.2, stratify=y_train e random_state=SEED.
X_tr_s, X_val_s, y_tr, y_val = train_test_split(
    X_train_s, y_train, test_size=0.2, stratify=y_train, random_state=SEED
)
# Fim do codigo"""
content = re.sub(r"# Seu codigo aqui\n# Separe o treino.*?y_val = None\n# Fim do codigo", s3, content, flags=re.DOTALL)

s4 = """# Seu codigo aqui
perceptron = keras.Sequential([keras.layers.Dense(1, activation='sigmoid', input_shape=(X_tr_s.shape[1],))])
perceptron.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
perceptron.fit(X_tr_s, y_tr, epochs=15, batch_size=32, validation_data=(X_val_s, y_val), verbose=0)
proba_perc = perceptron.predict(X_test_s).flatten()
# Fim do codigo"""
content = re.sub(r"# Seu codigo aqui\nperceptron = None\nproba_perc = None\n# Fim do codigo", s4, content, flags=re.DOTALL)

s5 = """# Seu codigo aqui
rede = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_tr_s.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
rede.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
rede.fit(X_tr_s, y_tr, epochs=20, batch_size=32, validation_data=(X_val_s, y_val), verbose=0)
proba_rede = rede.predict(X_test_s).flatten()
acc_rede = accuracy_score(y_test, (proba_rede >= 0.5).astype(int))
# Fim do codigo"""
content = re.sub(r"# Seu codigo aqui\nrede = None\nproba_rede = None\nacc_rede = None\n# Fim do codigo", s5, content, flags=re.DOTALL)

s6 = """# Seu codigo aqui
pred_perc = (proba_perc >= 0.5).astype(int)
acc_perc = accuracy_score(y_test, pred_perc)
tabela = pd.DataFrame({
    "Modelo": ["Perceptron", "Rede Multicamada (MLP)"],
    "Acuracia": [acc_perc, acc_rede]
})
# Fim do codigo"""
content = re.sub(r"# Seu codigo aqui\ntabela = None\n# Fim do codigo", s6, content, flags=re.DOTALL)

s7 = """# Seu codigo aqui
df_resultado = pd.DataFrame({
    'sex': sens_test['sex'].values,
    'y_true': np.array(y_test),
    'y_pred': (proba_rede >= 0.5).astype(int)
})
metricas_grupo = df_resultado.groupby('sex').apply(lambda x: pd.Series({
    'taxa_predicao_positiva': x['y_pred'].mean(),
    'taxa_verdadeiro_positivo': x[x['y_true'] == 1]['y_pred'].mean()
}))
# Fim do codigo"""
content = re.sub(r"# Seu codigo aqui\nmetricas_grupo = None\n# Fim do codigo", s7, content, flags=re.DOTALL)

with open("AV2_RedesNeurais_Aluno.py", "w") as f:
    f.write(content)

