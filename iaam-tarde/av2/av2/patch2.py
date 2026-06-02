import re

with open("AV2_RedesNeurais_Aluno.py", "r") as f:
    content = f.read()

# Fix Secao 1 to ensure 'y' is a Pandas Series (fixes NoneType subscript error)
s1 = """# Seu codigo aqui
# Defina 'y' como o alvo BINARIO (1 se a renda for maior que 50K, 0 caso contrario).
y = df["class"].astype(str).str.contains(">50K").astype(int)
# Fim do codigo"""
content = re.sub(r"# Seu codigo aqui\n# Defina 'y'.*?\n# Fim do codigo", s1, content, flags=re.DOTALL)

with open("AV2_RedesNeurais_Aluno.py", "w") as f:
    f.write(content)

