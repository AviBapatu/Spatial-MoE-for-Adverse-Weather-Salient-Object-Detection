import ast

with open("generate_notebook.py", "r") as f:
    code = f.read()

try:
    ast.parse(code)
    print("Syntax OK")
except SyntaxError as e:
    print(f"Syntax Error: {e}")
