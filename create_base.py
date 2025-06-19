from pathlib import Path

output = "data/dataset.csv"

def iterar_pastas_pathlib(caminho_pasta):
  """Itera sobre os arquivos e subpastas dentro de um diretório usando o módulo pathlib.

  Args:
    caminho_pasta: O caminho da pasta a ser iterada (Path object).
  """
  for item in caminho_pasta.iterdir():
      if item.is_file():
          print(f"Arquivo: {item}")
      elif item.is_dir():
          print(f"Diretório: {item}")
          # Pode chamar a função recursivamente para iterar sobre subpastas
          iterar_pastas_pathlib(item)

# Exemplo de uso
caminho_raiz = Path("/caminho/para/sua/pasta")  # Substitua pelo caminho da sua pasta
iterar_pastas_pathlib(caminho_raiz)