# Detector e Classificador de Hanzis
## Feito como trabalho de Visão Computacional/Reconhecimento de Padrões - UFSC 2025
## Gian Carlo e Nemo R. L. Neto

#### Para extrair hanzis de uma imagem basta rodar
```bash
python extractor.py caminho/imagem -d` ou -m
```
#### e procurar a pasta segmented_characters

#### Para classificar sua imagem execute:
```bash
python predicted_image.py caminho/imagem
```
#### Os demais classificadores requerem
```bash
pip install torch torchvision timm==1.0.11 pandas fastai==2.7.18
```

#### Fontes disponíveis em: https://chinesefonts.org/
