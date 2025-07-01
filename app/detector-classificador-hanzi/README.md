# Detector Classificador Hanzi

Este projeto é uma aplicação de reconhecimento de caracteres utilizando Kivy e Pillow. A aplicação permite capturar imagens através da câmera, selecionar uma área de interesse, e processar essa área para reconhecimento de caracteres.

## Estrutura do Projeto

```
detector-classificador-hanzi
├── app
│   ├── main.kv        # Define a interface do usuário para a aplicação Kivy
│   └── main.py        # Contém a lógica principal da aplicação
├── requirements.txt    # Lista de dependências necessárias
└── README.md           # Documentação do projeto
```

## Instalação

Para instalar as dependências necessárias, execute o seguinte comando:

```
pip install -r requirements.txt
```

## Uso

1. Execute o script `main.py` para iniciar a aplicação.
2. A aplicação abrirá uma visualização da câmera.
3. Clique no botão "Capturar Foto" para tirar uma foto.
4. Após a captura, um quadrado de seleção aparecerá sobre a imagem.
5. Ajuste o quadrado de seleção conforme necessário.
6. Clique em "Processar Seleção" para recortar a área selecionada e processá-la.
7. O resultado do reconhecimento será exibido na tela.

## Dependências

- Kivy: Para a interface gráfica e manipulação da câmera.
- Pillow: Para processamento de imagens.

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.

## Licença

Este projeto está licenciado sob a MIT License - veja o arquivo LICENSE para mais detalhes.