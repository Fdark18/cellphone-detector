# 1. Instalar dependências básicas
poetry install

# 2. Instalar PyTorch com CUDA manualmente
poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 3. Verificar instalação
poetry run python -c "import torch; print(f'CUDA disponível: {torch.cuda.is_available()}'); print(f'Versão PyTorch: {torch.__version__}')"

# se der erro Desinstalar PyTorch atual (sem CUDA) e faça todo processo de novo
poetry run pip uninstall torch torchvision -y