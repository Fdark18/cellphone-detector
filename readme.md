# Sistema de Detecção de Uso de Celular em Ambiente Industrial

## Visão Geral

Sistema automatizado de monitoramento baseado em visão computacional para detectar o uso de dispositivos móveis por funcionários em ambientes industriais. Utiliza Deep Learning com arquitetura YOLOv8 para realizar detecções em tempo real através de câmeras de vigilância 
## Objetivo

Fornecer uma ferramenta de monitoramento não invasiva que identifique automaticamente quando colaboradores estão utilizando celulares durante o expediente, registrando evidências fotográficas e métricas de tempo de uso para fins de compliance e segurança operacional.

## Aplicações

- Monitoramento de zonas restritas em plantas industriais
- Controle de uso de dispositivos em áreas de produção
- Auditoria de conformidade com políticas de segurança
- Análise de padrões de comportamento operacional
- Geração de relatórios automáticos de incidentes

## Arquitetura do Sistema

### Componentes Principais
```
┌─────────────────────────────────────────────────────────┐
│ Servidor de Processamento (Linux)                      │
│ - Modelo YOLOv8 treinado                                │
│ - GPU NVIDIA (CUDA)                                     │
│ - Sistema de detecção em tempo real                    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ NVR (Network Video Recorder)                            │
│ - Streams RTSP das câmeras                              │
│ - Gravação de backup                                    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Câmeras IP (Rede Local)                                 │
│ - Monitoramento de áreas específicas                    │
│ - Resolução mínima: 1280x720                            │
└─────────────────────────────────────────────────────────┘
```

## Especificações Técnicas

### Requisitos de Hardware

**Servidor de Processamento:**
- Sistema Operacional: Ubuntu 22.04 LTS ou superior
- GPU: NVIDIA com suporte CUDA (mínimo 4GB VRAM)
- RAM: 16GB recomendado
- Armazenamento: 100GB+ para logs e evidências
- Processador: Intel i5/AMD Ryzen 5 ou superior

**Rede:**
- Conexão Gigabit Ethernet
- Acesso ao NVR via protocolo RTSP
- Largura de banda: 10 Mbps por câmera

### Requisitos de Software
```
Python: 3.12+
PyTorch: 2.5+ com CUDA 12.6
Ultralytics YOLOv8: 8.3+
OpenCV: 4.10+
CUDA Toolkit: 12.6
cuDNN: 8.9+
```

## Estrutura do Projeto
```
lib_label_detect_cell/
├── pyproject.toml              # Configuração Poetry
├── README.md                   # Documentação
├── notebook_treinamento.ipynb  # Notebook para treinamento
├── teste_camera.py             # Script de produção
├── best.pt                     # Modelo treinado
├── dataset.yaml                # Configuração do dataset
│
├── dataset/                    # Dataset organizado
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
│
├── deteccoes/                  # Evidências e logs
│   ├── celular_YYYYMMDD_HHMMSS.jpg
│   └── log_YYYYMMDD.txt
│
└── runs/                       # Outputs do treinamento
    └── detect/
        └── celular_detector/
            ├── weights/
            │   └── best.pt
            └── results.png
```

## Instalação

### 1. Preparação do Ambiente
```bash
# Atualizar sistema
sudo apt update && sudo apt upgrade -y

# Instalar dependências do sistema
sudo apt install python3-pip python3-dev build-essential -y

# Verificar NVIDIA drivers
nvidia-smi

# Instalar CUDA Toolkit (se necessário)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt update
sudo apt install cuda-12-6 -y
```

### 2. Instalação do Projeto
```bash
# Clonar ou criar diretório do projeto
mkdir -p ~/lib_label_detect_cell
cd ~/lib_label_detect_cell

# Instalar Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Instalar dependências
poetry install

# Instalar PyTorch com CUDA
poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 3. Verificação da Instalação
```bash
# Verificar CUDA
poetry run python -c "import torch; print(f'CUDA disponível: {torch.cuda.is_available()}')"

# Verificar GPU
poetry run python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

## Treinamento do Modelo

### 1. Preparação do Dataset

O sistema utiliza o dataset MUID-IITR do Kaggle, contendo imagens de pessoas usando celulares em atividades cotidianas.
```bash
# Executar notebook de treinamento
poetry run jupyter notebook notebook_treinamento.ipynb
```

### 2. Configuração de Treinamento

Parâmetros recomendados para GPU RTX 2050 (4GB):
```python
MODELO_BASE = 'yolov8m.pt'      # Modelo Medium
EPOCHS = 100                     # Número de épocas
BATCH_SIZE = 12                  # Tamanho do batch
IMG_SIZE = 640                   # Resolução das imagens
PATIENCE = 20                    # Early stopping
```

### 3. Processo de Treinamento

Executar sequencialmente as células do notebook:

1. Instalar dependências
2. Baixar dataset do Kaggle
3. Preparar dados no formato YOLO
4. Configurar parâmetros de treinamento
5. Treinar modelo
6. Avaliar métricas
7. Exportar modelo treinado

Tempo estimado de treinamento: 2-3 horas em RTX 2050

### 4. Métricas de Avaliação

O sistema calcula automaticamente:

- **mAP50**: Mean Average Precision a 50% IoU
- **mAP50-95**: Mean Average Precision de 50% a 95% IoU
- **Precision**: Taxa de acertos nas detecções
- **Recall**: Taxa de detecção dos objetos
- **F1-Score**: Média harmônica entre Precision e Recall
- **Matriz de Confusão**: Análise de erros e acertos
- **Curva Precision-Recall**: Análise de threshold ótimo

Métricas esperadas:
- mAP50 > 0.80: Excelente
- mAP50 > 0.70: Bom
- mAP50 > 0.50: Aceitável

## Implantação em Produção

### 1. Configuração do NVR

Obter URLs RTSP das câmeras no formato:
```
rtsp://usuario:senha@IP_NVR:554/stream1
rtsp://usuario:senha@IP_NVR:554/stream2
```

### 2. Configuração do Script

Editar `teste_camera.py`:
```python
# Configurações principais
MODELO_PATH = 'best.pt'          # Caminho do modelo treinado
CAMERA_ID = 0                    # ID da câmera ou URL RTSP
CONFIDENCE_THRESHOLD = 0.6       # Threshold de confiança (0.0-1.0)
RESOLUCAO = (1280, 720)          # Resolução do vídeo
```

Para múltiplas câmeras via RTSP:
```python
CAMERA_ID = 'rtsp://usuario:senha@192.168.1.100:554/stream1'
```

### 3. Execução
```bash
# Modo interativo (desenvolvimento)
poetry run python teste_camera.py

# Modo daemon (produção)
nohup poetry run python teste_camera.py > output.log 2>&1 &
```

### 4. Systemd Service (Recomendado)

Criar arquivo `/etc/systemd/system/detector-celular.service`:
```ini
[Unit]
Description=Sistema de Detecção de Celular
After=network.target

[Service]
Type=simple
User=seu_usuario
WorkingDirectory=/home/seu_usuario/lib_label_detect_cell
ExecStart=/home/seu_usuario/.local/bin/poetry run python teste_camera.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Ativar serviço:
```bash
sudo systemctl daemon-reload
sudo systemctl enable detector-celular
sudo systemctl start detector-celular
sudo systemctl status detector-celular
```

## Funcionalidades do Sistema

### Detecção em Tempo Real

- Processamento de vídeo a 25-30 FPS
- Latência < 100ms por frame
- Detecção com threshold configurável
- Bounding boxes coloridos por confiança

### Registro de Eventos

Cada detecção gera:

1. **Evidência fotográfica**: Imagem JPEG com timestamp
2. **Log de evento**: Registro com data, hora e duração
3. **Métricas de tempo**: Contagem automática do tempo de uso

Formato do log:
```
[2025-10-24 14:32:15] DETECÇÃO INICIADA - Evidência: celular_20251024_143215.jpg
[2025-10-24 14:35:42] DETECÇÃO FINALIZADA - Duração: 207s
```

### Interface Visual

- Status de monitoramento em tempo real
- Bounding boxes nas detecções
- Contador de tempo de uso
- Indicadores visuais de confiança
- Informações de sistema

## Otimizações de Performance

### Configurações CUDA

O sistema ativa automaticamente:
```python
torch.backends.cudnn.benchmark = True       # Auto-tuning
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### Multiprocessing

Linux permite 8 workers para carregamento de dados, resultando em:

- 2-3x mais rápido que Windows
- Melhor utilização da GPU
- Menor uso de RAM

### Threading

Script de produção utiliza threads dedicadas:

- Thread de captura de vídeo
- Thread de inferência GPU
- Thread principal de exibição

## Manutenção

### Logs

Localização: `deteccoes/log_YYYYMMDD.txt`

Rotação recomendada: Diária
```bash
# Exemplo de rotação com logrotate
/home/usuario/lib_label_detect_cell/deteccoes/log_*.txt {
    daily
    rotate 30
    compress
    missingok
    notifempty
}
```

### Evidências Fotográficas

Localização: `deteccoes/celular_*.jpg`

Recomendações:
- Backup semanal
- Retenção de 90 dias
- Compressão após 30 dias

### Monitoramento do Sistema
```bash
# Verificar uso de GPU
watch -n 1 nvidia-smi

# Verificar logs em tempo real
tail -f deteccoes/log_$(date +%Y%m%d).txt

# Verificar status do serviço
sudo systemctl status detector-celular

# Verificar uso de disco
du -sh deteccoes/
```

## Resolução de Problemas

### GPU não detectada
```bash
# Verificar drivers
nvidia-smi

# Reinstalar drivers se necessário
sudo apt purge nvidia-*
sudo apt install nvidia-driver-535
sudo reboot
```

### Erro de memória GPU

Reduzir batch size ou resolução:
```python
BATCH_SIZE = 8
IMG_SIZE = 416
```

### Baixa taxa de FPS

1. Verificar carga da GPU: `nvidia-smi`
2. Reduzir resolução da câmera
3. Aumentar threshold de confiança
4. Processar frames alternados

### Falsos positivos

1. Aumentar threshold: `CONFIDENCE_THRESHOLD = 0.7`
2. Retreinar com mais dados negativos
3. Ajustar augmentations no treinamento

## Segurança e Compliance

### Proteção de Dados

- Logs e evidências armazenados localmente
- Acesso restrito via permissões Linux
- Comunicação via rede interna (sem internet)

### LGPD

Sistema projetado para:
- Monitorar comportamentos, não indivíduos
- Armazenamento local controlado
- Retenção configurável de dados
- Possibilidade de anonimização


### Políticas Recomendadas

1. Informar funcionários sobre monitoramento
2. Definir áreas de monitoramento claramente
3. Estabelecer política de uso de dispositivos
4. Documentar procedimentos de resposta
5. Revisar logs periodicamente

## Suporte Técnico

### Documentação Adicional

- Ultralytics YOLOv8: https://docs.ultralytics.com
- PyTorch: https://pytorch.org/docs
- OpenCV: https://docs.opencv.org

### Contato

Para questões técnicas ou suporte, consulte a equipe de TI ou o departamento responsável pela implementação.


## Histórico de Versões

### v2.0 (2025-10-24)
- Versão otimizada para Linux
- Suporte a múltiplas câmeras via RTSP
- Threading para melhor performance
- Cosine learning rate no treinamento
- Documentação completa

### v1.0 (2025-10-01)
- Versão inicial para Windows
- Detecção básica em tempo real
- Modelo YOLOv8n

## Roadmap

### Próximas Funcionalidades

- Dashboard web para visualização de métricas
- Integração com banco de dados
- Alertas via email/SMS
- API REST para integração
- Suporte a múltiplas GPUs
- Reconhecimento facial (opcional)
- Análise de padrões temporais
- Relatórios automáticos PDF

---

**Última atualização**: Outubro 2025  
**Versão do documento**: 2.0  
**Autor**: Jhonatan Frossard