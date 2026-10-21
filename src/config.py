"""Configuração central do projeto.

Todos os hiperparâmetros, caminhos e listas de temas ficam aqui,
para que o resto do código não tenha "números mágicos".
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Caminhos
# ----------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
DIR_DADOS = RAIZ / "data"
DIR_SAIDA = RAIZ / "outputs"
DIR_CHECKPOINTS = DIR_SAIDA / "checkpoints"
DIR_MODELO = DIR_SAIDA / "modelo_carros"
DIR_GRAFICOS = DIR_SAIDA / "graficos"
ARQ_ARTIGOS = DIR_DADOS / "artigos_carros.jsonl"   # 1 artigo por linha (titulo, texto)
ARQ_CORPUS = DIR_DADOS / "corpus_carros.txt"       # todos os artigos concatenados
ARQ_SPLIT = DIR_DADOS / "split_treino_validacao.json"
ARQ_METRICAS = DIR_SAIDA / "metricas.json"

SEED = 42

# ----------------------------------------------------------------------------
# Modelo
# ----------------------------------------------------------------------------
MODELO_BASE = "pierreguillou/gpt2-small-portuguese"  # ~124M parâmetros
MODELO_ALTERNATIVO = "distilgpt2"                    # plano B (inglês, ~82M)

# ----------------------------------------------------------------------------
# Fonte de dados
#   "wikipedia"   -> coleta pela API da Wikipédia (padrão)
#   "pasta_txt"   -> arquivos .txt em data/custom_txt/
#   "csv"         -> arquivo data/custom.csv com uma coluna de texto
#   "huggingface" -> dataset do Hugging Face Hub (veja HF_* abaixo)
# ----------------------------------------------------------------------------
FONTE_DADOS = "wikipedia"

# --- Wikipédia ---
WIKI_API = "https://pt.wikipedia.org/w/api.php"
WIKI_USER_AGENT = "TransformersCarros/1.0 (trabalho academico; fine-tuning GPT-2)"
MAX_ARTIGOS = 250
MIN_CARACTERES_ARTIGO = 1500
PAUSA_ENTRE_REQUISICOES = 0.2  # segundos (boa educação com a API)

TITULOS_SEMENTE = [
    "Automóvel", "Motor de combustão interna", "Carro elétrico", "Veículo híbrido",
    "Caixa de câmbio", "Transmissão automática", "Volkswagen Fusca", "Ferrari",
    "Fórmula 1", "Ford Model T", "Tesla, Inc.", "Toyota", "Volkswagen", "Chevrolet",
    "Fiat", "Porsche", "Lamborghini", "BMW", "Mercedes-Benz", "Honda", "Hyundai",
    "Renault", "Peugeot", "Volvo", "Freio", "Airbag", "Turbocompressor",
    "Pneu", "Suspensão", "Automobilismo", "Rali Dakar", "Carro esportivo",
    "Utilitário esportivo", "Picape", "Motor a diesel", "Injeção eletrônica",
]
CATEGORIAS_WIKI = [
    "Categoria:Automóveis",
    "Categoria:Tecnologias automotivas",
    "Categoria:Fabricantes de automóveis",
]
BUSCAS_WIKI = [
    "automóvel", "motor de carro", "carro esportivo", "montadora de automóveis",
    "carro elétrico", "modelo de automóvel", "SUV", "picape", "câmbio automático",
    "segurança veicular", "indústria automobilística no Brasil",
]
SECOES_IGNORADAS = {
    "referências", "ligações externas", "ver também", "notas", "bibliografia",
    "links externos", "galeria", "leitura adicional", "fontes", "ligações",
}

# --- Fontes alternativas ---
PASTA_TXT = DIR_DADOS / "custom_txt"
ARQ_CSV = DIR_DADOS / "custom.csv"
COLUNA_TEXTO_CSV = "texto"
TAMANHO_PSEUDO_DOC = 3000          # usado quando há 1 só arquivo grande
HF_DATASET = None                  # ex.: "usuario/nome-do-dataset"
HF_CONFIG = None
HF_SPLIT = "train"
HF_COLUNA_TEXTO = "text"
HF_PALAVRAS_CHAVE = ["carro", "automóvel", "veículo", "motor", "montadora"]
HF_MAX_DOCUMENTOS = 500

# ----------------------------------------------------------------------------
# Pré-processamento
# ----------------------------------------------------------------------------
BLOCK_SIZE = 256
VAL_FRAC = 0.10

# ----------------------------------------------------------------------------
# Treino
# ----------------------------------------------------------------------------
EPOCAS = 3
LEARNING_RATE = 5e-5
BATCH_SIZE = 8
GRAD_ACCUM = 2
WEIGHT_DECAY = 0.01
WARMUP_FRAC = 0.10
EVAL_STEPS = 25
PACIENCIA = 3        # nº de avaliações sem melhora antes do early stopping

# ----------------------------------------------------------------------------
# Geração
# ----------------------------------------------------------------------------
TEMPERATURA = 0.8
TOP_K = 50
TOP_P = 0.95
MAX_NEW_TOKENS = 80
REPETITION_PENALTY = 1.0   # 1.0 = sem penalidade (deixa a repetição aparecer)
