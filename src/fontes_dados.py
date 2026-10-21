"""Carrega documentos de diferentes fontes e devolve sempre o mesmo formato:

    [{"titulo": str, "texto": str}, ...]

Assim o restante do pipeline não precisa saber de onde os textos vieram.
Para trocar a fonte, basta alterar `FONTE_DADOS` em config.py.
"""
import csv
import sys

from src import config


def _quebrar_em_pseudo_documentos(texto, tamanho):
    """Divide um texto grande em pedaços (por parágrafo) para permitir o split treino/validação."""
    docs, atual = [], ""
    for par in texto.split("\n\n"):
        atual += par + "\n\n"
        if len(atual) >= tamanho:
            docs.append(atual.strip())
            atual = ""
    if atual.strip():
        docs.append(atual.strip())
    return docs


def _de_wikipedia():
    from src import coleta_dados
    if config.ARQ_ARTIGOS.exists():
        return coleta_dados.ler_artigos()
    return coleta_dados.coletar()


def _de_pasta_txt():
    arquivos = sorted(config.PASTA_TXT.glob("*.txt"))
    if not arquivos:
        raise FileNotFoundError(f"Coloque arquivos .txt em {config.PASTA_TXT}")
    docs = []
    for arq in arquivos:
        texto = arq.read_text(encoding="utf-8", errors="ignore")
        docs.append({"titulo": arq.stem, "texto": texto})
    if len(docs) == 1:  # um único arquivo: quebra em pedaços
        pedacos = _quebrar_em_pseudo_documentos(docs[0]["texto"], config.TAMANHO_PSEUDO_DOC)
        docs = [{"titulo": f"{docs[0]['titulo']}_{i}", "texto": p} for i, p in enumerate(pedacos)]
    return docs


def _de_csv():
    if not config.ARQ_CSV.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {config.ARQ_CSV}")
    csv.field_size_limit(sys.maxsize)
    docs = []
    with open(config.ARQ_CSV, encoding="utf-8", newline="") as f:
        for i, linha in enumerate(csv.DictReader(f)):
            texto = (linha.get(config.COLUNA_TEXTO_CSV) or "").strip()
            if texto:
                docs.append({"titulo": f"linha_{i}", "texto": texto})
    return docs


def _de_huggingface():
    from datasets import load_dataset
    if not config.HF_DATASET:
        raise ValueError("Defina HF_DATASET em config.py")
    ds = load_dataset(config.HF_DATASET, config.HF_CONFIG, split=config.HF_SPLIT, streaming=True)
    docs = []
    chaves = tuple(k.lower() for k in config.HF_PALAVRAS_CHAVE)
    for i, exemplo in enumerate(ds):
        texto = (exemplo.get(config.HF_COLUNA_TEXTO) or "").strip()
        # filtro por palavras-chave: mantém apenas textos do tema "carros"
        if texto and any(k in texto.lower() for k in chaves):
            docs.append({"titulo": f"hf_{i}", "texto": texto})
        if len(docs) >= config.HF_MAX_DOCUMENTOS:
            break
    return docs


_FONTES = {
    "wikipedia": _de_wikipedia,
    "pasta_txt": _de_pasta_txt,
    "csv": _de_csv,
    "huggingface": _de_huggingface,
}


def carregar_documentos(fonte=None):
    fonte = fonte or config.FONTE_DADOS
    if fonte not in _FONTES:
        raise ValueError(f"Fonte desconhecida: {fonte}. Opções: {list(_FONTES)}")
    docs = _FONTES[fonte]()
    print(f"[dados] fonte='{fonte}' | {len(docs)} documentos carregados")
    return docs
