"""Limpeza, divisão treino/validação (por documento) e tokenização em blocos."""
import json
import random
import re

from datasets import Dataset, DatasetDict

from src import config


def limpar_texto(texto):
    texto = texto.replace("\u00a0", " ").replace("\t", " ")
    texto = re.sub(r"[ ]{2,}", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def dividir_por_documento(documentos, val_frac=config.VAL_FRAC, seed=config.SEED):
    """Divide ANTES de tokenizar, por documento, para evitar vazamento entre treino e validação."""
    docs = list(documentos)
    random.Random(seed).shuffle(docs)
    n_val = max(1, round(len(docs) * val_frac))
    return docs[n_val:], docs[:n_val]


def agrupar_em_blocos(textos, tokenizer, block_size=config.BLOCK_SIZE):
    """Concatena todos os textos (separados por EOS) e fatia em blocos de tamanho fixo."""
    eos = tokenizer.eos_token_id
    ids_todos = []
    limite_antigo = tokenizer.model_max_length
    tokenizer.model_max_length = int(1e9)  # evita o aviso de "sequência maior que o máximo"
    try:
        for t in textos:
            ids_todos.extend(tokenizer(t, add_special_tokens=False)["input_ids"])
            ids_todos.append(eos)
    finally:
        tokenizer.model_max_length = limite_antigo
    n = (len(ids_todos) // block_size) * block_size
    blocos = [ids_todos[i:i + block_size] for i in range(0, n, block_size)]
    # Em LM causal, labels == input_ids (o deslocamento de 1 posição é feito dentro do modelo).
    return Dataset.from_dict({"input_ids": blocos, "labels": [list(b) for b in blocos]})


def preparar_datasets(documentos, tokenizer, block_size=config.BLOCK_SIZE):
    docs = [{"titulo": d["titulo"], "texto": limpar_texto(d["texto"])} for d in documentos]
    docs = [d for d in docs if d["texto"]]
    treino, validacao = dividir_por_documento(docs)

    config.DIR_DADOS.mkdir(parents=True, exist_ok=True)
    with open(config.ARQ_SPLIT, "w", encoding="utf-8") as f:
        json.dump({"treino": [d["titulo"] for d in treino],
                   "validacao": [d["titulo"] for d in validacao]}, f, ensure_ascii=False, indent=2)

    ds = DatasetDict({
        "train": agrupar_em_blocos([d["texto"] for d in treino], tokenizer, block_size),
        "validation": agrupar_em_blocos([d["texto"] for d in validacao], tokenizer, block_size),
    })
    print(f"[preprocessamento] documentos: treino={len(treino)} | validação={len(validacao)}")
    print(f"[preprocessamento] blocos de {block_size} tokens: "
          f"treino={len(ds['train'])} | validação={len(ds['validation'])}")
    if len(ds["validation"]) == 0:
        raise RuntimeError("Validação vazia: o corpus é pequeno demais ou BLOCK_SIZE é grande demais.")
    return ds
