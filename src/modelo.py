"""Carregamento do tokenizador e do GPT-2."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import config


def dispositivo():
    return "cuda" if torch.cuda.is_available() else "cpu"


def carregar_tokenizer(nome=config.MODELO_BASE):
    tok = AutoTokenizer.from_pretrained(nome)
    if tok.pad_token is None:  # GPT-2 não tem token de padding; reutilizamos o EOS
        tok.pad_token = tok.eos_token
    return tok


def carregar_modelo(nome=config.MODELO_BASE, tokenizer=None):
    modelo = AutoModelForCausalLM.from_pretrained(nome)
    if tokenizer is not None:
        modelo.config.pad_token_id = tokenizer.pad_token_id
    return modelo.to(dispositivo())


def resumo_modelo(modelo):
    c = modelo.config
    n_params = sum(p.numel() for p in modelo.parameters())
    print("=" * 60)
    print(f"Parâmetros totais     : {n_params / 1e6:.1f} M")
    print(f"Camadas (blocos)      : {getattr(c, 'n_layer', '?')}")
    print(f"Cabeças de atenção    : {getattr(c, 'n_head', '?')}")
    print(f"Dimensão (d_model)    : {getattr(c, 'n_embd', '?')}")
    print(f"Vocabulário           : {c.vocab_size}")
    print(f"Contexto máximo       : {getattr(c, 'n_positions', '?')} tokens")
    print(f"Dispositivo           : {next(modelo.parameters()).device}")
    print("=" * 60)
