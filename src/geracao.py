"""Geração de texto e comparação lado a lado entre o modelo base e o ajustado."""
import torch

from src import config

# Prompts dentro do domínio "carros": usados também nas métricas de vocabulário/diversidade.
PROMPTS_DOMINIO = [
    "O motor de combustão interna funciona",
    "Para trocar o óleo do carro, o primeiro passo é",
    "Os carros elétricos têm como principal vantagem",
    "A diferença entre câmbio manual e automático é",
    "Os freios ABS servem para",
]

# Prompts "armadilha": fatos precisos, assunto fora do tema e a piada do Transformers.
PROMPTS_ARMADILHA = [
    "O Ferrari 250 GTO foi fabricado em",          # fato numérico -> alucinação
    "Optimus Prime é um caminhão",                 # trocadilho Transformers x carros
    "A receita de bolo de chocolate começa com",   # fora do domínio -> esquecimento
    "Na Fórmula 1, o campeão mundial de 1994 foi", # fato histórico -> alucinação
]

PROMPTS_TESTE = PROMPTS_DOMINIO + PROMPTS_ARMADILHA


def gerar(modelo, tokenizer, prompt, max_new_tokens=config.MAX_NEW_TOKENS,
          temperature=config.TEMPERATURA, top_k=config.TOP_K, top_p=config.TOP_P,
          do_sample=True, repetition_penalty=config.REPETITION_PENALTY,
          num_return_sequences=1, seed=None, incluir_prompt=False):
    """Gera continuações para `prompt`. Retorna uma lista de strings."""
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    modelo.eval()
    entrada = tokenizer(prompt, return_tensors="pt").to(modelo.device)
    kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id,
              repetition_penalty=repetition_penalty, num_return_sequences=num_return_sequences)
    if do_sample:
        kw.update(do_sample=True, temperature=max(float(temperature), 1e-5),
                  top_k=int(top_k) if top_k else 0, top_p=float(top_p))
    else:
        kw.update(do_sample=False)
    with torch.no_grad():
        saida = modelo.generate(**entrada, **kw)
    n_prompt = entrada["input_ids"].shape[1]
    return [tokenizer.decode(seq if incluir_prompt else seq[n_prompt:],
                             skip_special_tokens=True).strip() for seq in saida]


def comparar(modelo_base, modelo_ajustado, tokenizer, prompts=None, seed=config.SEED, **kwargs):
    """Gera com os dois modelos (mesma seed) e imprime lado a lado."""
    resultados = []
    for p in (prompts or PROMPTS_TESTE):
        base = gerar(modelo_base, tokenizer, p, seed=seed, **kwargs)[0]
        ajustado = gerar(modelo_ajustado, tokenizer, p, seed=seed, **kwargs)[0]
        resultados.append({"prompt": p, "base": base, "ajustado": ajustado})
        print("=" * 78)
        print(f"PROMPT   : {p}")
        print(f"BASE     : {base}")
        print(f"AJUSTADO : {ajustado}")
    return resultados
