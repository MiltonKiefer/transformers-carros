"""Métricas: perplexidade (domínio e fora do domínio), vocabulário automotivo e diversidade."""
import json
import math
import re

import matplotlib.pyplot as plt
import torch

from src import config
from src.geracao import PROMPTS_DOMINIO, gerar

# Textos fora do domínio (escritos para este projeto) para medir o "esquecimento".
TEXTOS_FORA_DOMINIO = [
    "Para preparar um bom arroz, comece lavando os grãos até que a água fique clara. "
    "Em uma panela, refogue o alho picado em um fio de azeite, acrescente o arroz e mexa por "
    "alguns instantes. Cubra com água quente, tampe a panela e cozinhe em fogo baixo até a "
    "água secar. Depois, deixe descansar por cinco minutos antes de servir com feijão e salada.",
    "O futebol é o esporte mais popular do Brasil. Todos os fins de semana, milhares de "
    "torcedores lotam os estádios para acompanhar os jogos do campeonato nacional. Os times "
    "treinam durante a semana, definem estratégias e tentam surpreender os adversários. "
    "Quando a bola rola, a emoção toma conta das arquibancadas e das casas de todo o país.",
    "A fotossíntese é o processo pelo qual as plantas transformam a luz do sol em energia. "
    "Nas folhas, a clorofila absorve a luz e utiliza a água e o gás carbônico para produzir "
    "açúcares e liberar oxigênio. Esse processo é essencial para a vida na Terra, pois sustenta "
    "as cadeias alimentares e mantém o equilíbrio da atmosfera.",
]

# Termos exatos e raízes de palavras do universo automotivo (medida aproximada).
TERMOS_EXATOS = {"carro", "carros", "roda", "rodas", "freio", "freios", "pneu", "pneus",
                 "marcha", "marchas", "óleo", "volante", "motor", "motores", "piloto",
                 "pilotos", "pista", "pistas", "cilindro", "cilindros", "carroceria"}
RAIZES = ("automóvel", "automóve", "automotiv", "automobil", "veículo", "veiculo", "câmbio",
          "cambio", "combustí", "combusti", "gasolina", "diesel", "montadora", "turbo",
          "transmiss", "suspens", "potência", "cavalos", "hatch", "sedan", "caminhão",
          "caminhõe", "elétric", "eletric", "bateria", "aceleraç", "velocidade")


def _palavras(texto):
    return re.findall(r"\w+", texto.lower())


@torch.no_grad()
def perplexidade_dataset(modelo, dataset, batch_size=8):
    """Perplexidade = exp(loss média por token) sobre um dataset de blocos."""
    modelo.eval()
    total_loss, total_tokens = 0.0, 0
    for i in range(0, len(dataset), batch_size):
        ids = torch.tensor(dataset[i:i + batch_size]["input_ids"]).to(modelo.device)
        loss = modelo(input_ids=ids, labels=ids).loss.item()
        n = ids.shape[0] * (ids.shape[1] - 1)
        total_loss += loss * n
        total_tokens += n
    return math.exp(total_loss / total_tokens)


@torch.no_grad()
def perplexidade_textos(modelo, tokenizer, textos, max_len=config.BLOCK_SIZE):
    modelo.eval()
    total_loss, total_tokens = 0.0, 0
    for texto in textos:
        ids = tokenizer(texto, return_tensors="pt")["input_ids"][0]
        for i in range(0, len(ids), max_len):
            bloco = ids[i:i + max_len]
            if len(bloco) < 2:
                continue
            bloco = bloco.unsqueeze(0).to(modelo.device)
            loss = modelo(input_ids=bloco, labels=bloco).loss.item()
            total_loss += loss * (bloco.shape[1] - 1)
            total_tokens += bloco.shape[1] - 1
    return math.exp(total_loss / total_tokens)


def taxa_vocabulario_automotivo(textos):
    """Fração das palavras geradas que pertencem ao vocabulário automotivo."""
    total = acertos = 0
    for t in textos:
        for p in _palavras(t):
            total += 1
            if p in TERMOS_EXATOS or p.startswith(RAIZES):
                acertos += 1
    return acertos / total if total else 0.0


def distinct_n(textos, n=2):
    """Razão entre n-gramas únicos e totais (maior = mais diverso)."""
    total, unicos = 0, set()
    for t in textos:
        p = _palavras(t)
        grams = [tuple(p[i:i + n]) for i in range(len(p) - n + 1)]
        total += len(grams)
        unicos.update(grams)
    return len(unicos) / total if total else 0.0


def taxa_repeticao(textos, n=4):
    """Fração média de n-gramas repetidos dentro de cada texto (maior = mais repetitivo)."""
    taxas = []
    for t in textos:
        p = _palavras(t)
        grams = [tuple(p[i:i + n]) for i in range(len(p) - n + 1)]
        if grams:
            taxas.append(1 - len(set(grams)) / len(grams))
    return sum(taxas) / len(taxas) if taxas else 0.0


def amostrar(modelo, tokenizer, prompts, n_por_prompt=3, seed=config.SEED):
    textos = []
    for p in prompts:
        textos += gerar(modelo, tokenizer, p, num_return_sequences=n_por_prompt, seed=seed)
    return textos


def _metricas_geracao(modelo, tokenizer):
    textos = amostrar(modelo, tokenizer, PROMPTS_DOMINIO)
    return {
        "vocabulario_automotivo": taxa_vocabulario_automotivo(textos),
        "distinct_1": distinct_n(textos, 1),
        "distinct_2": distinct_n(textos, 2),
        "taxa_repeticao_4gram": taxa_repeticao(textos, 4),
        "n_amostras": len(textos),
    }


def plotar_perplexidades(metricas, caminho=None):
    rotulos = ["Validação\n(carros)", "Fora do domínio\n(culinária etc.)"]
    base = [metricas["base"]["perplexidade_validacao"], metricas["base"]["perplexidade_fora_dominio"]]
    aj = [metricas["ajustado"]["perplexidade_validacao"], metricas["ajustado"]["perplexidade_fora_dominio"]]
    x = range(len(rotulos))
    plt.figure(figsize=(6.5, 4))
    plt.bar([i - 0.2 for i in x], base, width=0.4, label="GPT-2 base")
    plt.bar([i + 0.2 for i in x], aj, width=0.4, label="GPT-2 ajustado (carros)")
    plt.xticks(list(x), rotulos)
    plt.ylabel("Perplexidade (menor = melhor)")
    plt.title("Perplexidade: base × ajustado")
    plt.legend()
    plt.tight_layout()
    caminho = caminho or config.DIR_GRAFICOS / "perplexidade.png"
    config.DIR_GRAFICOS.mkdir(parents=True, exist_ok=True)
    plt.savefig(caminho, dpi=150)
    plt.close()
    return caminho


def avaliar_tudo(modelo_base, modelo_ajustado, tokenizer, datasets):
    metricas = {}
    for nome, modelo in (("base", modelo_base), ("ajustado", modelo_ajustado)):
        print(f"[avaliacao] avaliando modelo '{nome}'...")
        m = {
            "perplexidade_validacao": perplexidade_dataset(modelo, datasets["validation"]),
            "perplexidade_fora_dominio": perplexidade_textos(modelo, tokenizer, TEXTOS_FORA_DOMINIO),
        }
        m.update(_metricas_geracao(modelo, tokenizer))
        metricas[nome] = m

    config.DIR_SAIDA.mkdir(parents=True, exist_ok=True)
    with open(config.ARQ_METRICAS, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)
    plotar_perplexidades(metricas)
    imprimir_tabela(metricas)
    return metricas


def imprimir_tabela(metricas):
    linhas = [("Perplexidade (validação, carros)", "perplexidade_validacao", "{:.2f}"),
              ("Perplexidade (fora do domínio)", "perplexidade_fora_dominio", "{:.2f}"),
              ("Vocabulário automotivo gerado", "vocabulario_automotivo", "{:.1%}"),
              ("Distinct-1", "distinct_1", "{:.3f}"),
              ("Distinct-2", "distinct_2", "{:.3f}"),
              ("Repetição (4-gramas)", "taxa_repeticao_4gram", "{:.1%}")]
    print(f"\n{'Métrica':<36}{'Base':>12}{'Ajustado':>12}")
    print("-" * 60)
    for rotulo, chave, fmt in linhas:
        print(f"{rotulo:<36}{fmt.format(metricas['base'][chave]):>12}"
              f"{fmt.format(metricas['ajustado'][chave]):>12}")
