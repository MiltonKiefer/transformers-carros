"""Coleta de artigos da Wikipédia em português sobre automóveis.

Estratégia: reúne títulos candidatos (lista semente + categorias + buscas),
baixa o texto puro de cada artigo, remove seções finais inúteis
(referências, links externos...) e salva em JSONL.
"""
import json
import re
import time

import requests

from src import config

_REGEX_TITULO = re.compile(r"^(={2,})\s*(.+?)\s*\1\s*$")


def _sessao():
    s = requests.Session()
    s.headers.update({"User-Agent": config.WIKI_USER_AGENT})
    return s


def _chamar_api(sessao, params):
    params = {**params, "format": "json", "formatversion": 2}
    for tentativa in range(3):
        try:
            r = sessao.get(config.WIKI_API, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError):
            time.sleep(2 ** tentativa)
    return None


def titulos_da_categoria(sessao, categoria, profundidade=1):
    """Lista páginas de uma categoria (e de suas subcategorias, 1 nível)."""
    titulos = []
    tipos = "page|subcat" if profundidade > 0 else "page"
    params = {"action": "query", "list": "categorymembers", "cmtitle": categoria,
              "cmlimit": 100, "cmtype": tipos}
    while True:
        dados = _chamar_api(sessao, params)
        if not dados or "query" not in dados:
            break
        for item in dados["query"]["categorymembers"]:
            if item["ns"] == 0:
                titulos.append(item["title"])
            elif item["ns"] == 14 and profundidade > 0:
                titulos += titulos_da_categoria(sessao, item["title"], profundidade - 1)
        if "continue" in dados and len(titulos) < 400:
            params["cmcontinue"] = dados["continue"]["cmcontinue"]
        else:
            break
        time.sleep(config.PAUSA_ENTRE_REQUISICOES)
    return titulos


def titulos_da_busca(sessao, termo, limite=50):
    dados = _chamar_api(sessao, {"action": "query", "list": "search", "srsearch": termo,
                                 "srlimit": limite, "srnamespace": 0})
    if not dados or "query" not in dados:
        return []
    return [r["title"] for r in dados["query"]["search"]]


def reunir_titulos(sessao):
    candidatos = list(config.TITULOS_SEMENTE)
    for cat in config.CATEGORIAS_WIKI:
        candidatos += titulos_da_categoria(sessao, cat)
    for termo in config.BUSCAS_WIKI:
        candidatos += titulos_da_busca(sessao, termo)
    vistos, unicos = set(), []
    for t in candidatos:  # remove duplicatas preservando a ordem
        if t not in vistos:
            vistos.add(t)
            unicos.append(t)
    return unicos


def baixar_artigo(sessao, titulo):
    """Retorna (titulo_final, texto) ou None se a página não existir."""
    dados = _chamar_api(sessao, {"action": "query", "prop": "extracts", "explaintext": 1,
                                 "exsectionformat": "wiki", "titles": titulo, "redirects": 1})
    if not dados or "query" not in dados:
        return None
    pagina = dados["query"]["pages"][0]
    if pagina.get("missing") or not pagina.get("extract"):
        return None
    return pagina["title"], pagina["extract"]


def limpar_artigo_wikipedia(texto):
    """Corta seções finais inúteis e transforma cabeçalhos '== X ==' em linhas simples."""
    linhas_saida = []
    for linha in texto.splitlines():
        m = _REGEX_TITULO.match(linha.strip())
        if m:
            if m.group(2).strip().lower() in config.SECOES_IGNORADAS:
                break
            linhas_saida.append(m.group(2).strip())
        else:
            linhas_saida.append(linha.rstrip())
    texto = "\n".join(linhas_saida)
    return re.sub(r"\n{3,}", "\n\n", texto).strip()


def coletar(max_artigos=config.MAX_ARTIGOS, forcar=False):
    """Coleta os artigos e salva em data/. Reaproveita o arquivo se já existir."""
    if config.ARQ_ARTIGOS.exists() and not forcar:
        print(f"[coleta] {config.ARQ_ARTIGOS.name} já existe; reutilizando (use forcar=True para refazer).")
        return ler_artigos()

    config.DIR_DADOS.mkdir(parents=True, exist_ok=True)
    sessao = _sessao()
    print("[coleta] reunindo títulos candidatos...")
    titulos = reunir_titulos(sessao)
    print(f"[coleta] {len(titulos)} candidatos. Baixando até {max_artigos} artigos...")

    artigos = []
    for i, titulo in enumerate(titulos, 1):
        if len(artigos) >= max_artigos:
            break
        resultado = baixar_artigo(sessao, titulo)
        time.sleep(config.PAUSA_ENTRE_REQUISICOES)
        if resultado is None:
            continue
        titulo_final, bruto = resultado
        texto = limpar_artigo_wikipedia(bruto)
        if len(texto) < config.MIN_CARACTERES_ARTIGO:
            continue
        if any(a["titulo"] == titulo_final for a in artigos):
            continue
        artigos.append({"titulo": titulo_final, "texto": texto})
        if len(artigos) % 25 == 0:
            print(f"[coleta]   {len(artigos)} artigos válidos ({i}/{len(titulos)} candidatos testados)")

    if not artigos:
        raise RuntimeError("Nenhum artigo coletado. Verifique a conexão ou use outra FONTE_DADOS.")

    with open(config.ARQ_ARTIGOS, "w", encoding="utf-8") as f:
        for a in artigos:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    with open(config.ARQ_CORPUS, "w", encoding="utf-8") as f:
        f.write("\n\n".join(f"{a['titulo']}\n\n{a['texto']}" for a in artigos))

    imprimir_estatisticas(artigos)
    return artigos


def ler_artigos():
    with open(config.ARQ_ARTIGOS, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def imprimir_estatisticas(artigos):
    n_chars = sum(len(a["texto"]) for a in artigos)
    n_palavras = sum(len(a["texto"].split()) for a in artigos)
    print(f"[coleta] artigos: {len(artigos)} | caracteres: {n_chars:,} | palavras: {n_palavras:,}")


if __name__ == "__main__":
    coletar()
