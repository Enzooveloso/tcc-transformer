"""Gera as tabelas do capítulo de Resultados a partir dos CSVs de ``resultados/``.

Cada tabela é escrita em ``escrita/06_capitulo/tabelas/<nome>.tex`` contendo
apenas o ambiente ``tabular``; legenda, rótulo e posição ficam no
``resultados.tex``, que inclui cada arquivo com ``\\input``. Assim nenhum número
das tabelas é digitado à mão, e rodar o script de novo reproduz todas elas.

    python gerar_tabelas.py

Convenções (ver Seção 4.1.3 da monografia):
- energia sempre relativa ao modelo denso da mesma série, medido na mesma
  sessão (``esparsidade_alvo == 0``), nunca em valor absoluto;
- resumo por configuração: média geométrica da perplexity e média aritmética
  de ΔE nas taxas de 10% a 50% (``FAIXA_RESUMO``); η é calculado a partir dessas
  duas médias;
- ponto de eficiência S*_E: maior taxa com perplexity até 5% acima da do
  modelo de referência no mesmo regime (``LIMIAR_SE``).

Só usa a biblioteca padrão: roda sem PyTorch.
"""

import csv
import math
from pathlib import Path
from statistics import mean

RAIZ = Path(__file__).resolve().parent
CSV = RAIZ / "resultados"
SAIDA = RAIZ.parent / "escrita" / "06_capitulo" / "tabelas"

TAXAS = list(range(0, 100, 10))   # em %
FAIXA_RESUMO = range(10, 60, 10)  # 10% a 50%: sem camada no teto (Seção 6.1)
LIMIAR_SE = 0.05


def ler(nome):
    with open(CSV / nome, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def serie(linhas, **filtro):
    """Linhas de uma configuração, indexadas pela taxa de poda em %."""
    out = {round(float(r["esparsidade_alvo"]) * 100): r for r in linhas
           if all(r[k] == v for k, v in filtro.items())}
    assert sorted(out) == TAXAS, filtro
    return out


def br(x, casas=2):
    """Número no padrão do texto: vírgula decimal, sem separador de milhar."""
    return f"{x:.{casas}f}".replace(".", ",")


def pct(x):
    return f"{x}\\,\\%"


def escrever(nome, linhas):
    SAIDA.mkdir(parents=True, exist_ok=True)
    (SAIDA / f"{nome}.tex").write_text(
        f"% Gerado por codigo/gerar_tabelas.py a partir de codigo/resultados/. Não editar.\n"
        + "\n".join(linhas) + "\n", encoding="utf-8")


mag, est, sen = ler("magnitude.csv"), ler("estruturada.csv"), ler("sensibilidade.csv")
ft = ler("finetune.csv")
perfil50, perfil30 = ler("sensibilidade_perfil.csv"), ler("sensibilidade_perfil_sonda03.csv")

CONFIGS = {  # (estratégia, alocação) -> série
    ("mag", "global"): serie(mag, escopo="global"),
    ("mag", "uniforme"): serie(mag, escopo="uniforme"),
    ("mag", "b1"): serie(sen, estrategia="magnitude", beta="1.0"),
    ("mag", "b05"): serie(sen, estrategia="magnitude", beta="0.5"),
    ("est", "global"): serie(est, escopo="global", alvo="ambos"),
    ("est", "uniforme"): serie(est, escopo="uniforme", alvo="ambos"),
    ("est", "b1"): serie(sen, estrategia="estruturada", beta="1.0"),
    ("est", "b05"): serie(sen, estrategia="estruturada", beta="0.5"),
}
ALOCACOES = ("global", "uniforme", "b1", "b05")
PPL0 = float(CONFIGS[("mag", "global")][0]["perplexity"])

CAB_ALOCACOES = [
    r"& \multicolumn{2}{c}{\textbf{Sem análise de sensibilidade}}"
    r" & \multicolumn{2}{c}{\textbf{Com análise de sensibilidade}} \\",
    r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
    r"\textbf{Taxa de poda} & \textbf{Global} & \textbf{Uniforme}"
    r" & $\boldsymbol{\beta = 1{,}0}$ & $\boldsymbol{\beta = 0{,}5}$ \\",
]


def ppl(r):
    return float(r["perplexity"])


def delta_e(s, taxa):
    """Redução relativa de energia (%) frente ao denso da mesma série."""
    return (1 - float(s[taxa]["energia_kwh"]) / float(s[0]["energia_kwh"])) * 100


# ---------------------------------------------------------------- qualidade
def tabela_qualidade(estrategia):
    linhas = [r"\begin{tabular}{c cc cc}", r"\toprule", *CAB_ALOCACOES, r"\midrule",
              rf"0\,\% & \multicolumn{{4}}{{c}}{{{br(PPL0)} (modelo base)}} \\"]
    for t in TAXAS[1:]:
        vals = [ppl(CONFIGS[(estrategia, a)][t]) for a in ALOCACOES]
        menor = min(vals)
        cels = [rf"\textbf{{{br(v)}}}" if v == menor else br(v) for v in vals]
        linhas.append(f"{pct(t)} & " + " & ".join(cels) + r" \\")
    linhas += [r"\bottomrule", r"\end{tabular}"]
    escrever(f"qualidade_{'magnitude' if estrategia == 'mag' else 'estruturada'}", linhas)


tabela_qualidade("mag")
tabela_qualidade("est")

# ---------------------------------------------------------------- custo teórico
cm, ce = CONFIGS[("mag", "uniforme")], CONFIGS[("est", "global")]
linhas = [r"\begin{tabular}{c cc cc cc}", r"\toprule",
          r"& \multicolumn{2}{c}{\textbf{Parâmetros (M)}} & \multicolumn{2}{c}{\textbf{Tamanho (MB)}}"
          r" & \multicolumn{2}{c}{\textbf{FLOPs (G)}} \\",
          r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
          r"\textbf{Taxa de poda}" + r" & \textbf{Magn.} & \textbf{Estrut.}" * 3 + r" \\", r"\midrule"]
for t in TAXAS:
    a, b = cm[t], ce[t]
    linhas.append(f"{pct(t)} & {br(int(a['parametros_nao_zerados']) / 1e6, 1)}"
                  f" & {br(int(b['parametros_total']) / 1e6, 1)}"
                  f" & {br(float(a['tamanho_mb']), 1)} & {br(float(b['tamanho_mb']), 1)}"
                  f" & {br(float(a['flops']) / 1e9, 1)} & {br(float(b['flops']) / 1e9, 1)} \\\\")
linhas += [r"\bottomrule", r"\end{tabular}"]
escrever("custo", linhas)

# ---------------------------------------------------------------- energia (ΔE)
ROTULO = {"global": "Global", "uniforme": "Uniforme",
          "b1": r"$\beta = 1{,}0$", "b05": r"$\beta = 0{,}5$"}
linhas = [r"\begin{tabular}{p{2.8cm} l " + "r" * 9 + "}", r"\toprule",
          r"& & \multicolumn{9}{c}{\textbf{Taxa de poda}} \\", r"\cmidrule(lr){3-11}",
          r"\textbf{Análise} & \textbf{Alocação} & " + " & ".join(pct(t) for t in TAXAS[1:]) + r" \\"]
for estrategia, nome in (("mag", "Poda não estruturada --- poda por magnitude"),
                         ("est", "Poda estruturada --- cabeças de atenção e neurônios do MLP")):
    linhas += [r"\midrule", rf"\multicolumn{{11}}{{l}}{{\textit{{{nome}}}}} \\"]
    for i, a in enumerate(ALOCACOES):
        grupo = ""
        if i == 0:
            grupo = r"\multirow{2}{2.8cm}{Sem análise de sensibilidade}"
        elif i == 2:
            grupo = r"\multirow{2}{2.8cm}{Com análise de sensibilidade}"
        s = CONFIGS[(estrategia, a)]
        linhas.append(f"{grupo} & {ROTULO[a]} & "
                      + " & ".join(br(delta_e(s, t), 1) for t in TAXAS[1:]) + r" \\")
linhas += [r"\bottomrule", r"\end{tabular}"]
escrever("energia", linhas)

# ---------------------------------------------------------------- perfil (duas sondas)
def deltas(perfil, estrategia):
    rows = sorted((r for r in perfil if r["estrategia"] == estrategia), key=lambda r: int(r["camada"]))
    return [float(r["delta_log_ppl"]) for r in rows]


def postos(d):
    ordem = sorted(range(len(d)), key=lambda i: -d[i])
    return [ordem.index(i) + 1 for i in range(len(d))]


colunas = [deltas(p, e) for e in ("magnitude", "estruturada") for p in (perfil30, perfil50)]
linhas = [r"\begin{tabular}{c cc cc}", r"\toprule",
          r"& \multicolumn{2}{c}{\textbf{Poda por magnitude}} & \multicolumn{2}{c}{\textbf{Poda estruturada}} \\",
          r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
          r"\textbf{Camada} & \textbf{Sonda 30\,\%} & \textbf{Sonda 50\,\%}"
          r" & \textbf{Sonda 30\,\%} & \textbf{Sonda 50\,\%} \\", r"\midrule"]
pcol = [postos(c) for c in colunas]
for camada in range(12):
    linhas.append(f"{camada} & " + " & ".join(f"{br(c[camada], 3)} ({p[camada]})"
                                              for c, p in zip(colunas, pcol)) + r" \\")
linhas += [r"\midrule", r"\textbf{Razão máx./mín.} & "
           + " & ".join(f"{br(max(c) / min(c), 1)}$\\times$" for c in colunas) + r" \\",
           r"\bottomrule", r"\end{tabular}"]
escrever("perfil", linhas)

# ---------------------------------------------------------------- alocação a 50 %
linhas = [r"\begin{tabular}{l cccccccccccc}", r"\toprule",
          r"\textbf{Camada} & " + " & ".join(str(c) for c in range(12)) + r" \\", r"\midrule"]
for estrategia, nome in (("mag", "Magnitude (\\%)"), ("est", "Estruturada (\\%)")):
    taxas = [float(x) for x in CONFIGS[(estrategia, "b1")][50]["taxas_por_camada"].split(";")]
    linhas.append(f"{nome} & " + " & ".join(str(round(t * 100)) for t in taxas) + r" \\")
linhas += [r"\bottomrule", r"\end{tabular}"]
escrever("alocacao_50", linhas)

# ---------------------------------------------------------------- fine-tuning
def linha_ft(estrategia, taxa):
    return [r for r in ft if r["estrategia"] == estrategia
            and round(float(r["esparsidade_alvo"]) * 100) == taxa][0]


denso_ft = linha_ft("nenhuma", 0)
PPL0_FT, E0_FT = float(denso_ft["perplexity"]), float(denso_ft["energia_kwh"])
LINHAS_FT = [  # (rótulo, estratégia sem análise no CSV, estratégia com análise, taxa)
    ("Magnitude", "magnitude", None, 30),
    ("Magnitude", "magnitude", None, 50),
    ("Estruturada", "estruturada", "sensibilidade", 10),
    ("Estruturada", "estruturada", "sensibilidade", 30),
]


def celulas_ft(estrategia, taxa):
    if estrategia is None:
        return ["---"] * 3
    r = linha_ft(estrategia, taxa)
    return [br(float(r["perplexity_pre"])), br(ppl(r)), br(ppl(r) / PPL0_FT)]


linhas = [r"\begin{tabular}{l rrr rrr}", r"\toprule",
          r"& \multicolumn{3}{c}{\textbf{Sem análise de sensibilidade}}"
          r" & \multicolumn{3}{c}{\textbf{Com análise de sensibilidade}} \\",
          r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
          r"\textbf{Configuração}" + r" & \textbf{\textit{One-shot}} & \textbf{Pós-FT} & \textbf{Razão}" * 2
          + r" \\", r"\midrule",
          rf"Denso (controle) & \multicolumn{{6}}{{c}}{{{br(float(denso_ft['perplexity_pre']))}"
          rf" \textit{{one-shot}}; {br(PPL0_FT)} pós-FT (referência da razão)}} \\", r"\midrule"]
for rot, sem, com, t in LINHAS_FT:
    linhas.append(f"{rot} {pct(t)} & " + " & ".join(celulas_ft(sem, t) + celulas_ft(com, t)) + r" \\")
linhas += [r"\bottomrule", r"\end{tabular}"]
escrever("finetune", linhas)

# ---------------------------------------------------------------- trade-off
def ponto_eficiencia(s):
    """Maior taxa com PPL <= (1 + LIMIAR_SE) * PPL0, percorrendo as taxas em ordem."""
    melhor = None
    for t in TAXAS[1:]:
        if ppl(s[t]) > (1 + LIMIAR_SE) * PPL0:
            break
        melhor = t
    return pct(melhor) if melhor else "---"


NOME_ALOC = {"global": "Sem análise de sensibilidade --- global",
             "uniforme": "Sem análise de sensibilidade --- uniforme",
             "b1": r"Com análise de sensibilidade --- $\beta = 1{,}0$",
             "b05": r"Com análise de sensibilidade --- $\beta = 0{,}5$"}
SUB = r"\multicolumn{{5}}{{l}}{{\quad\textit{{{}}}}} \\"
linhas = [r"\begin{tabular}{l r r r c}", r"\toprule",
          r"\textbf{Configuração} & \textbf{PPL} & $\boldsymbol{\Delta E}$ \textbf{(\%)}"
          r" & $\boldsymbol{\eta}$ & $\boldsymbol{S^*_E}$ \\", r"\midrule",
          r"\multicolumn{5}{l}{\textbf{Sem retreinamento} (médias de 10\,\% a 50\,\%)} \\",
          rf"Modelo base & {br(PPL0)} & 0,0 & 1,000 & --- \\"]
for estrategia, nome in (("mag", "Poda por magnitude"), ("est", "Poda estruturada")):
    linhas.append(SUB.format(nome))
    for a in ALOCACOES:
        s = CONFIGS[(estrategia, a)]
        geo = math.exp(mean(math.log(ppl(s[t])) for t in FAIXA_RESUMO))
        de = mean(delta_e(s, t) for t in FAIXA_RESUMO)
        eta = (PPL0 / geo) / (1 - de / 100)
        linhas.append(rf"\qquad {NOME_ALOC[a]} & {br(geo)} & {br(de, 1)} & {br(eta, 3)}"
                      rf" & {ponto_eficiencia(s)} \\")
linhas += [r"\midrule", r"\multicolumn{5}{l}{\textbf{Após o \textit{fine-tuning} de recuperação}} \\",
           rf"Modelo base retreinado & {br(PPL0_FT)} & 0,0 & 1,000 & --- \\"]
CONFIGS_FT = [("Poda por magnitude", "magnitude", "uniforme", 30),
              ("Poda por magnitude", "magnitude", "uniforme", 50),
              ("Poda estruturada", "estruturada", "global", 10),
              ("Poda estruturada", "sensibilidade", "b1", 10),
              ("Poda estruturada", "estruturada", "global", 30),
              ("Poda estruturada", "sensibilidade", "b1", 30)]
anterior = None
for nome, e, a, t in CONFIGS_FT:
    if nome != anterior:
        linhas.append(SUB.format(nome))
        anterior = nome
    r = linha_ft(e, t)
    de = (1 - float(r["energia_kwh"]) / E0_FT) * 100
    eta = (PPL0_FT / ppl(r)) * (E0_FT / float(r["energia_kwh"]))
    dentro = ppl(r) <= (1 + LIMIAR_SE) * PPL0_FT
    linhas.append(rf"\qquad {NOME_ALOC[a]}, {pct(t)} & {br(ppl(r))} & {br(de, 1)} & {br(eta, 3)}"
                  rf" & {pct(t) if dentro else '---'} \\")
linhas += [r"\bottomrule", r"\end{tabular}"]
escrever("tradeoff", linhas)

print(f"Tabelas escritas em {SAIDA}")
