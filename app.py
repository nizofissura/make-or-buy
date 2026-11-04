# -*- coding: utf-8 -*-
"""
Make or Buy Decision Engine - Plataforma Corporativa
Software de Engenharia Econômica para decisões Fabricar (Make) vs. Terceirizar (Buy).

Execução:
    pip install streamlit plotly pandas numpy numpy-financial openpyxl
    streamlit run app.py
"""

from io import BytesIO

import numpy as np
import numpy_financial as npf
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA E ESTILO
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Make or Buy Decision Engine",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

GREEN = "#1B8A5A"
ORANGE = "#E8772E"
RED = "#C0392B"
NAVY = "#1F3A5F"
GRAY = "#6B7280"

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: #F7F8FA; border: 1px solid #E5E7EB; border-radius: 10px;
        padding: 14px 16px;
    }
    div[data-testid="stMetricValue"] {font-size: 1.55rem;}
    .reco-box {border-radius: 12px; padding: 18px 22px; margin: 10px 0 16px 0;
               color: #FFFFFF;}
    .reco-box h2 {margin: 0 0 4px 0; color: #FFFFFF; font-size: 1.5rem;}
    .reco-box p {margin: 2px 0; color: #FFFFFF; font-size: 0.98rem;}
    .small-note {color: #6B7280; font-size: 0.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# FUNÇÕES AUXILIARES DE FORMATAÇÃO
# ---------------------------------------------------------------------------
def brl(value, decimals=2):
    """Formata número como moeda brasileira (R$ 1.234,56)."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/d"
    s = f"{value:,.{decimals}f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"


def num(value, decimals=0):
    """Formata número no padrão brasileiro."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/d"
    s = f"{value:,.{decimals}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def pct(value, decimals=2):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/d"
    return f"{num(value * 100, decimals)}%"


# ---------------------------------------------------------------------------
# MOTOR FINANCEIRO
# ---------------------------------------------------------------------------
def annuity_factor(rate, n):
    """Fator de valor presente de uma anuidade unitária (t = 1..n)."""
    if abs(rate) < 1e-12:
        return float(n)
    return (1.0 - (1.0 + rate) ** (-n)) / rate


def delta_npv(p, Q=None, P=None):
    """
    VPL diferencial (Terceirizar - Fabricar) em forma fechada.
    Aceita Q e P escalares ou arrays NumPy (usado em Monte Carlo e sensibilidade).
    VPL > 0  => fabricar é financeiramente superior.
    """
    Q = p["Q"] if Q is None else Q
    P = p["P"] if P is None else P
    n = int(p["N"])
    r = p["r"]
    tax = p["tax"]
    dep = (p["I0"] - p["VR"]) / n
    a_f = annuity_factor(r, n)
    saving_pre_tax = (p["CFb"] - p["CFm"]) + (P - p["CV"]) * Q
    return (
        -p["I0"]
        + a_f * (saving_pre_tax * (1.0 - tax) + tax * dep)
        + p["VR"] / (1.0 + r) ** n
    )


def linear_cross(a, b):
    """
    Resolve a + b*Q = 0 para uma função linear em que 'a + b*Q > 0' significa
    que FABRICAR é melhor. Retorna (Q*, status, direção).
    status: 'ok' | 'always' (fabricar sempre melhor) | 'never' (nunca compensa)
    direção: 'above' (fabricar vence acima de Q*) | 'below' (abaixo de Q*)
    """
    if abs(b) < 1e-12:
        return (None, "always", None) if a > 0 else (None, "never", None)
    q0 = -a / b
    if b > 0:
        if q0 <= 0:
            return None, "always", None
        return q0, "ok", "above"
    if q0 <= 0:
        return None, "never", None
    return q0, "ok", "below"


def crossover_nominal(p):
    """Crossover do custo total nominal: CT_fabricar(Q) = CT_terceirizar(Q)."""
    n = int(p["N"])
    const = (p["I0"] - p["VR"]) / n + p["CFm"] - p["CFb"]  # custo fixo extra por período
    slope = p["P"] - p["CV"]
    # (CT_terceirizar - CT_fabricar)/N = -const + slope * Q
    return linear_cross(-const, slope)


def crossover_npv(p):
    """Crossover em valor presente: Q em que o VPL diferencial é zero."""
    a = float(delta_npv(p, Q=0.0))
    b = annuity_factor(p["r"], int(p["N"])) * (p["P"] - p["CV"]) * (1.0 - p["tax"])
    return linear_cross(a, b)


def total_costs(p, Q=None):
    """Custos totais nominais (formulas do enunciado)."""
    Q = p["Q"] if Q is None else Q
    n = int(p["N"])
    ct_make = p["I0"] + (p["CFm"] + p["CV"] * Q) * n - p["VR"]
    ct_buy = (p["CFb"] + p["P"] * Q) * n
    return ct_make, ct_buy


def build_table(p):
    """Memória de cálculo período a período (DRE incremental + fluxo de caixa)."""
    n = int(p["N"])
    t = np.arange(0, n + 1)
    buy_opex = np.where(t >= 1, p["CFb"] + p["P"] * p["Q"], 0.0)
    make_opex = np.where(t >= 1, p["CFm"] + p["CV"] * p["Q"], 0.0)
    econ = buy_opex - make_opex
    dep = np.where(t >= 1, (p["I0"] - p["VR"]) / n, 0.0)
    taxable = econ - dep
    tax_amt = p["tax"] * taxable
    fco = econ - tax_amt
    invest = np.zeros(n + 1)
    invest[0] = -p["I0"]
    invest[n] += p["VR"]
    fc = fco + invest
    factor = 1.0 / (1.0 + p["r"]) ** t
    fcd = fc * factor
    cum = np.cumsum(fcd)
    return pd.DataFrame(
        {
            "Período": t,
            "Custo Terceirizar (R$)": buy_opex,
            "Custo Fabricar (R$)": make_opex,
            "Economia Operacional Bruta (R$)": econ,
            "Depreciação (R$)": dep,
            "Lucro Tributável Incremental (R$)": taxable,
            "Impostos (R$)": tax_amt,
            "Fluxo Operacional Líquido (R$)": fco,
            "Investimento / Valor Residual (R$)": invest,
            "FC Diferencial (R$)": fc,
            "Fator de Desconto": factor,
            "FC Descontado (R$)": fcd,
            "FC Descontado Acumulado (R$)": cum,
        }
    )


def compute_irr(cf):
    """TIR com tratamento de não-convergência / ausência de raiz."""
    cf = np.asarray(cf, dtype=float)
    if not (np.any(cf > 0) and np.any(cf < 0)):
        return None
    try:
        val = npf.irr(cf)
    except Exception:
        return None
    if val is None or not np.isfinite(val):
        return None
    return float(val)


def compute_discounted_payback(cum):
    """Período exato (interpolado) em que o fluxo descontado acumulado cruza zero."""
    cum = np.asarray(cum, dtype=float)
    if cum[0] >= 0:
        return 0.0
    for i in range(1, len(cum)):
        if cum[i] >= 0:
            prev = cum[i - 1]
            step = cum[i] - prev
            if step == 0:
                return float(i)
            return float(i - 1 + (-prev) / step)
    return None


def run_full_analysis(p):
    table = build_table(p)
    cf = table["FC Diferencial (R$)"].values
    npv_val = float(npf.npv(p["r"], cf))
    irr_val = compute_irr(cf)
    payback = compute_discounted_payback(table["FC Descontado Acumulado (R$)"].values)
    ct_make, ct_buy = total_costs(p)
    return {
        "table": table,
        "cf": cf,
        "npv": npv_val,
        "irr": irr_val,
        "payback": payback,
        "ct_make": ct_make,
        "ct_buy": ct_buy,
        "cross_nom": crossover_nominal(p),
        "cross_npv": crossover_npv(p),
    }


def cross_text(cross, unit_q="un./período"):
    q, status, direction = cross
    if status == "ok":
        return f"{num(q, 0)} {unit_q}"
    if status == "always":
        return "Fabricar sempre vence"
    return "Fabricar nunca compensa"


def with_overrides(p, **kwargs):
    q = dict(p)
    q.update(kwargs)
    return q


# ---------------------------------------------------------------------------
# BARRA LATERAL - ENTRADAS
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Parâmetros da Análise")
period_unit = st.sidebar.selectbox(
    "Unidade do período",
    ["Anos", "Meses"],
    help="Define a unidade de tempo usada no horizonte (N), nos custos por período e na TMA.",
)
unit_singular = "ano" if period_unit == "Anos" else "mês"
unit_label = "Ano" if period_unit == "Anos" else "Mês"

with st.sidebar.expander("🏭 Opção FABRICAR (Make)", expanded=True):
    I0 = st.number_input(
        "Investimento Inicial em Equipamentos/Ativos (I₀) [R$]",
        min_value=0.0, value=500000.0, step=10000.0, format="%.2f",
        help="Desembolso feito no período 0 para montar a capacidade produtiva própria "
             "(máquinas, instalações, ferramental).",
    )
    CFm = st.number_input(
        "Custo Fixo Operacional por Período (CF fabricar) [R$]",
        min_value=0.0, value=20000.0, step=1000.0, format="%.2f",
        help="Custos que não variam com o volume: aluguel, mão de obra fixa, manutenção "
             "preventiva, seguros, por período.",
    )
    CV = st.number_input(
        "Custo Variável Unitário de Produção (CV fabricar) [R$/unidade]",
        min_value=0.0, value=35.0, step=0.5, format="%.2f",
        help="Matéria-prima, energia e mão de obra direta consumidas para produzir 1 unidade.",
    )
    N = st.number_input(
        f"Vida Útil / Horizonte do Projeto (N) [{period_unit}]",
        min_value=1, max_value=600, value=5, step=1,
        help=f"Número de períodos ({period_unit.lower()}) da análise. Também é o prazo "
             "de depreciação linear dos ativos.",
    )
    VR = st.number_input(
        "Valor Residual dos Equipamentos (VR) [R$]",
        min_value=0.0, value=50000.0, step=5000.0, format="%.2f",
        help="Valor estimado de venda dos ativos ao final do horizonte. "
             "Não pode ser maior que o investimento inicial.",
    )

with st.sidebar.expander("🤝 Opção TERCEIRIZAR (Buy)", expanded=True):
    P = st.number_input(
        "Preço de Aquisição Unitário do Fornecedor (P) [R$/unidade]",
        min_value=0.0, value=60.0, step=0.5, format="%.2f",
        help="Preço pago ao fornecedor por unidade, já com frete e demais custos variáveis de compra.",
    )
    CFb = st.number_input(
        "Custos Fixos de Gestão de Contrato/Logística (CF terceirizar) [R$]",
        min_value=0.0, value=5000.0, step=500.0, format="%.2f",
        help="Custos fixos por período para gerir o fornecedor: contratos, qualidade, "
             "estoque de segurança, logística.",
    )

with st.sidebar.expander("📐 Premissas Operacionais e Financeiras", expanded=True):
    Q = st.number_input(
        "Demanda / Volume por Período (Q) [unidades]",
        min_value=0.0, value=8000.0, step=100.0, format="%.0f",
        help="Quantidade de unidades necessárias por período (assumida constante no horizonte).",
    )
    TMA_pct = st.number_input(
        f"Taxa Mínima de Atratividade (TMA) [% ao {unit_singular}]",
        min_value=0.0, max_value=500.0, value=12.0, step=0.5, format="%.2f",
        help="Retorno mínimo exigido pelo investidor. Na unidade do período escolhido "
             f"(% ao {unit_singular}).",
    )
    TAX_pct = st.number_input(
        "Alíquota de Impostos/Tributos Incidentes [%]",
        min_value=0.0, max_value=99.0, value=34.0, step=1.0, format="%.2f",
        help="Alíquota sobre o lucro tributável incremental (ex.: IRPJ+CSLL ≈ 34%). "
             "Custos e depreciação geram escudo fiscal. Use 0 para análise pré-impostos.",
    )

st.sidebar.markdown("---")
st.sidebar.caption("Make or Buy Decision Engine • Engenharia Econômica")

# Validações de entrada
input_errors = []
if VR > I0:
    input_errors.append("O Valor Residual (VR) não pode ser maior que o Investimento Inicial (I₀).")

params = {
    "I0": float(I0),
    "CFm": float(CFm),
    "CV": float(CV),
    "N": int(N),
    "VR": float(VR),
    "P": float(P),
    "CFb": float(CFb),
    "Q": float(Q),
    "r": float(TMA_pct) / 100.0,
    "tax": float(TAX_pct) / 100.0,
}

# ---------------------------------------------------------------------------
# CABEÇALHO
# ---------------------------------------------------------------------------
st.title("🏭 Make or Buy Decision Engine")
st.caption(
    "Plataforma corporativa de Engenharia Econômica — análise determinística e estocástica "
    "para decisões de Fabricar Internamente vs. Terceirizar."
)

if input_errors:
    for msg in input_errors:
        st.error(msg)
    st.stop()

res = run_full_analysis(params)
npv_val = res["npv"]
irr_val = res["irr"]
payback = res["payback"]
ct_make, ct_buy = res["ct_make"], res["ct_buy"]
cross_nom = res["cross_nom"]
cross_npv_res = res["cross_npv"]
table = res["table"]
make_wins = npv_val > 0

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "💡 Simulador Principal",
        "📈 Cenários & Sensibilidade",
        "📄 Relatório Executivo",
        "❓ Guia & Glossário",
    ]
)

# ===========================================================================
# ABA 1 - SIMULADOR PRINCIPAL
# ===========================================================================
with tab1:
    saving = ct_buy - ct_make
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Economia Estimada (nominal)",
        brl(abs(saving), 0),
        delta=("Fabricar economiza" if saving > 0 else "Terceirizar economiza"),
        delta_color="normal" if saving > 0 else "inverse",
        help="Diferença entre os custos totais nominais (CT terceirizar − CT fabricar), "
             "sem desconto e sem impostos.",
    )
    c2.metric(
        "VPL Diferencial",
        brl(npv_val, 0),
        delta="Favorece Fabricar" if npv_val > 0 else "Favorece Terceirizar",
        delta_color="normal" if npv_val > 0 else "inverse",
        help="Valor presente do fluxo diferencial (Terceirizar − Fabricar) à TMA, "
             "já com impostos. Positivo => fabricar cria valor.",
    )
    c3.metric(
        "TIR Diferencial",
        pct(irr_val) if irr_val is not None else "Não definida",
        delta=(
            f"{'acima' if irr_val > params['r'] else 'abaixo'} da TMA ({pct(params['r'])})"
            if irr_val is not None else "sem convergência / sem mudança de sinal"
        ),
        delta_color=(
            "off" if irr_val is None else ("normal" if irr_val > params["r"] else "inverse")
        ),
        help="Taxa que zera o VPL do fluxo diferencial. Compare com a TMA.",
    )
    c4.metric(
        "Ponto de Crossover (Q)",
        cross_text(cross_nom),
        delta=f"Demanda atual: {num(Q, 0)} un.",
        delta_color="off",
        help="Volume por período em que o custo total de fabricar iguala o de terceirizar "
             "(custos nominais).",
    )

    # ----- Painel de recomendação -----
    if make_wins:
        reco_title = "RECOMENDAÇÃO: FABRICAR INTERNAMENTE"
        reco_color = GREEN
    else:
        reco_title = "RECOMENDAÇÃO: TERCEIRIZAR / COMPRAR"
        reco_color = ORANGE

    q_cross, st_cross, dir_cross = cross_nom
    if st_cross == "ok":
        if dir_cross == "above":
            rel = "acima" if Q > q_cross else "abaixo"
            q_line = (
                f"A demanda atual ({num(Q, 0)} un.) está {rel} do volume de crossover "
                f"({num(q_cross, 0)} un.); fabricar tem custo nominal menor apenas acima desse volume."
            )
        else:
            rel = "abaixo" if Q < q_cross else "acima"
            q_line = (
                f"A demanda atual ({num(Q, 0)} un.) está {rel} do volume de crossover "
                f"({num(q_cross, 0)} un.); neste cenário fabricar só tem custo nominal menor abaixo desse volume."
            )
    elif st_cross == "always":
        q_line = ("Em custos nominais, fabricar é mais barato em qualquer volume "
                  "(não há ponto de crossover positivo).")
    else:
        q_line = ("Em custos nominais, fabricar não é mais barato em nenhum volume — o custo "
                  "variável próprio não compensa o custo fixo/investimento adicional.")

    if make_wins:
        npv_line = (f"VPL diferencial de {brl(npv_val, 0)} à TMA de {pct(params['r'])}: fabricar "
                    f"gera valor frente a terceirizar.")
    else:
        npv_line = (f"VPL diferencial de {brl(npv_val, 0)} à TMA de {pct(params['r'])}: o "
                    f"investimento em fabricar não remunera o capital exigido.")

    if irr_val is None:
        irr_line = ("A TIR não pôde ser determinada (fluxo sem mudança de sinal ou sem convergência numérica); "
                    "a decisão se apoia no VPL.")
    elif irr_val > params["r"]:
        irr_line = f"TIR de {pct(irr_val)} supera a TMA ({pct(params['r'])}), confirmando a atratividade."
    else:
        irr_line = f"TIR de {pct(irr_val)} fica abaixo da TMA ({pct(params['r'])}), indicando retorno insuficiente."

    if payback is None:
        pb_line = f"O payback descontado não é atingido dentro do horizonte de {params['N']} {period_unit.lower()}."
    else:
        pb_line = f"Payback descontado em {num(payback, 2)} {period_unit.lower()} (horizonte: {params['N']})."

    st.markdown(
        f"""
        <div class="reco-box" style="background:{reco_color};">
            <h2>{reco_title}</h2>
            <p>• {npv_line}</p>
            <p>• {irr_line}</p>
            <p>• {q_line}</p>
            <p>• {pb_line}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cross_n_q, cross_n_status, _ = cross_npv_res
    st.caption(
        f"Crossover em valor presente (VPL = 0, com impostos e TMA): {cross_text(cross_npv_res)}. "
        "O crossover nominal ignora o valor do dinheiro no tempo e os impostos — "
        "por isso os dois podem divergir; a recomendação segue o VPL."
    )

    # ----- Gráfico 1: Crossover -----
    st.subheader("Gráfico 1 — Análise de Crossover (Ponto de Equilíbrio)")
    q_candidates = [Q]
    if cross_nom[1] == "ok":
        q_candidates.append(cross_nom[0])
    q_max = max(q_candidates) * 2.0
    if q_max <= 0:
        q_max = 1000.0
    q_axis = np.linspace(0, q_max, 200)
    make_curve = np.array([total_costs(params, q)[0] for q in q_axis])
    buy_curve = np.array([total_costs(params, q)[1] for q in q_axis])

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=q_axis, y=make_curve, name="Custo Total — Fabricar",
                              line=dict(color=GREEN, width=3)))
    fig1.add_trace(go.Scatter(x=q_axis, y=buy_curve, name="Custo Total — Terceirizar",
                              line=dict(color=ORANGE, width=3)))
    if cross_nom[1] == "ok":
        qx = cross_nom[0]
        yx = total_costs(params, qx)[0]
        fig1.add_trace(go.Scatter(
            x=[qx], y=[yx], mode="markers+text", name="Crossover",
            marker=dict(size=13, color=NAVY, symbol="diamond"),
            text=[f"Q* = {num(qx, 0)}"], textposition="top center",
        ))
    fig1.add_vline(x=Q, line_dash="dash", line_color=GRAY,
                   annotation_text=f"Demanda atual ({num(Q, 0)})", annotation_position="top left")
    fig1.update_layout(
        template="plotly_white", height=430,
        xaxis_title=f"Volume por período (Q) [unidades]",
        yaxis_title=f"Custo total nominal em {params['N']} {period_unit.lower()} [R$]",
        legend=dict(orientation="h", y=1.12), margin=dict(t=60),
    )
    st.plotly_chart(fig1, width="stretch")

    g2, g3 = st.columns(2)

    # ----- Gráfico 2: Sensibilidade à TMA -----
    with g2:
        st.subheader("Gráfico 2 — Sensibilidade do VPL à TMA")
        rates = np.arange(0.0, 50.0001, 0.5) / 100.0
        npv_curve = [float(npf.npv(rt, res["cf"])) for rt in rates]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=rates * 100, y=npv_curve, name="VPL diferencial",
                                  line=dict(color=NAVY, width=3)))
        fig2.add_hline(y=0, line_color=GRAY, line_dash="dot")
        fig2.add_vline(x=params["r"] * 100, line_dash="dash", line_color=ORANGE,
                       annotation_text=f"TMA {pct(params['r'])}", annotation_position="top")
        if irr_val is not None and 0 <= irr_val <= 0.5:
            fig2.add_trace(go.Scatter(x=[irr_val * 100], y=[0], mode="markers", name="TIR",
                                      marker=dict(size=12, color=GREEN, symbol="diamond")))
        fig2.update_layout(
            template="plotly_white", height=380,
            xaxis_title=f"TMA [% ao {unit_singular}]", yaxis_title="VPL diferencial [R$]",
            legend=dict(orientation="h", y=1.15), margin=dict(t=50),
        )
        st.plotly_chart(fig2, width="stretch")

    # ----- Gráfico 3: Payback descontado -----
    with g3:
        st.subheader("Gráfico 3 — Payback Descontado")
        fig3 = go.Figure()
        cum = table["FC Descontado Acumulado (R$)"].values
        colors = [GREEN if v >= 0 else RED for v in cum]
        fig3.add_trace(go.Bar(x=table["Período"], y=cum, marker_color=colors,
                              name="Fluxo acumulado descontado"))
        fig3.add_trace(go.Scatter(x=table["Período"], y=cum, mode="lines+markers",
                                  line=dict(color=NAVY, width=2), name="Tendência",
                                  showlegend=False))
        fig3.add_hline(y=0, line_color=GRAY)
        if payback is not None:
            fig3.add_vline(x=payback, line_dash="dash", line_color=GREEN,
                           annotation_text=f"Payback ≈ {num(payback, 2)}",
                           annotation_position="top")
        fig3.update_layout(
            template="plotly_white", height=380,
            xaxis_title=f"Período [{period_unit}]", yaxis_title="Fluxo acumulado descontado [R$]",
            legend=dict(orientation="h", y=1.15), margin=dict(t=50),
        )
        st.plotly_chart(fig3, width="stretch")

    with st.expander("🔎 Custos totais nominais (fórmulas do modelo)"):
        st.latex(r"CT_{fabricar} = I_0 + (CF_{fabricar} + CV_{fabricar}\cdot Q)\cdot N - VR")
        st.latex(r"CT_{terceirizar} = (CF_{terceirizar} + P_{terceirizar}\cdot Q)\cdot N")
        ca, cb, cc = st.columns(3)
        ca.metric("CT Fabricar", brl(ct_make, 0))
        cb.metric("CT Terceirizar", brl(ct_buy, 0))
        cc.metric("Diferença (Terc. − Fabr.)", brl(ct_buy - ct_make, 0))


# ===========================================================================
# ABA 2 - CENÁRIOS & SENSIBILIDADE
# ===========================================================================
with tab2:
    st.header("Comparativo de Cenários")
    st.caption(
        "Defina variações percentuais sobre a Demanda (Q) e o Preço do Fornecedor (P). "
        "Convenção: **Otimista** = melhor para a opção Fabricar (mais demanda, fornecedor mais caro); "
        "**Pessimista** = pior para Fabricar."
    )

    sc_defaults = {
        "Pessimista": (-20.0, -10.0),
        "Realista": (0.0, 0.0),
        "Otimista": (20.0, 10.0),
    }
    cols = st.columns(3)
    scenario_inputs = {}
    for col, (name, (dq0, dp0)) in zip(cols, sc_defaults.items()):
        with col:
            st.markdown(f"**{name}**")
            dq = st.number_input(f"Δ Demanda (Q) — {name} [%]", min_value=-99.0, max_value=500.0,
                                 value=dq0, step=5.0, key=f"dq_{name}",
                                 help="Variação percentual aplicada à demanda base.")
            dp = st.number_input(f"Δ Preço do fornecedor — {name} [%]", min_value=-99.0,
                                 max_value=500.0, value=dp0, step=5.0, key=f"dp_{name}",
                                 help="Variação percentual aplicada ao preço unitário do fornecedor.")
            scenario_inputs[name] = (dq, dp)

    rows = []
    for name, (dq, dp) in scenario_inputs.items():
        ps = with_overrides(params, Q=params["Q"] * (1 + dq / 100.0), P=params["P"] * (1 + dp / 100.0))
        rs = run_full_analysis(ps)
        rows.append({
            "Cenário": name,
            "Demanda Q (un.)": ps["Q"],
            "Preço Fornecedor (R$)": ps["P"],
            "VPL Diferencial (R$)": rs["npv"],
            "TIR": pct(rs["irr"]) if rs["irr"] is not None else "n/d",
            "Payback Desc.": (num(rs["payback"], 2) if rs["payback"] is not None else "Não atingido"),
            "Crossover Nominal (un.)": cross_text(rs["cross_nom"], ""),
            "Crossover VPL (un.)": cross_text(rs["cross_npv"], ""),
            "Recomendação": "Fabricar" if rs["npv"] > 0 else "Terceirizar",
        })
    df_sc = pd.DataFrame(rows)
    st.dataframe(
        df_sc.style.format({
            "Demanda Q (un.)": lambda v: num(v, 0),
            "Preço Fornecedor (R$)": lambda v: brl(v),
            "VPL Diferencial (R$)": lambda v: brl(v, 0),
        }),
        width="stretch", hide_index=True,
    )

    figsc = go.Figure(go.Bar(
        x=df_sc["Cenário"], y=df_sc["VPL Diferencial (R$)"],
        marker_color=[GREEN if v > 0 else ORANGE for v in df_sc["VPL Diferencial (R$)"]],
        text=[brl(v, 0) for v in df_sc["VPL Diferencial (R$)"]], textposition="outside",
    ))
    figsc.add_hline(y=0, line_color=GRAY)
    figsc.update_layout(template="plotly_white", height=340, yaxis_title="VPL diferencial [R$]",
                        margin=dict(t=30))
    st.plotly_chart(figsc, width="stretch")

    st.markdown("---")
    # ----------------------------- Monte Carlo -----------------------------
    st.header("Simulação Estocástica — Monte Carlo")
    st.caption("Sorteia a demanda (e opcionalmente o preço do fornecedor) em torno do caso base e "
               "mede a probabilidade de Fabricar superar Terceirizar (VPL diferencial > 0).")

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc_margin = mc1.slider("Incerteza da demanda (±%)", 0, 80, 20, 1,
                           help="Margem de variação da demanda em torno do valor base.")
    mc_price_margin = mc2.slider("Incerteza do preço do fornecedor (±%)", 0, 50, 0, 1,
                                 help="Use 0 para variar apenas a demanda.")
    mc_dist = mc3.selectbox("Distribuição", ["Uniforme", "Triangular", "Normal (±margem = 2σ)"],
                            help="Forma da distribuição de probabilidade dos sorteios.")
    mc_n = mc4.select_slider("Nº de simulações", options=[1000, 5000, 10000, 20000, 50000],
                             value=10000, help="Mais simulações = estimativa mais estável.")
    mc_seed = st.number_input("Semente aleatória (reprodutibilidade)", min_value=0, value=42, step=1,
                              help="Mesma semente gera os mesmos resultados.")

    def draw(rng, center, margin_pct, n_draw, dist):
        m = margin_pct / 100.0
        if center == 0 or m == 0:
            return np.full(n_draw, center, dtype=float)
        lo, hi = center * (1 - m), center * (1 + m)
        if dist == "Uniforme":
            vals = rng.uniform(lo, hi, n_draw)
        elif dist == "Triangular":
            vals = rng.triangular(lo, center, hi, n_draw)
        else:
            vals = rng.normal(center, center * m / 2.0, n_draw)
        return np.clip(vals, 0.0, None)

    rng = np.random.default_rng(int(mc_seed))
    q_s = draw(rng, params["Q"], mc_margin, mc_n, mc_dist)
    p_s = draw(rng, params["P"], mc_price_margin, mc_n, mc_dist)
    npv_s = np.asarray(delta_npv(params, Q=q_s, P=p_s), dtype=float)

    prob_make = float(np.mean(npv_s > 0))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Prob. de Fabricar ser superior", f"{prob_make * 100:.1f}%",
              help="Percentual das simulações com VPL diferencial > 0.")
    m2.metric("VPL médio", brl(float(np.mean(npv_s)), 0))
    m3.metric("VPL P5 (cenário adverso)", brl(float(np.percentile(npv_s, 5)), 0))
    m4.metric("VPL P95 (cenário favorável)", brl(float(np.percentile(npv_s, 95)), 0))

    fig_mc = go.Figure()
    fig_mc.add_trace(go.Histogram(x=npv_s, nbinsx=60, marker_color=NAVY, opacity=0.85,
                                  name="Distribuição do VPL"))
    fig_mc.add_vline(x=0, line_color=RED, line_dash="dash", annotation_text="VPL = 0")
    fig_mc.update_layout(template="plotly_white", height=360, xaxis_title="VPL diferencial [R$]",
                         yaxis_title="Frequência", margin=dict(t=30), showlegend=False)
    st.plotly_chart(fig_mc, width="stretch")

    if prob_make >= 0.7:
        st.success(f"Em {prob_make * 100:.1f}% das simulações fabricar supera terceirizar — decisão robusta a favor de Fabricar.")
    elif prob_make <= 0.3:
        st.warning(f"Apenas {prob_make * 100:.1f}% das simulações favorecem Fabricar — decisão robusta a favor de Terceirizar.")
    else:
        st.info(f"Probabilidade de {prob_make * 100:.1f}%: decisão sensível à incerteza; considere negociar contratos "
                "flexíveis ou obter melhores estimativas de demanda.")

    st.markdown("---")
    # ----------------------------- Tornado ---------------------------------
    st.header("Gráfico de Tornado — Sensibilidade do VPL")
    tv = st.slider("Variação aplicada a cada variável (±%)", 5, 50, 20, 5,
                   help="Cada variável é alterada isoladamente para cima e para baixo.")
    f = tv / 100.0
    tornado_vars = [
        ("Investimento Inicial (I₀)", "I0"),
        ("Preço do Fornecedor (P)", "P"),
        ("Custo Variável de Fabricação (CV)", "CV"),
        ("Demanda (Q)", "Q"),
        ("Custo Fixo Fabricar", "CFm"),
        ("Custo Fixo Terceirizar", "CFb"),
        ("TMA", "r"),
    ]
    base_npv = float(delta_npv(params))
    t_rows = []
    for label, key in tornado_vars:
        lo = float(delta_npv(with_overrides(params, **{key: params[key] * (1 - f)})))
        hi = float(delta_npv(with_overrides(params, **{key: params[key] * (1 + f)})))
        t_rows.append({"Variável": label, "VPL (−)": lo, "VPL (+)": hi,
                       "Amplitude": abs(hi - lo)})
    df_t = pd.DataFrame(t_rows).sort_values("Amplitude", ascending=True)

    fig_t = go.Figure()
    fig_t.add_trace(go.Bar(
        y=df_t["Variável"], x=df_t["VPL (−)"] - base_npv, orientation="h",
        name=f"−{tv}%", marker_color=ORANGE,
        customdata=df_t["VPL (−)"], hovertemplate="VPL: R$ %{customdata:,.0f}<extra>−" + str(tv) + "%</extra>",
    ))
    fig_t.add_trace(go.Bar(
        y=df_t["Variável"], x=df_t["VPL (+)"] - base_npv, orientation="h",
        name=f"+{tv}%", marker_color=GREEN,
        customdata=df_t["VPL (+)"], hovertemplate="VPL: R$ %{customdata:,.0f}<extra>+" + str(tv) + "%</extra>",
    ))
    fig_t.update_layout(
        barmode="overlay", template="plotly_white", height=420,
        xaxis_title=f"Variação do VPL em relação ao caso base ({brl(base_npv, 0)}) [R$]",
        legend=dict(orientation="h", y=1.1), margin=dict(t=50),
    )
    fig_t.add_vline(x=0, line_color=GRAY)
    st.plotly_chart(fig_t, width="stretch")

    st.dataframe(
        df_t.sort_values("Amplitude", ascending=False).style.format({
            "VPL (−)": lambda v: brl(v, 0), "VPL (+)": lambda v: brl(v, 0),
            "Amplitude": lambda v: brl(v, 0),
        }),
        width="stretch", hide_index=True,
    )


# ===========================================================================
# ABA 3 - RELATÓRIO EXECUTIVO
# ===========================================================================
with tab3:
    st.header("Relatório Executivo & Memória de Cálculo")

    st.subheader("DRE Incremental Simplificada e Fluxo de Caixa Diferencial")
    st.caption(
        "Valores positivos no fluxo = vantagem de fabricar em relação a terceirizar. "
        "Impostos negativos representam crédito fiscal (escudo tributário)."
    )
    money_cols = [c for c in table.columns if "(R$)" in c]
    fmt = {c: (lambda v: brl(v)) for c in money_cols}
    fmt["Fator de Desconto"] = lambda v: f"{v:.6f}"
    fmt["Período"] = lambda v: f"{int(v)}"
    st.dataframe(table.style.format(fmt), width="stretch", hide_index=True)

    totals = {
        "Total Economia Operacional": table["Economia Operacional Bruta (R$)"].sum(),
        "Total Depreciação": table["Depreciação (R$)"].sum(),
        "Total Impostos": table["Impostos (R$)"].sum(),
        "Soma FC Diferencial (nominal)": table["FC Diferencial (R$)"].sum(),
        "VPL (soma FC descontado)": table["FC Descontado (R$)"].sum(),
    }
    tc = st.columns(len(totals))
    for col, (k, v) in zip(tc, totals.items()):
        col.metric(k, brl(v, 0))

    st.subheader("Resumo Executivo (pronto para apresentação)")
    decision_txt = "FABRICAR INTERNAMENTE" if make_wins else "TERCEIRIZAR / COMPRAR"
    summary_lines = [
        "MAKE OR BUY — RESUMO EXECUTIVO",
        "=" * 46,
        f"Decisão recomendada: {decision_txt}",
        "",
        "PREMISSAS",
        f"• Horizonte: {params['N']} {period_unit.lower()} | Demanda: {num(Q, 0)} un./{unit_singular}",
        f"• TMA: {pct(params['r'])} a.{'a.' if period_unit == 'Anos' else 'm.'} | Alíquota: {pct(params['tax'])}",
        f"• Fabricar: I₀ {brl(I0, 0)} | CF {brl(CFm, 0)} | CV {brl(CV)}/un. | VR {brl(VR, 0)}",
        f"• Terceirizar: P {brl(P)}/un. | CF {brl(CFb, 0)}",
        "",
        "RESULTADOS",
        f"• CT Fabricar (nominal): {brl(ct_make, 0)}",
        f"• CT Terceirizar (nominal): {brl(ct_buy, 0)}",
        f"• Economia nominal: {brl(ct_buy - ct_make, 0)} "
        f"({'a favor de Fabricar' if ct_buy - ct_make > 0 else 'a favor de Terceirizar'})",
        f"• VPL diferencial: {brl(npv_val, 0)}",
        f"• TIR diferencial: {pct(irr_val) if irr_val is not None else 'não definida'}",
        f"• Payback descontado: "
        f"{(num(payback, 2) + ' ' + period_unit.lower()) if payback is not None else 'não atingido no horizonte'}",
        f"• Crossover nominal: {cross_text(cross_nom)}",
        f"• Crossover em VPL: {cross_text(cross_npv_res)}",
        f"• Prob. de Fabricar superior (Monte Carlo, ±{mc_margin}% demanda): {prob_make * 100:.1f}%",
    ]
    summary_text = "\n".join(summary_lines)
    st.code(summary_text, language=None)

    st.subheader("Exportação de Dados")
    csv_bytes = table.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
    e1, e2 = st.columns(2)
    e1.download_button(
        "⬇️ Baixar memória de cálculo (CSV)", data=csv_bytes,
        file_name="make_or_buy_memoria_calculo.csv", mime="text/csv", width="stretch",
    )
    try:
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            table.to_excel(writer, sheet_name="Memoria_Calculo", index=False)
            pd.DataFrame({
                "Parâmetro": ["I0", "CF fabricar", "CV fabricar", "N", "VR", "P terceirizar",
                              "CF terceirizar", "Q", "TMA", "Alíquota"],
                "Valor": [params["I0"], params["CFm"], params["CV"], params["N"], params["VR"],
                          params["P"], params["CFb"], params["Q"], params["r"], params["tax"]],
            }).to_excel(writer, sheet_name="Premissas", index=False)
            pd.DataFrame({"Resumo Executivo": summary_lines}).to_excel(
                writer, sheet_name="Resumo", index=False)
        e2.download_button(
            "⬇️ Baixar relatório completo (Excel)", data=buffer.getvalue(),
            file_name="make_or_buy_relatorio.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )
    except Exception:
        e2.warning("Exportação para Excel indisponível. Instale o pacote `openpyxl` "
                   "(pip install openpyxl) ou use o CSV.")

    with st.expander("📌 Premissas e limitações do modelo"):
        st.markdown(
            """
            - Demanda, preços e custos são **constantes** em todos os períodos do horizonte.
            - **Depreciação linear** do investimento até o valor residual; o valor residual é
              considerado igual ao valor contábil final (sem ganho/perda de capital tributável).
            - Impostos incidem sobre o **lucro tributável incremental** (economia operacional − depreciação);
              prejuízo incremental gera crédito fiscal imediato.
            - O investimento ocorre no período 0 e os fluxos operacionais ao final de cada período.
            - Capital de giro, inflação e financiamento do investimento não são modelados.
            - O VPL é calculado sobre o fluxo diferencial (Terceirizar − Fabricar): **VPL > 0 ⇒ fabricar**.
            """
        )


# ===========================================================================
# ABA 4 - GUIA & GLOSSÁRIO
# ===========================================================================
with tab4:
    st.header("Guia de Uso")
    st.markdown(
        """
        **Passo a passo**
        1. **Preencha a barra lateral** com os dados de Fabricar, Terceirizar e as premissas
           (demanda, TMA, impostos). Passe o mouse sobre o ícone **?** de cada campo para ver o que digitar.
        2. **Leia a Recomendação Executiva** na aba *Simulador Principal*: ela combina VPL, TIR, payback e a
           posição da demanda atual em relação ao ponto de crossover.
        3. **Interprete os KPIs**: VPL > 0 e TIR > TMA indicam que fabricar remunera o capital acima do exigido.
        4. **Teste a robustez** na aba *Cenários & Sensibilidade*: compare cenários, observe a probabilidade
           de Monte Carlo e identifique no tornado as variáveis críticas.
        5. **Documente a decisão** na aba *Relatório Executivo*: copie o resumo e exporte a memória de cálculo.

        **Como decidir**
        - **VPL diferencial > 0** → Fabricar internamente.
        - **VPL diferencial < 0** → Terceirizar.
        - Se a **probabilidade de Monte Carlo** estiver entre 30% e 70%, a decisão é frágil: busque contratos
          flexíveis, projeções de demanda melhores ou negocie o preço do fornecedor.
        - Se a **demanda atual está próxima do crossover**, pequenas mudanças de volume podem inverter a decisão.
        - Se **TIR não é definida**, o fluxo diferencial não muda de sinal (ex.: fabricar é melhor ou pior
          em todos os períodos); use o VPL.
        """
    )

    st.header("Glossário de Engenharia Econômica")
    with st.expander("VPL — Valor Presente Líquido", expanded=True):
        st.markdown(
            "Soma dos fluxos de caixa futuros trazidos a valor presente pela TMA, menos o investimento inicial. "
            "Mede **quanto valor** a alternativa cria em reais de hoje. Neste app é calculado sobre o fluxo "
            "diferencial (Terceirizar − Fabricar): positivo significa que fabricar é melhor."
        )
        st.latex(r"VPL = \sum_{t=0}^{N} \frac{\Delta CF_t}{(1+TMA)^t}")
    with st.expander("TIR — Taxa Interna de Retorno", expanded=True):
        st.markdown(
            "Taxa de desconto que torna o VPL igual a zero. É o retorno percentual do investimento em fabricar. "
            "Se **TIR > TMA**, o investimento é atrativo. Pode não existir quando o fluxo não troca de sinal "
            "ou pode haver mais de uma raiz em fluxos não convencionais."
        )
    with st.expander("Payback Descontado", expanded=True):
        st.markdown(
            "Tempo necessário para que o fluxo de caixa acumulado **já descontado pela TMA** recupere o "
            "investimento inicial. Considera o valor do dinheiro no tempo (mais rigoroso que o payback simples). "
            "Quanto menor, menor o risco de exposição do capital."
        )
    with st.expander("TMA — Taxa Mínima de Atratividade", expanded=True):
        st.markdown(
            "Retorno mínimo que a empresa exige para aceitar um investimento; reflete o custo de oportunidade "
            "do capital, o custo de financiamento e o risco do projeto. É a taxa usada para descontar os fluxos."
        )
    with st.expander("Ponto de Crossover / Equilíbrio Operacional", expanded=True):
        st.markdown(
            "Volume de produção por período em que o custo total de fabricar iguala o de terceirizar. "
            "Fabricar tem custo fixo/investimento maior e custo variável menor; por isso costuma compensar "
            "**acima** de certo volume. Fórmula nominal:"
        )
        st.latex(r"Q_{crossover} = \frac{\frac{I_0 - VR}{N} + CF_{fabricar} - CF_{terceirizar}}{P_{terceirizar} - CV_{fabricar}}")
        st.markdown(
            "Se **P = CV**, o denominador é zero e não existe crossover finito: a vantagem depende apenas dos "
            "custos fixos. O app trata esse caso e informa se fabricar sempre ou nunca compensa."
        )
    with st.expander("Análise de Monte Carlo e Gráfico de Tornado"):
        st.markdown(
            "**Monte Carlo** repete a análise milhares de vezes com valores sorteados para as variáveis incertas "
            "e informa a probabilidade de cada decisão ser a melhor. **Tornado** varia uma variável por vez e "
            "ordena pelo impacto no VPL, mostrando onde concentrar esforços de estimativa e negociação."
        )

st.markdown("---")
st.caption("Make or Buy Decision Engine • Ferramenta de apoio à decisão — "
           "resultados dependem da qualidade das premissas informadas.")
