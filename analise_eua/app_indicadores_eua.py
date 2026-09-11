"""Dashboard Streamlit dos indicadores fundamentalistas das empresas dos EUA.

Uso:
    streamlit run analise_eua/app_indicadores_eua.py

Os arquivos são organizados como ``<setor>/<empresa>/<ticker>_indicators.xlsx``
e contêm as abas ``indicators_10k`` (anual) e ``indicators_10q`` (trimestral).
Os demonstrativos estão em US$ milhões; o valor de mercado, em US$ milhares.
"""

import shutil
import subprocess
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from curl_cffi.requests import RequestsError
from scipy.stats import t as student_t
from yfinance.exceptions import YFException

RAIZ_PADRAO = r"C:\B3\historico-arquivos\indicadores_fundamentalistas_eua"
RAIZ_VALUATIONS = r"C:\B3\historico-arquivos\valuations\atualizado"

SETORES = {
    "energia_eletrica": "Energia elétrica",
    "semicondutores": "Semicondutores",
    "tecnologia": "Tecnologia",
}

PALETA = {
    "light": {
        "serie": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
        "texto": "#0b0b0b", "texto_2": "#52514e", "grade": "#e5e4e0",
        "positivo": "#2a78d6", "negativo": "#e34948", "neutro": "#8f8e88",
    },
    "dark": {
        "serie": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
        "texto": "#ffffff", "texto_2": "#c3c2b7", "grade": "#383835",
        "positivo": "#3987e5", "negativo": "#e66767", "neutro": "#7c7b76",
    },
}


@dataclass(frozen=True)
class Metrica:
    label: str
    tipo: str  # usd_bi | pct | x | usd
    divisor: float = 1.0
    ajuda: str = ""

    @property
    def unidade(self) -> str:
        return {"usd_bi": "US$ bi", "pct": "%", "x": "x", "usd": "US$"}[self.tipo]


# Demonstrativos em US$ milhões; valor de mercado em US$ milhares.
MILHAO_PARA_BILHAO = 1e3
MILHAR_PARA_BILHAO = 1e6
METRICAS = {
    "pl_damodaran": Metrica("P/L Damodaran", "x", 1, "Preço sobre lucro pelo modelo Damodaran"),
    "pl": Metrica("P/L", "x"),
    "lp": Metrica("L/P (earnings yield)", "pct"),
    "ebitda": Metrica("EBITDA", "usd_bi", MILHAO_PARA_BILHAO),
    "margem_liquida": Metrica("Margem líquida", "pct"),
    "pvp": Metrica("P/VPA", "x"),
    "valor_mercado": Metrica("Valor de mercado", "usd_bi", MILHAR_PARA_BILHAO),
    "divida_bruta": Metrica("Dívida bruta", "usd_bi", MILHAO_PARA_BILHAO),
    "caixa": Metrica("Caixa e equivalentes", "usd_bi", MILHAO_PARA_BILHAO),
    "divida_liquida": Metrica("Dívida líquida", "usd_bi", MILHAO_PARA_BILHAO),
    "dl_ebitda": Metrica("Dívida líquida / EBITDA", "x"),
    "dl_pl": Metrica("Dívida líquida / PL", "x"),
    "ev_ebitda": Metrica("EV/EBITDA", "x"),
    "roe": Metrica("ROE", "pct"),
    "roic": Metrica("ROIC", "pct"),
    "fco": Metrica("Fluxo de caixa operacional", "usd_bi", MILHAO_PARA_BILHAO),
    "fci": Metrica("Fluxo de caixa de investimento", "usd_bi", MILHAO_PARA_BILHAO),
    "fcf": Metrica("Fluxo de caixa de financiamento", "usd_bi", MILHAO_PARA_BILHAO),
    "free_cash_flow": Metrica("Free cash flow", "usd_bi", MILHAO_PARA_BILHAO),
    "capex": Metrica("Capex", "usd_bi", MILHAO_PARA_BILHAO),
    "net_capex": Metrica("Net capex", "usd_bi", MILHAO_PARA_BILHAO),
    "rd": Metrica("P&D", "usd_bi", MILHAO_PARA_BILHAO),
    "adj_net_capex": Metrica("Net capex ajustado", "usd_bi", MILHAO_PARA_BILHAO),
    "working_capital": Metrica("Capital de giro", "usd_bi", MILHAO_PARA_BILHAO),
    "reinvestment_rate": Metrica("Taxa de reinvestimento", "pct"),
    "fcfe": Metrica("FCFE", "usd_bi", MILHAO_PARA_BILHAO),
    "fcff": Metrica("FCFF", "usd_bi", MILHAO_PARA_BILHAO),
    "buyback": Metrica("Recompra de ações", "usd_bi", MILHAO_PARA_BILHAO),
    "dpa": Metrica("Dividendos por ação", "usd"),
    "payout": Metrica("Payout", "pct"),
}

RANKING = {"lp": "maior", "pl": "menor", "ev_ebitda": "menor", "roe": "maior", "roic": "maior", "dl_ebitda": "menor", "dl_pl": "menor"}
SO_POSITIVOS = ("pl", "ev_ebitda")

# Alguns arquivos antigos trazem caracteres acentuados corrompidos. A normalização
# abaixo identifica a coluna pelo conteúdo e produz nomes internos estáveis.
COLUNAS = {
    "p/l damodaran": "pl_damodaran", "p/l": "pl", "l/p": "lp", "ebitda": "ebitda",
    "margem liquida": "margem_liquida", "p/vpa": "pvp", "valor de mercado": "valor_mercado",
    "divida bruta": "divida_bruta", "caixa e equivalentes": "caixa", "divida liquida": "divida_liquida",
    "divida liquida/ebitda": "dl_ebitda", "divida liquida/pl": "dl_pl", "ev/ebitda": "ev_ebitda",
    "roe": "roe", "roic": "roic", "fco": "fco", "fci": "fci", "fcf": "fcf",
    "free cash flow": "free_cash_flow", "capex": "capex", "net capex": "net_capex",
    "r&d": "rd", "adj net capex": "adj_net_capex", "working capital": "working_capital",
    "reinvestment rate": "reinvestment_rate", "fcfe": "fcfe", "fcff": "fcff", "buyback": "buyback",
    "dpa": "dpa", "payout": "payout",
}


def nome_setor(setor: str) -> str:
    return SETORES.get(setor, setor.replace("_", " ").capitalize())


def nome_empresa(pasta: str) -> str:
    return pasta.replace("_", " ").title()


@st.cache_data(show_spinner=False)
def mapear(raiz: str) -> dict:
    """Devolve {setor: {empresa: {tipo: caminho}}} a partir dos arquivos Excel."""
    catalogo: dict = {}
    for arquivo in sorted(Path(raiz).glob("*/*/*_indicators.xlsx")):
        setor, empresa = arquivo.parent.parent.name, arquivo.parent.name
        catalogo.setdefault(setor, {})[empresa] = str(arquivo)
    return catalogo


def _chave_coluna(nome: object) -> str:
    return unicodedata.normalize("NFKD", str(nome)).encode("ascii", "ignore").decode().lower().strip()


@st.cache_data(show_spinner=False)
def carregar(caminho: str, aba: str, _mtime: float) -> pd.DataFrame:
    """Lê uma aba de indicadores e prepara períodos e nomes internos de colunas."""
    bruto = pd.read_excel(caminho, sheet_name=aba)
    periodo_bruto = bruto.iloc[:, 0]
    renomear = {c: COLUNAS[_chave_coluna(c)] for c in bruto.columns if _chave_coluna(c) in COLUNAS}
    df = bruto.rename(columns=renomear).copy()
    if aba == "indicators_10k":
        df["periodo"] = periodo_bruto.astype("Int64").astype(str)
        df = df.sort_values("periodo")
    else:
        datas = pd.to_datetime(periodo_bruto, errors="coerce")
        df["periodo"] = datas.dt.year.astype("Int64").astype(str) + "T" + datas.dt.quarter.astype("Int64").astype(str)
        df["_data"] = datas
        df = df.sort_values("_data")
    return df.reset_index(drop=True)


def escala(df: pd.DataFrame, coluna: str) -> pd.Series:
    return pd.to_numeric(df[coluna], errors="coerce") / METRICAS[coluna].divisor


def numero(valor: float) -> str:
    if valor is None or pd.isna(valor):
        return "—"
    return f"{valor:,.2f}"


def fmt(valor: float, tipo: str) -> str:
    texto = numero(valor)
    if texto == "—":
        return texto
    return {"usd_bi": f"US$ {texto} bi", "pct": f"{texto}%", "x": f"{texto}x", "usd": f"US$ {texto}"}[tipo]


def _layout(fig: go.Figure, cores: dict, titulo: str, unidade: str, n_series: int) -> go.Figure:
    margem_inferior = 58 if n_series >= 2 else 8
    fig.update_layout(title={'text': titulo, 'font': {'size': 15, 'color': cores["texto"]}, 'x': 0, 'xanchor': "left"}, height=360,
                      margin={'l': 8, 'r': 64, 't': 56, 'b': margem_inferior}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font={'color': cores["texto_2"], 'size': 12}, hovermode="x unified", showlegend=n_series >= 2,
                      legend={'orientation': "h", 'yanchor': "top", 'y': -0.2, 'xanchor': "left", 'x': 0, 'bgcolor': "rgba(0,0,0,0)", 'font': {'color': cores["texto_2"]}}, bargap=0.35)
    fig.update_xaxes(showgrid=False, linecolor=cores["grade"], tickcolor=cores["grade"], ticks="outside", type="category")
    fig.update_yaxes(showgrid=True, gridcolor=cores["grade"], zeroline=True, zerolinecolor=cores["grade"], zerolinewidth=1,
                     title={'text': unidade, 'font': {'size': 11, 'color': cores["texto_2"]}})
    return fig


def linhas(df: pd.DataFrame, colunas: list[str], titulo: str, cores: dict) -> go.Figure:
    fig = go.Figure()
    for i, coluna in enumerate(colunas):
        metrica, valores = METRICAS[coluna], escala(df, coluna)
        fig.add_trace(go.Scatter(x=df["periodo"], y=valores, name=metrica.label, mode="lines+markers",
                                 line={'color': cores["serie"][i], 'width': 2}, marker={'size': 8, 'color': cores["serie"][i]},
                                 hovertemplate=f"{metrica.label}: %{{customdata}}<extra></extra>", customdata=[fmt(v, metrica.tipo) for v in valores]))
    return _layout(fig, cores, titulo, METRICAS[colunas[0]].unidade, len(colunas))


def barras(df: pd.DataFrame, colunas: list[str], titulo: str, cores: dict) -> go.Figure:
    fig, uma_serie = go.Figure(), len(colunas) == 1
    for i, coluna in enumerate(colunas):
        metrica, valores = METRICAS[coluna], escala(df, coluna)
        cor = [cores["positivo"] if v >= 0 else cores["negativo"] for v in valores.fillna(0)] if uma_serie else cores["serie"][i]
        fig.add_trace(go.Bar(x=df["periodo"], y=valores, name=metrica.label, marker={'color': cor, 'line': {'width': 0}},
                             text=[numero(v) for v in valores] if uma_serie and len(df) <= 12 else None, textposition="outside", cliponaxis=False,
                             hovertemplate=f"{metrica.label}: %{{customdata}}<extra></extra>", customdata=[fmt(v, metrica.tipo) for v in valores]))
    fig.update_layout(barmode="group")
    return _layout(fig, cores, titulo, METRICAS[colunas[0]].unidade, len(colunas))


def bloco(df: pd.DataFrame, cores: dict, graficos: list) -> None:
    graficos = [g for g in graficos if g[1]]
    if not graficos:
        st.info("Nenhum indicador desta seção está disponível neste arquivo.")
        return
    for inicio in range(0, len(graficos), 2):
        for coluna_st, (desenhar, colunas, titulo) in zip(st.columns(2), graficos[inicio:inicio + 2]):
            with coluna_st:
                st.plotly_chart(desenhar(df, colunas, titulo, cores), width="stretch", key=titulo)


def periodos_do_catalogo(catalogo: dict, tipo: str, setores: list[str]) -> list[str]:
    periodos: set[str] = set()
    for setor in setores:
        for caminho in catalogo.get(setor, {}).values():
            df = carregar(caminho, tipo, Path(caminho).stat().st_mtime)
            periodos.update(df["periodo"])
    return sorted(periodos, reverse=True)


def comparativo(catalogo: dict, tipo: str, setores: list[str], periodo: str | None) -> pd.DataFrame:
    registros = []
    for setor in setores:
        for empresa, caminho in sorted(catalogo.get(setor, {}).items()):
            df = carregar(caminho, tipo, Path(caminho).stat().st_mtime)
            recorte = df if periodo is None else df[df["periodo"] == periodo]
            if recorte.empty:
                continue
            linha = recorte.iloc[-1]
            registro = {"Empresa": nome_empresa(empresa), "Setor": nome_setor(setor), "Período": linha["periodo"]}
            for coluna in RANKING:
                registro[coluna] = escala(recorte, coluna).iloc[-1] if coluna in df.columns else float("nan")
            registros.append(registro)
    return pd.DataFrame(registros, columns=["Empresa", "Setor", "Período", *RANKING])


def ranking_barras(dados: pd.DataFrame, coluna: str, cores: dict, destaque: str) -> go.Figure:
    metrica, serie = METRICAS[coluna], dados.dropna(subset=[coluna]).copy()
    if coluna in SO_POSITIVOS:
        serie = serie[serie[coluna] > 0]
    serie = serie.sort_values(coluna, ascending=RANKING[coluna] == "menor")
    sentido = "menor é melhor" if RANKING[coluna] == "menor" else "maior é melhor"
    fig = go.Figure(go.Bar(x=serie[coluna], y=serie["Empresa"], orientation="h",
                           marker={'color': [cores["serie"][0] if empresa == destaque else cores["neutro"] for empresa in serie["Empresa"]]},
                           text=[numero(v) for v in serie[coluna]], textposition="outside", cliponaxis=False,
                           customdata=serie[["Setor", "Período"]].values,
                           hovertemplate=f"<b>%{{y}}</b><br>{metrica.label}: %{{text}}<br>%{{customdata[0]}} · %{{customdata[1]}}<extra></extra>"))
    fig.update_layout(title={'text': f"{metrica.label} · {sentido}", 'font': {'size': 15, 'color': cores["texto"]}, 'x': 0, 'xanchor': "left"},
                      height=max(260, 26 * len(serie) + 90), margin={'l': 8, 'r': 64, 't': 52, 'b': 8}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font={'color': cores["texto_2"], 'size': 12}, showlegend=False, bargap=0.3)
    fig.update_xaxes(showgrid=True, gridcolor=cores["grade"], zeroline=True, zerolinecolor=cores["grade"], title=metrica.unidade)
    fig.update_yaxes(type="category", autorange="reversed", showgrid=False)
    return fig


# ---------------------------------------------------------------------------
# Valuation e preco de mercado
# ---------------------------------------------------------------------------
def arquivo_valuation_mais_recente(empresa: str) -> Path | None:
    """Localiza o Excel mais recente pelo prefixo MMYYYY de seu nome."""
    pasta = Path(RAIZ_VALUATIONS) / empresa
    candidatos = []
    for arquivo in pasta.glob("*.xlsx"):
        try:
            data = datetime.strptime(arquivo.name[:6], "%m%Y")  # noqa: DTZ007
        except ValueError:
            continue
        candidatos.append((data, arquivo.name, arquivo))
    return max(candidatos, default=(None, None, None))[2]


def _numero_da_celula(valor: object) -> float | None:
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        return None
    return numero if pd.notna(numero) else None


def _valor_valuation(caminho: Path) -> float | None:
    """Le B33 e usa a linha de valor por acao em modelos com outro layout."""
    livro = openpyxl.load_workbook(caminho, read_only=True, data_only=True)
    if "Valuation output" not in livro.sheetnames:
        return None
    aba = livro["Valuation output"]

    # Layout padrao dos modelos de valuation existentes.
    valor_b33 = _numero_da_celula(aba["B33"].value)
    if valor_b33 is not None:
        return valor_b33

    # O modelo da NVIDIA separa os fluxos de caixa por negocio. Nele, B33 e o
    # valor presente do segmento de IA; o valuation por acao fica na linha
    # identificada como "Estimated value /share" (atualmente B52).
    for rotulo, valor in aba.iter_rows(min_col=1, max_col=2, values_only=True):
        texto = unicodedata.normalize("NFKD", str(rotulo or "")).encode("ascii", "ignore").decode().lower()
        if "estimated value" in texto and "share" in texto:
            return _numero_da_celula(valor)
    return None


def _aguardar_arquivo(caminho: Path, segundos: int = 12) -> bool:
    limite = time.monotonic() + segundos
    while time.monotonic() < limite:
        if caminho.is_file() and caminho.stat().st_size > 0:
            return True
        time.sleep(0.25)
    return False


def _recalcular_b33(caminho: Path) -> float | None:
    """Recalcula uma copia do workbook no LibreOffice, sem tocar no arquivo fonte."""
    executavel = shutil.which("soffice")
    if not executavel:
        padrao = Path(r"C:\Program Files\LibreOffice\program\soffice.exe")
        executavel = str(padrao) if padrao.is_file() else None
    if not executavel:
        return None

    # O LibreOffice nao consegue gravar no diretorio de permissoes restritas criado
    # por TemporaryDirectory no Windows. Este cache herda as permissoes do Temp.
    temp = Path(tempfile.gettempdir()) / "analise-fundamentalista-valuation" / f"{caminho.stem}-{caminho.stat().st_mtime_ns}"
    temp.mkdir(parents=True, exist_ok=True)
    ods = temp / f"{caminho.stem}.ods"
    xlsx = temp / f"{caminho.stem}.xlsx"
    if xlsx.is_file():
        return _valor_valuation(xlsx)

    for origem, destino, perfil in ((caminho, ods, f"perfil-ods-{time.time_ns()}"), (ods, xlsx, f"perfil-xlsx-{time.time_ns()}")):
        perfil_uri = "file:///" + str(temp / perfil).replace("\\", "/")
        formato = "ods" if destino.suffix == ".ods" else "xlsx"
        try:
            subprocess.run(
                [executavel, f"-env:UserInstallation={perfil_uri}", "--headless", "--convert-to", formato,
                 "--outdir", str(temp), str(origem)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=90, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if not _aguardar_arquivo(destino):
            return None
    return _valor_valuation(xlsx)


@st.cache_data(show_spinner=False)
def carregar_valuation(caminho: str, _mtime: float) -> float | None:
    """Le o valuation por acao; recorre ao LibreOffice se o Excel nao tiver resultado salvo."""
    arquivo = Path(caminho)
    return _valor_valuation(arquivo) or _recalcular_b33(arquivo)


@st.cache_data(ttl=3600, show_spinner=False)
def carregar_ohlc(ticker: str, periodo: str) -> pd.DataFrame:
    """Baixa OHLC diario para o candlestick, com cache local de uma hora."""
    import yfinance as yf

    yf.cache.set_cache_location(str(Path(tempfile.gettempdir()) / "analise-fundamentalista-yfinance"))
    colunas = ["Open", "High", "Low", "Close"]
    try:
        dados = yf.download(ticker, period=periodo, interval="1d", auto_adjust=False,
                            multi_level_index=False, progress=False)
    except (KeyError, RequestsError, ValueError, YFException):
        return pd.DataFrame(columns=colunas)
    return dados.dropna(subset=colunas)[colunas] if set(colunas).issubset(dados.columns) else pd.DataFrame(columns=colunas)


@st.cache_data(ttl=3600, show_spinner=False)
def carregar_precos_fechamento(tickers: tuple[str, ...], periodo: str) -> pd.DataFrame:
    """Baixa os fechamentos ajustados dos ativos selecionados em uma unica consulta."""
    import yfinance as yf

    yf.cache.set_cache_location(str(Path(tempfile.gettempdir()) / "analise-fundamentalista-yfinance"))
    try:
        dados = yf.download(list(tickers), period=periodo, interval="1d", auto_adjust=True,
                            multi_level_index=False, progress=False)
    except (KeyError, RequestsError, ValueError, YFException):
        return pd.DataFrame(columns=list(tickers))

    if dados.empty:
        return pd.DataFrame(columns=list(tickers))
    if isinstance(dados.columns, pd.MultiIndex):
        if "Close" in dados.columns.get_level_values(0):
            fechamento = dados["Close"]
        elif "Close" in dados.columns.get_level_values(1):
            fechamento = dados.xs("Close", axis=1, level=1)
        else:
            return pd.DataFrame(columns=list(tickers))
    else:
        if "Close" not in dados.columns:
            return pd.DataFrame(columns=list(tickers))
        fechamento = dados["Close"]

    if isinstance(fechamento, pd.Series):
        fechamento = fechamento.to_frame(name=tickers[0])
    return fechamento.reindex(columns=list(tickers)).dropna(how="all")


def linhas_mercado(dados: pd.DataFrame, titulo: str, unidade: str, cores: dict,
                   formato: str = "usd") -> go.Figure:
    """Desenha linhas de mercado com o mesmo estilo dos demais graficos do painel."""
    fig = go.Figure()
    for indice, ticker in enumerate(dados.columns):
        valores = dados[ticker]
        texto_hover = ([f"{valor:.2%}" if pd.notna(valor) else "—" for valor in valores]
                       if formato == "pct"
                       else [f"US$ {valor:,.2f}" if pd.notna(valor) else "—" for valor in valores])
        fig.add_trace(go.Scatter(
            x=dados.index, y=valores, name=ticker, mode="lines",
            line={'color': cores["serie"][indice], 'width': 2},
            hovertemplate=f"<b>{ticker}</b>: %{{customdata}}<extra></extra>",
            customdata=texto_hover,
        ))
    fig = _layout(fig, cores, titulo, unidade, len(dados.columns))
    fig.update_xaxes(type="date")
    if formato == "pct":
        fig.update_yaxes(tickformat=".0%")
    return fig


def grafico_retornos_logaritmicos(precos: pd.DataFrame) -> go.Figure:
    """Reutiliza o subplot de histogramas e retornos logaritmicos do modulo comum."""
    try:
        from funcoes_eua import plot_indicators_subplot_histogram
    except ModuleNotFoundError:
        from analise_eua.funcoes_eua import plot_indicators_subplot_histogram
    return plot_indicators_subplot_histogram(precos, "acoes selecionadas")


def coeficiente_hurst(serie: pd.Series) -> float:
    """Calcula o expoente de Hurst usando a funcao comum do modulo dos EUA."""
    try:
        from funcoes_eua import hurst_exponent
    except ModuleNotFoundError:
        from analise_eua.funcoes_eua import hurst_exponent

    valores = serie.dropna().to_numpy(dtype=float)
    max_lag = min(100, len(valores) // 2)
    if max_lag < 3:
        return float("nan")
    try:
        resultado = hurst_exponent(valores, max_lag)
        return float(resultado) if np.isfinite(resultado) else float("nan")
    except (ValueError, TypeError, FloatingPointError):
        return float("nan")


def tabela_risco_retorno(precos: pd.DataFrame, preco_mercado: pd.Series,
                          alpha: float = 0.05) -> pd.DataFrame:
    """Calcula indicadores diarios de risco e retorno para cada ativo selecionado."""
    retorno_mercado = preco_mercado.pct_change().dropna()
    registros = []
    for ticker in precos.columns:
        serie = precos[ticker].dropna()
        retornos = serie.pct_change().dropna()
        alinhados = pd.concat([retornos.rename("ativo"), retorno_mercado.rename("mercado")], axis=1).dropna()
        variancia_mercado = alinhados["mercado"].var()
        beta = (alinhados["ativo"].cov(alinhados["mercado"]) / variancia_mercado
                if len(alinhados) > 1 and variancia_mercado else float("nan"))

        anos = (serie.index[-1] - serie.index[0]).days / 365.25 if len(serie) > 1 else 0
        cagr = (serie.iloc[-1] / serie.iloc[0]) ** (1 / anos) - 1 if anos > 0 and serie.iloc[0] > 0 else float("nan")
        maior_drawdown = (serie / serie.cummax() - 1).min()
        hurst = coeficiente_hurst(serie)
        var = retornos.quantile(alpha)
        cvar = retornos[retornos <= var].mean()

        try:
            graus_liberdade, loc, escala_t = student_t.fit(retornos)
            quantil_t = student_t.ppf(alpha, graus_liberdade)
            var_t = loc + escala_t * quantil_t
            cvar_t = (loc - escala_t * (graus_liberdade + quantil_t ** 2)
                      * student_t.pdf(quantil_t, graus_liberdade) / (alpha * (graus_liberdade - 1)))
            if graus_liberdade <= 1:
                cvar_t = float("nan")
        except (ValueError, FloatingPointError):
            var_t, cvar_t = float("nan"), float("nan")

        registros.append({
            "Ação": ticker,
            "Beta": beta,
            "Hurst": hurst,
            "CAGR": cagr * 100,
            "Maior drawdown": maior_drawdown * 100,
            "VaR": var * 100,
            "CVaR": cvar * 100,
            "VaR (student-t)": var_t * 100,
            "CVaR (student-t)": cvar_t * 100,
        })
    return pd.DataFrame(registros)


def heatmap_retornos_anuais(precos: pd.DataFrame, cores: dict) -> go.Figure | None:
    """Mostra o retorno de cada ano-calendario em uma matriz de acoes e anos."""
    fechamentos_anuais = precos.resample("YE").last()
    retornos_anuais = (fechamentos_anuais.pct_change().iloc[1:] * 100).dropna(how="all")
    if retornos_anuais.empty:
        return None

    matriz = retornos_anuais.T
    anos = [str(data.year) for data in matriz.columns]
    texto = [[f"{valor:.1f}%" if pd.notna(valor) else "" for valor in linha] for linha in matriz.to_numpy()]
    fig = go.Figure(go.Heatmap(
        z=matriz.to_numpy(), x=anos, y=matriz.index, text=texto, texttemplate="%{text}",
        colorscale=[[0, cores["negativo"]], [0.5, cores["grade"]], [1, cores["positivo"]]],
        zmid=0, colorbar={'title': "%"},
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        title={'text': "Heatmap dos retornos anuais", 'font': {'size': 15, 'color': cores["texto"]}, 'x': 0, 'xanchor': "left"},
        height=max(260, 44 * len(matriz) + 145), margin={'l': 8, 'r': 64, 't': 56, 'b': 8},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={'color': cores["texto_2"], 'size': 12},
    )
    fig.update_xaxes(side="bottom", type="category")
    fig.update_yaxes(type="category", autorange="reversed")
    return fig


def heatmap_correlacao(precos: pd.DataFrame, cores: dict) -> go.Figure | None:
    """Mostra a correlacao dos retornos diarios entre as acoes selecionadas."""
    correlacao = precos.pct_change().corr().dropna(how="all").dropna(how="all", axis=1)
    if correlacao.empty:
        return None

    texto = [[f"{valor:.2f}" if pd.notna(valor) else "" for valor in linha] for linha in correlacao.to_numpy()]
    fig = go.Figure(go.Heatmap(
        z=correlacao.to_numpy(), x=correlacao.columns, y=correlacao.index, text=texto, texttemplate="%{text}",
        colorscale=[[0, cores["negativo"]], [0.5, cores["grade"]], [1, cores["positivo"]]],
        zmin=-1, zmax=1, zmid=0, colorbar={'title': "Correlação"},
        hovertemplate="<b>%{y} × %{x}</b><br>Correlação: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title={'text': "Heatmap de correlação", 'font': {'size': 15, 'color': cores["texto"]}, 'x': 0, 'xanchor': "left"},
        height=max(300, 48 * len(correlacao) + 145), margin={'l': 8, 'r': 86, 't': 56, 'b': 8},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={'color': cores["texto_2"], 'size': 12},
    )
    fig.update_xaxes(side="bottom", type="category")
    fig.update_yaxes(type="category", autorange="reversed")
    return fig


def grafico_risco_retorno(precos: pd.DataFrame, tabela: pd.DataFrame, cores: dict) -> go.Figure:
    """Relaciona volatilidade anualizada e CAGR das acoes selecionadas."""
    volatilidade = precos.pct_change().std() * (252 ** 0.5) * 100
    dados = pd.DataFrame({
        "Ação": tabela.iloc[:, 0],
        "Volatilidade anualizada": tabela.iloc[:, 0].map(volatilidade),
        "CAGR": tabela["CAGR"],
    }).dropna()
    fig = go.Figure()
    for indice, linha in dados.reset_index(drop=True).iterrows():
        fig.add_trace(go.Scatter(
            x=[linha["Volatilidade anualizada"]], y=[linha["CAGR"]], mode="markers+text",
            name=linha["Ação"], text=[linha["Ação"]], textposition="top center",
            marker={'size': 13, 'color': cores["serie"][indice], 'line': {'color': cores["texto"], 'width': 1}},
            hovertemplate=(f"<b>{linha['Ação']}</b><br>Volatilidade anualizada: "
                           "%{x:.2f}%<br>CAGR: %{y:.2f}%<extra></extra>"),
        ))
    fig.update_layout(
        title={'text': "Risco e retorno", 'font': {'size': 15, 'color': cores["texto"]}, 'x': 0, 'xanchor': "left"},
        height=430, margin={'l': 8, 'r': 24, 't': 56, 'b': 8}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={'color': cores["texto_2"], 'size': 12}, showlegend=False,
    )
    fig.update_xaxes(title="Volatilidade anualizada (%)", showgrid=True, gridcolor=cores["grade"], zeroline=False)
    fig.update_yaxes(title="CAGR (%)", showgrid=True, gridcolor=cores["grade"], zeroline=True,
                     zerolinecolor=cores["negativo"], zerolinewidth=1)
    return fig


def candlestick(precos: pd.DataFrame, ticker: str, valor: float, arquivo: Path, cores: dict) -> go.Figure:
    fig = go.Figure(go.Candlestick(
        x=precos.index, open=precos["Open"], high=precos["High"], low=precos["Low"], close=precos["Close"],
        name=ticker, increasing_line_color=cores["positivo"], decreasing_line_color=cores["negativo"],
    ))
    fig.add_hline(y=valor, line={'color': cores["serie"][1], 'width': 2, 'dash': "dash"},
                  annotation_text=f"Valuation: US$ {valor:,.2f}", annotation_position="top left",
                  annotation_font_color=cores["serie"][1])
    fig.update_layout(
        title={'text': f"{ticker} - preco de mercado e valuation", 'font': {'size': 15, 'color': cores["texto"]}, 'x': 0, 'xanchor': "left"},
        height=520, margin={'l': 8, 'r': 64, 't': 56, 'b': 8}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={'color': cores["texto_2"], 'size': 12}, hovermode="x unified", showlegend=False,
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(showgrid=False, linecolor=cores["grade"])
    fig.update_yaxes(showgrid=True, gridcolor=cores["grade"], zeroline=False, title="US$ por acao")
    fig.add_annotation(xref="paper", yref="paper", x=0, y=-0.14, showarrow=False, xanchor="left",
                       text=f"Valuation obtido de {arquivo.name}, na aba Valuation output.",
                       font={'size': 11, 'color': cores["texto_2"]})
    return fig


st.set_page_config(page_title="Indicadores fundamentalistas — EUA", page_icon="📊", layout="wide")
tema = getattr(st.context.theme, "type", None) or "light"
cores = PALETA.get(tema, PALETA["light"])

st.sidebar.header("Empresa")
raiz = st.sidebar.text_input("Diretório dos indicadores", value=RAIZ_PADRAO)
if not Path(raiz).is_dir():
    st.error(f"Diretório não encontrado: `{raiz}`")
    st.stop()
catalogo = mapear(raiz)
if not catalogo:
    st.error(f"Nenhum arquivo no padrão `<setor>/<empresa>/<ticker>_indicators.xlsx` em `{raiz}`.")
    st.stop()

setor = st.sidebar.selectbox("Setor", sorted(catalogo), format_func=nome_setor)
empresa = st.sidebar.selectbox("Empresa", sorted(catalogo[setor]), format_func=nome_empresa)
caminho = catalogo[setor][empresa]
tipo = st.sidebar.radio("Periodicidade", ["indicators_10k", "indicators_10q"], horizontal=True,
                        format_func=lambda valor: "Anual (10-K)" if valor == "indicators_10k" else "Trimestral (10-Q)")
if st.sidebar.button("Atualizar lista de empresas", width="stretch"):
    mapear.clear()
    carregar.clear()
    st.rerun()

arquivo = Path(caminho)
try:
    df = carregar(caminho, tipo, arquivo.stat().st_mtime)
except ValueError:
    st.error(f"A aba `{tipo}` não foi encontrada em `{arquivo.name}`.")
    st.stop()
disponiveis = [coluna for coluna in METRICAS if coluna in df.columns]

st.sidebar.header("Período")
periodos = df["periodo"].tolist()
inicio, fim = st.sidebar.select_slider("Intervalo", options=periodos, value=(periodos[0], periodos[-1]), label_visibility="collapsed")
df = df[(df["periodo"] >= inicio) & (df["periodo"] <= fim)].reset_index(drop=True)

titulo_empresa = nome_empresa(empresa)
st.title(f"{titulo_empresa} — indicadores fundamentalistas")
st.caption(f"{nome_setor(setor)} · {'anual (10-K)' if tipo == 'indicators_10k' else 'trimestral (10-Q)'} · {len(df)} períodos "
           f"({df['periodo'].iloc[0]} a {df['periodo'].iloc[-1]}) · valores monetários em US$ bilhões · {arquivo.name}")

st.subheader(f"Último período · {df['periodo'].iloc[-1]}")
destaques = [c for c in ("pl", "pvp", "ev_ebitda", "roe", "roic", "margem_liquida", "dl_ebitda", "payout") if c in disponiveis]
for faixa in (destaques[:4], destaques[4:]):
    if faixa:
        for coluna_st, coluna in zip(st.columns(len(faixa)), faixa):
            metrica, valores = METRICAS[coluna], escala(df, coluna)
            atual, anterior = valores.iloc[-1], valores.iloc[-2] if len(valores) > 1 else None
            delta = None if anterior is None or pd.isna(atual) or pd.isna(anterior) else atual - anterior
            coluna_st.metric(metrica.label, fmt(atual, metrica.tipo), None if delta is None else fmt(delta, metrica.tipo), help=metrica.ajuda or None)

abas = st.tabs(["Valuation", "DCF Valuation", "Rentabilidade", "Endividamento", "Fluxo de caixa", "Proventos", "Explorar", "Dados", "Ranking", "Indicadores gerais"])

def existentes(*colunas: str) -> list[str]:
    return [coluna for coluna in colunas if coluna in disponiveis]

with abas[0]:
    bloco(df, cores, [(linhas, existentes("pl_damodaran", "pl", "pvp", "ev_ebitda"), "Múltiplos de mercado"),
                       (linhas, existentes("lp"), "Earnings yield (L/P)"),
                       (barras, existentes("valor_mercado"), "Valor de mercado")])

with abas[1]:
    arquivo_valuation = arquivo_valuation_mais_recente(empresa)
    st.subheader("Preco de mercado e valuation")
    if arquivo_valuation is None:
        st.info("Nao ha arquivo de valuation para esta empresa no diretorio de valuations.")
    else:
        st.caption(f"A linha horizontal usa o valuation do arquivo **{arquivo_valuation.name}**.")
        with st.spinner("Lendo o valuation mais recente..."):
            valor_valuation = carregar_valuation(str(arquivo_valuation), arquivo_valuation.stat().st_mtime)
        if valor_valuation is None:
            st.warning(f"Nao foi possivel obter o valuation por acao de `{arquivo_valuation.name}`.")
        else:
            opcoes_candlestick = {"6 meses": "6mo", "1 ano": "1y", "2 anos": "2y"}
            intervalo_candlestick = st.selectbox("Historico de precos", list(opcoes_candlestick), index=1)
            ticker_acao = arquivo.stem.rsplit("_", 1)[0].upper()
            with st.spinner(f"Baixando precos de {ticker_acao}..."):
                precos = carregar_ohlc(ticker_acao, opcoes_candlestick[intervalo_candlestick])
            if precos.empty:
                st.warning(f"Nao foi possivel obter as cotacoes OHLC de {ticker_acao}.")
            else:
                st.plotly_chart(candlestick(precos, ticker_acao, valor_valuation, arquivo_valuation, cores), width="stretch")

with abas[2]:
    bloco(df, cores, [(linhas, existentes("roe", "roic"), "ROE e ROIC"),
                       (linhas, existentes("margem_liquida"), "Margem líquida"),
                       (barras, existentes("ebitda"), "EBITDA"),
                       (linhas, existentes("reinvestment_rate"), "Taxa de reinvestimento")])
with abas[3]:
    bloco(df, cores, [(barras, existentes("divida_bruta", "caixa", "divida_liquida"), "Dívida bruta, caixa e dívida líquida"),
                       (linhas, existentes("dl_ebitda", "dl_pl"), "Alavancagem")])
with abas[4]:
    bloco(df, cores, [(barras, existentes("fco", "fci", "fcf"), "Fluxos de caixa (FCO, FCI, FCF)"),
                       (barras, existentes("free_cash_flow"), "Free cash flow"),
                       (barras, existentes("capex", "net_capex", "rd", "adj_net_capex"), "Capex e P&D"),
                       (barras, existentes("fcfe", "fcff"), "FCFE e FCFF"),
                       (barras, existentes("working_capital"), "Capital de giro")])
with abas[5]:
    bloco(df, cores, [(barras, existentes("buyback"), "Recompra de ações"),
                       (linhas, existentes("dpa"), "Dividendos por ação"),
                       (linhas, existentes("payout"), "Payout")])
with abas[6]:
    escolhidos = st.multiselect("Indicadores (combine somente indicadores da mesma unidade)", options=disponiveis, default=disponiveis[:1], format_func=lambda c: f"{METRICAS[c].label} ({METRICAS[c].unidade})")
    unidades = {METRICAS[c].unidade for c in escolhidos}
    if not escolhidos:
        st.info("Escolha ao menos um indicador.")
    elif len(unidades) > 1:
        st.warning(f"Unidades diferentes selecionadas ({', '.join(sorted(unidades))}). Escolha indicadores da mesma unidade.")
    else:
        forma = st.radio("Forma", ["Linha", "Barra"], horizontal=True, label_visibility="collapsed")
        st.plotly_chart((linhas if forma == "Linha" else barras)(df, escolhidos[:8], " · ".join(METRICAS[c].label for c in escolhidos[:8]), cores), width="stretch")
with abas[7]:
    tabela = pd.DataFrame({"Período": df["periodo"]})
    for coluna in disponiveis:
        tabela[f"{METRICAS[coluna].label} ({METRICAS[coluna].unidade})"] = escala(df, coluna).round(2)
    st.dataframe(tabela.set_index("Período").T, width="stretch")
    st.download_button("Baixar CSV filtrado", df.drop(columns=["_data"], errors="ignore").to_csv(index=False).encode("utf-8"), file_name=f"{arquivo.stem}_{tipo}.csv", mime="text/csv")
with abas[8]:
    st.caption(f"Compara as empresas pela aba **{tipo}**. {titulo_empresa} aparece destacado em azul nos gráficos.")
    col_setores, col_periodo = st.columns([3, 2])
    setores_sel = col_setores.multiselect("Setores", sorted(catalogo), default=sorted(catalogo), format_func=nome_setor)
    modo = col_periodo.radio("Período de comparação", ["Último de cada empresa", "Um período específico"])
    periodo_ranking = None
    if modo == "Um período específico":
        opcoes = periodos_do_catalogo(catalogo, tipo, setores_sel)
        if opcoes:
            periodo_ranking = col_periodo.selectbox("Período", opcoes)
    dados = comparativo(catalogo, tipo, setores_sel, periodo_ranking)
    if dados.empty:
        st.info("Selecione ao menos um setor com empresas no período escolhido.")
    else:
        if periodo_ranking is None and dados["Período"].nunique() > 1:
            st.warning("As empresas têm últimos períodos diferentes. Para comparar a mesma data, escolha um período específico.")
        for inicio in range(0, len(RANKING), 2):
            for coluna_st, coluna in zip(st.columns(2), list(RANKING)[inicio:inicio + 2]):
                with coluna_st:
                    st.plotly_chart(ranking_barras(dados, coluna, cores, titulo_empresa), width="stretch", key=f"rank_{coluna}")
        st.subheader("Tabela comparativa")
        st.dataframe(dados.round(2), width="stretch", hide_index=True, column_config={c: st.column_config.NumberColumn(f"{METRICAS[c].label} ({METRICAS[c].unidade})", format="%.2f") for c in RANKING})

with abas[9]:
    acoes = {
        Path(caminho_empresa).stem.removesuffix("_indicators").upper(): nome_empresa(nome_empresa_pasta)
        for empresas in catalogo.values()
        for nome_empresa_pasta, caminho_empresa in empresas.items()
    }
    ticker_atual = arquivo.stem.removesuffix("_indicators").upper()
    tickers = st.multiselect(
        "Ações", options=sorted(acoes), default=[ticker_atual], max_selections=len(cores["serie"]),
        format_func=lambda ticker: (f"{ticker} - {acoes[ticker]}" if ticker in acoes
                                    else f"{str(ticker).upper()} (ticker manual)"),
        accept_new_options=True, placeholder="Selecione ou digite um ticker (ex.: MSFT)",
        key="indicadores_gerais_acoes",
    )
    tickers = list(dict.fromkeys(str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()))
    periodos_mercado = {"2 anos": "2y", "5 anos": "5y", "10 anos": "10y"}
    periodo_mercado = st.selectbox("Histórico de preços", list(periodos_mercado), index=0,
                                  key="indicadores_gerais_periodo")
    if not tickers:
        st.info("Escolha ao menos uma ação.")
    else:
        with st.spinner("Baixando precos das ações selecionadas..."):
            precos_acoes = carregar_precos_fechamento(tuple(tickers), periodos_mercado[periodo_mercado])
        if precos_acoes.empty:
            st.warning("Nao foi possivel obter os precos das ações selecionadas.")
        else:
            st.plotly_chart(linhas_mercado(precos_acoes, "Preços das ações", "US$", cores),
                            width="stretch")
            retorno_acumulado = precos_acoes.apply(
                lambda serie: serie / serie.dropna().iloc[0] - 1 if serie.notna().any() else serie
            ).dropna(how="all")
            grafico_retorno = linhas_mercado(retorno_acumulado, "Retorno acumulado", "%", cores, formato="pct")
            grafico_retorno.add_hline(y=0, line={'color': cores["negativo"], 'width': 1.5})
            st.plotly_chart(grafico_retorno, width="stretch")
            heatmap = heatmap_retornos_anuais(precos_acoes, cores)
            if heatmap is None:
                st.info("O histórico selecionado não contém anos suficientes para calcular os retornos anuais.")
            else:
                st.plotly_chart(heatmap, width="stretch")
            momentum = (precos_acoes - precos_acoes.shift(252)).dropna(how="all")
            if momentum.empty:
                st.info("O historico selecionado nao contem 252 pregoes para calcular o momentum de um ano.")
            else:
                grafico_momentum = linhas_mercado(momentum, "Momentum - 1 ano (252 pregões)", "US$", cores)
                grafico_momentum.add_hline(y=0, line={'color': cores["negativo"], 'width': 1.5})
                st.plotly_chart(grafico_momentum, width="stretch")
            retornos_log = np.log(precos_acoes / precos_acoes.shift(1))
            volatilidade = (retornos_log.rolling(252, min_periods=252).std() * (252 ** 0.5)).dropna(how="all")
            if volatilidade.empty:
                st.info("O historico selecionado nao contem 252 pregoes para calcular a volatilidade anualizada.")
            else:
                st.plotly_chart(
                    linhas_mercado(volatilidade, "Volatilidade anualizada (252 pregões)", "%", cores, formato="pct"),
                    width="stretch",
                )
            volatilidade_movel = (retornos_log.rolling(60, min_periods=60).std() * (252 ** 0.5)).dropna(how="all")
            if volatilidade_movel.empty:
                st.info("O histórico selecionado não contém 60 pregões para calcular a volatilidade móvel.")
            else:
                st.plotly_chart(
                    linhas_mercado(volatilidade_movel, "Volatilidade móvel (60 pregões)", "%", cores, formato="pct"),
                    width="stretch",
                )
            st.plotly_chart(grafico_retornos_logaritmicos(precos_acoes), width="stretch")
            st.subheader("Indicadores de risco e retorno")
            st.caption("Beta contra o S&P 500; CAGR do período selecionado; VaR e CVaR diários a 95% de confiança.")
            with st.spinner("Calculando os indicadores de risco e retorno..."):
                precos_mercado = carregar_precos_fechamento(("^GSPC",), periodos_mercado[periodo_mercado])
            if precos_mercado.empty or precos_mercado["^GSPC"].dropna().empty:
                st.warning("Não foi possível obter os preços do S&P 500 para calcular o beta.")
            else:
                tabela_risco = tabela_risco_retorno(precos_acoes, precos_mercado["^GSPC"])
                st.dataframe(
                    tabela_risco,
                    width="stretch", hide_index=True,
                    column_config={
                        "Beta": st.column_config.NumberColumn("Beta", format="%.2f"),
                        "Hurst": st.column_config.NumberColumn("Hurst", format="%.3f"),
                        "CAGR": st.column_config.NumberColumn("CAGR", format="%.2f%%"),
                        "Maior drawdown": st.column_config.NumberColumn("Maior drawdown", format="%.2f%%"),
                        "VaR": st.column_config.NumberColumn("VaR", format="%.2f%%"),
                        "CVaR": st.column_config.NumberColumn("CVaR", format="%.2f%%"),
                        "VaR (student-t)": st.column_config.NumberColumn("VaR (student-t)", format="%.2f%%"),
                        "CVaR (student-t)": st.column_config.NumberColumn("CVaR (student-t)", format="%.2f%%"),
                    },
                )
                st.caption("Hurst: abaixo de 0,5 indica reversão à média; perto de 0,5, comportamento aleatório; acima de 0,5, persistência de tendência.")
                st.plotly_chart(grafico_risco_retorno(precos_acoes, tabela_risco, cores), width="stretch")
                heatmap_correlacoes = heatmap_correlacao(precos_acoes, cores)
                if heatmap_correlacoes is not None:
                    st.plotly_chart(heatmap_correlacoes, width="stretch")
