"""
Dashboard Streamlit dos indicadores fundamentalistas.

Uso:
    streamlit run analise_brasil/app_indicadores.py

Lê os arquivos gerados em `historico-arquivos/indicadores_fundamentalistas`,
organizados como `<setor>/<ticker>/<ticker>_indicadores_<anual|trimestral>.csv`
(CSV separado por `;`, uma linha por período em `dt_refer`).

Observação sobre unidades: os valores monetários das demonstrações vêm da CVM
em R$ mil; o valor de mercado vem em R$ (unidade). O app converte tudo para
R$ bilhões nos gráficos.
"""

from dataclasses import dataclass
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from curl_cffi.requests import RequestsError
from plotly.subplots import make_subplots
from scipy.stats import norm, t as student_t
from yfinance.exceptions import YFException

RAIZ_PADRAO = r'C:\B3\historico-arquivos\indicadores_fundamentalistas'

# Rótulos dos setores (o nome da pasta é sem acento); o fallback cobre setores novos
SETORES = {
    'aluguel': 'Aluguel',
    'bebida': 'Bebida',
    'celulose': 'Celulose',
    'energia-eletrica': 'Energia elétrica',
    'farmacia': 'Farmácia',
    'maq-equip': 'Máquinas e equipamentos',
    'material-aeronautico': 'Material aeronáutico',
    'mineracao': 'Mineração',
    'petroleo': 'Petróleo',
    'varejo': 'Varejo',
}

# Os diretórios dos indicadores usam o código-base da empresa. Esta tabela
# preserva o ticker completo que foi utilizado para calcular os indicadores e
# que o Yahoo Finance reconhece para consultar os preços da B3.
TICKERS_B3 = {
    'abev': 'ABEV3.SA', 'alup': 'ALUP3.SA', 'axia': 'AXIA3.SA', 'azza': 'AZZA3.SA',
    'ceab': 'CEAB3.SA', 'cmig': 'CMIG3.SA', 'cple': 'CPLE3.SA', 'egie': 'EGIE3.SA',
    'embj': 'EMBJ3.SA', 'enev': 'ENEV3.SA', 'eqtl': 'EQTL3.SA', 'isae': 'ISAE3.SA',
    'klbn': 'KLBN3.SA', 'lren': 'LREN3.SA', 'mils': 'MILS3.SA', 'petr': 'PETR3.SA',
    'prio': 'PRIO3.SA', 'radl': 'RADL3.SA', 'rani': 'RANI3.SA', 'rent': 'RENT3.SA',
    'sbfg': 'SBFG3.SA', 'suzb': 'SUZB3.SA', 'taee': 'TAEE3.SA', 'vale': 'VALE3.SA',
    'viva': 'VIVA3.SA', 'vulc': 'VULC3.SA', 'wege': 'WEGE3.SA',
}

# ---------------------------------------------------------------------------
# Paleta (validada para daltonismo; ordem fixa dos slots, nunca ciclada)
# ---------------------------------------------------------------------------
PALETA = {
    'light': {
        'serie': ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948'],
        'texto': '#0b0b0b',
        'texto_2': '#52514e',
        'grade': '#e5e4e0',
        'positivo': '#2a78d6',
        'negativo': '#e34948',
        'neutro': '#8f8e88',
    },
    'dark': {
        'serie': ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'],
        'texto': '#ffffff',
        'texto_2': '#c3c2b7',
        'grade': '#383835',
        'positivo': '#3987e5',
        'negativo': '#e66767',
        'neutro': '#7c7b76',
    },
}


@dataclass(frozen=True)
class Metrica:
    """Metadados de um indicador: rótulo, unidade e escala de exibição."""

    label: str
    tipo: str          # 'brl' (R$ bi) | 'pct' | 'x' (múltiplo) | 'rs' (R$ por ação)
    divisor: float = 1.0
    ajuda: str = ''

    @property
    def unidade(self) -> str:
        return {'brl': 'R$ bi', 'pct': '%', 'x': 'x', 'rs': 'R$'}[self.tipo]


MIL = 1e6      # R$ mil -> R$ bilhões
UNI = 1e9      # R$      -> R$ bilhões

# Colunas com sufixo `_acum` aparecem só nos arquivos trimestrais (acumulado no ano);
# `_nao_acum` é o valor do trimestre isolado. Nem toda empresa tem todas as colunas.
METRICAS = {
    'valor_mercado': Metrica('Valor de mercado', 'brl', UNI, 'Preço de fechamento x total de ações'),
    'lpa': Metrica('LPA', 'rs', 1, 'Lucro por ação'),
    'lpa_acum': Metrica('LPA acumulado', 'rs', 1, 'Lucro por ação acumulado no ano'),
    'vpa': Metrica('VPA', 'rs', 1, 'Valor patrimonial por ação'),
    'p/l': Metrica('P/L', 'x', 1, 'Preço sobre lucro'),
    'l/p': Metrica('L/P (earnings yield)', 'pct', 1, 'Lucro sobre preço'),
    'p/vp': Metrica('P/VP', 'x', 1, 'Preço sobre valor patrimonial'),
    'ev_ebitda': Metrica('EV/EBITDA', 'x', 1, 'Enterprise value sobre EBITDA'),
    'ebitda': Metrica('EBITDA', 'brl', MIL),
    'ebitda_acum': Metrica('EBITDA acumulado', 'brl', MIL, 'Acumulado no ano'),
    'divida_bruta': Metrica('Dívida bruta', 'brl', MIL),
    'caixa_total': Metrica('Caixa total', 'brl', MIL),
    'divida_liquida': Metrica('Dívida líquida', 'brl', MIL),
    'dl_ebitda': Metrica('Dívida líquida / EBITDA', 'x', 1, 'Alavancagem'),
    'dl_pl': Metrica('Dívida líquida / PL', 'x', 1),
    'margem_liquida': Metrica('Margem líquida', 'pct', 1),
    'roe': Metrica('ROE', 'pct', 1, 'Retorno sobre o patrimônio líquido'),
    'roic': Metrica('ROIC', 'pct', 1, 'Retorno sobre o capital investido'),
    'proventos': Metrica('Proventos', 'brl', MIL, 'Dividendos + JCP'),
    'payout': Metrica('Payout', 'pct', 1, 'Proventos sobre lucro líquido'),
    'buyback': Metrica('Recompra de ações', 'brl', MIL),
    'buyback_nao_acum': Metrica('Recompra de ações no trimestre', 'brl', MIL),
    'fco': Metrica('Fluxo de caixa operacional', 'brl', MIL),
    'fci': Metrica('Fluxo de caixa de investimento', 'brl', MIL),
    'fcf': Metrica('Fluxo de caixa de financiamento', 'brl', MIL),
    'fco_nao_acum': Metrica('FCO no trimestre', 'brl', MIL, 'Fluxo de caixa operacional'),
    'fci_nao_acum': Metrica('FCI no trimestre', 'brl', MIL, 'Fluxo de caixa de investimento'),
    'fcf_nao_acum': Metrica('FCF no trimestre', 'brl', MIL, 'Fluxo de caixa de financiamento'),
    'fco_acum': Metrica('FCO acumulado', 'brl', MIL, 'Acumulado no ano'),
    'fci_acum': Metrica('FCI acumulado', 'brl', MIL, 'Acumulado no ano'),
    'fcf_acum': Metrica('FCF acumulado', 'brl', MIL, 'Acumulado no ano'),
    'capex': Metrica('Capex', 'brl', MIL),
    'capex_acum': Metrica('Capex acumulado', 'brl', MIL, 'Acumulado no ano'),
    'capex_1': Metrica('Capex (conta 1)', 'brl', MIL, 'Conta que compõe o capex'),
    'capex_2': Metrica('Capex (conta 2)', 'brl', MIL, 'Conta que compõe o capex'),
    'capex_3': Metrica('Capex (conta 3)', 'brl', MIL, 'Conta que compõe o capex'),
    'capex_4': Metrica('Capex (conta 4)', 'brl', MIL, 'Conta que compõe o capex'),
    'net_capex': Metrica('Net capex', 'brl', MIL, 'Capex menos depreciação'),
    'rd': Metrica('P&D', 'brl', MIL),
    'rd_acum': Metrica('P&D acumulado', 'brl', MIL, 'Acumulado no ano'),
    'rd_nao_acum': Metrica('P&D no trimestre', 'brl', MIL),
    'adjusted_net_capex': Metrica('Net capex ajustado', 'brl', MIL, 'Net capex mais P&D'),
    'free_cash_flow': Metrica('Free cash flow', 'brl', MIL),
    'fcfe': Metrica('FCFE', 'brl', MIL, 'Fluxo de caixa do acionista'),
    'fcff': Metrica('FCFF', 'brl', MIL, 'Fluxo de caixa da firma'),
    'working_capital': Metrica('Capital de giro', 'brl', MIL),
    'change_in_non_cash_wc': Metrica('Variação do capital de giro', 'brl', MIL),
    'change_in_non_cash_wc_acum': Metrica('Variação do capital de giro acumulada', 'brl', MIL,
                                          'Acumulado no ano'),
    'reinvestment_rate': Metrica('Taxa de reinvestimento', 'pct', 0.01, 'Reinvestimento sobre NOPAT'),
    'effective_tax_rate': Metrica('Alíquota efetiva de IR', 'pct', 0.01),
}


# Indicadores do ranking entre empresas: coluna -> sentido do "melhor"
RANKING = {
    'l/p': 'maior',
    'p/l': 'menor',
    'ev_ebitda': 'menor',
    'roe': 'maior',
    'roic': 'maior',
    'dl_ebitda': 'menor',
    'dl_pl': 'menor',
}

# Múltiplos que não fazem sentido com numerador/denominador negativo (prejuízo,
# EBITDA negativo): a empresa sai do ranking em vez de aparecer como "a mais barata".
SO_POSITIVOS = ('p/l', 'ev_ebitda')


# ---------------------------------------------------------------------------
# Catálogo de arquivos
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def mapear(raiz: str) -> dict:
    """Varre `<raiz>/<setor>/<ticker>/` e devolve {setor: {TICKER: {tipo: caminho}}}."""
    catalogo: dict = {}
    for csv in sorted(Path(raiz).glob('*/*/*_indicadores_*.csv')):
        tipo = csv.stem.rsplit('_', 1)[-1].lower()             # anual | trimestral
        setor, ticker = csv.parent.parent.name, csv.parent.name.upper()
        catalogo.setdefault(setor, {}).setdefault(ticker, {})[tipo] = str(csv)
    return catalogo


def nome_setor(setor: str) -> str:
    """Rótulo legível do setor a partir do nome da pasta."""
    return SETORES.get(setor, setor.replace('-', ' ').capitalize())


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def carregar(caminho: str, _mtime: float) -> pd.DataFrame:
    """Lê o CSV de indicadores e ordena por período (`_mtime` invalida o cache)."""
    df = pd.read_csv(caminho, sep=';', decimal='.')
    df['dt_refer'] = pd.to_datetime(df['dt_refer'])
    df = df.sort_values('dt_refer').reset_index(drop=True)
    df['periodo'] = df['dt_refer'].dt.strftime('%Y')
    if df['periodo'].duplicated().any():                       # arquivos trimestrais
        df['periodo'] = (df['dt_refer'].dt.year.astype(str) + 'T'
                         + df['dt_refer'].dt.quarter.astype(str))
    return df


def escala(df: pd.DataFrame, coluna: str) -> pd.Series:
    """Série do indicador já convertida para a unidade de exibição."""
    return pd.to_numeric(df[coluna], errors='coerce') / METRICAS[coluna].divisor


def comparativo(catalogo: dict, tipo: str, setores_sel: list, periodo: str | None) -> pd.DataFrame:
    """Uma linha por empresa com os indicadores do ranking, já na unidade de exibição.

    Com `periodo=None` usa o último período de cada empresa; caso contrário usa a
    linha daquele período (empresas que não o têm ficam de fora).
    """
    registros = []
    for setor in setores_sel:
        for ticker_, tipos_ in sorted(catalogo.get(setor, {}).items()):
            caminho = tipos_.get(tipo)
            if not caminho:
                continue
            d = carregar(caminho, Path(caminho).stat().st_mtime)
            if periodo is None:
                linha = d.iloc[-1]
            else:
                recorte = d[d['periodo'] == periodo]
                if recorte.empty:
                    continue
                linha = recorte.iloc[-1]
            registro = {'Empresa': ticker_, 'Setor': nome_setor(setor), 'Período': linha['periodo']}
            for col in RANKING:
                valor = pd.to_numeric(linha[col], errors='coerce') if col in d.columns else None
                registro[col] = (float('nan') if valor is None or pd.isna(valor)
                                 else valor / METRICAS[col].divisor)
            registros.append(registro)
    return pd.DataFrame(registros, columns=['Empresa', 'Setor', 'Período', *RANKING])


def periodos_do_catalogo(catalogo: dict, tipo: str, setores_sel: list) -> list:
    """Períodos presentes em pelo menos uma empresa, do mais recente para o mais antigo."""
    todos = set()
    for setor in setores_sel:
        for tipos_ in catalogo.get(setor, {}).values():
            caminho = tipos_.get(tipo)
            if caminho:
                todos.update(carregar(caminho, Path(caminho).stat().st_mtime)['periodo'])
    return sorted(todos, reverse=True)


def numero(valor: float) -> str:
    """Número no padrão pt-BR (milhar com ponto, decimal com vírgula)."""
    if valor is None or pd.isna(valor):
        return '—'
    return f'{valor:,.2f}'.replace(',', '@').replace('.', ',').replace('@', '.')


def fmt(valor: float, tipo: str) -> str:
    """Formata um número com a unidade do indicador (hover, KPIs)."""
    texto = numero(valor)
    if texto == '—':
        return texto
    return {'brl': f'R$ {texto} bi', 'pct': f'{texto}%', 'x': f'{texto}x', 'rs': f'R$ {texto}'}[tipo]


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------
def _layout(fig: go.Figure, cores: dict, titulo: str, unidade: str, n_series: int) -> go.Figure:
    fig.update_layout(
        title=dict(text=titulo, font=dict(size=15, color=cores['texto']), x=0, xanchor='left'),
        height=360,
        margin=dict(l=8, r=64, t=56, b=8),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=cores['texto_2'], size=12),
        hovermode='x unified',
        showlegend=n_series >= 2,
        legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='left', x=0,
                    bgcolor='rgba(0,0,0,0)', font=dict(color=cores['texto_2'])),
        bargap=0.35,
    )
    fig.update_xaxes(showgrid=False, linecolor=cores['grade'], tickcolor=cores['grade'],
                     ticks='outside', type='category')
    fig.update_yaxes(showgrid=True, gridcolor=cores['grade'], zeroline=True,
                     zerolinecolor=cores['grade'], zerolinewidth=1,
                     title=dict(text=unidade, font=dict(size=11, color=cores['texto_2'])))
    return fig


def linhas(df: pd.DataFrame, colunas: list, titulo: str, cores: dict) -> go.Figure:
    """Séries temporais de indicadores que compartilham a mesma unidade."""
    fig = go.Figure()
    for i, col in enumerate(colunas):
        m, y = METRICAS[col], escala(df, col)
        fig.add_trace(go.Scatter(
            x=df['periodo'], y=y, name=m.label, mode='lines+markers',
            line=dict(color=cores['serie'][i], width=2),
            marker=dict(size=8, color=cores['serie'][i]),
            hovertemplate=f'{m.label}: %{{customdata}}<extra></extra>',
            customdata=[fmt(v, m.tipo) for v in y],
        ))
        validos = y.dropna()
        if len(colunas) == 1 and not validos.empty:            # rótulo direto só no último ponto
            # em eixo 'category' a anotação usa o índice da categoria, não o rótulo
            fig.add_annotation(x=int(df.index.get_loc(validos.index[-1])), y=float(validos.iloc[-1]),
                               text=numero(validos.iloc[-1]), showarrow=False,
                               xanchor='left', xshift=8, font=dict(size=11, color=cores['texto_2']))
    return _layout(fig, cores, titulo, METRICAS[colunas[0]].unidade, len(colunas))


def barras(df: pd.DataFrame, colunas: list, titulo: str, cores: dict) -> go.Figure:
    """Barras por período; com uma única série, a cor codifica o sinal."""
    fig = go.Figure()
    uma_serie = len(colunas) == 1
    rotular = uma_serie and len(df) <= 12
    for i, col in enumerate(colunas):
        m, y = METRICAS[col], escala(df, col)
        cor = ([cores['positivo'] if v >= 0 else cores['negativo'] for v in y.fillna(0)]
               if uma_serie else cores['serie'][i])
        fig.add_trace(go.Bar(
            x=df['periodo'], y=y, name=m.label,
            marker=dict(color=cor, line=dict(width=0)),
            text=[numero(v) for v in y] if rotular else None,
            textposition='outside', textfont=dict(size=11, color=cores['texto_2']),
            cliponaxis=False,
            hovertemplate=f'{m.label}: %{{customdata}}<extra></extra>',
            customdata=[fmt(v, m.tipo) for v in y],
        ))
    fig.update_layout(barmode='group')
    return _layout(fig, cores, titulo, METRICAS[colunas[0]].unidade, len(colunas))


def ranking_barras(dados: pd.DataFrame, col: str, cores: dict, destaque: str) -> go.Figure:
    """Barras horizontais das empresas ordenadas do melhor para o pior no indicador."""
    m = METRICAS[col]
    s = dados.dropna(subset=[col]).copy()
    if col in SO_POSITIVOS:
        s = s[s[col] > 0]
    s = s.sort_values(col, ascending=RANKING[col] == 'menor').reset_index(drop=True)

    sentido = 'menor é melhor' if RANKING[col] == 'menor' else 'maior é melhor'
    fig = go.Figure(go.Bar(
        x=s[col], y=s['Empresa'], orientation='h',
        marker=dict(color=[cores['serie'][0] if e == destaque else cores['neutro']
                           for e in s['Empresa']], line=dict(width=0)),
        text=[numero(v) for v in s[col]], textposition='outside',
        textfont=dict(size=11, color=cores['texto_2']), cliponaxis=False,
        customdata=s[['Setor', 'Período']].values,
        hovertemplate=('<b>%{y}</b><br>' + m.label + ': %{text}'
                       '<br>%{customdata[0]} · %{customdata[1]}<extra></extra>'),
    ))
    fig.update_layout(
        title=dict(text=f'{m.label} · {sentido}', font=dict(size=15, color=cores['texto']),
                   x=0, xanchor='left'),
        height=max(260, 26 * len(s) + 90),
        margin=dict(l=8, r=64, t=52, b=8),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=cores['texto_2'], size=12),
        showlegend=False, bargap=0.3,
    )
    fig.update_xaxes(showgrid=True, gridcolor=cores['grade'], zeroline=True,
                     zerolinecolor=cores['grade'], zerolinewidth=1,
                     title=dict(text=m.unidade, font=dict(size=11, color=cores['texto_2'])))
    fig.update_yaxes(type='category', autorange='reversed', showgrid=False,
                     linecolor=cores['grade'], tickfont=dict(size=11))
    return fig


def bloco(df: pd.DataFrame, cores: dict, graficos: list) -> None:
    """Renderiza em duas colunas os gráficos da aba que têm dados no arquivo."""
    graficos = [g for g in graficos if g[1]]
    if not graficos:
        st.info('Nenhum indicador desta seção está disponível neste arquivo.')
        return
    for i in range(0, len(graficos), 2):
        for coluna, (desenhar, colunas, titulo) in zip(st.columns(2), graficos[i:i + 2]):
            with coluna:
                st.plotly_chart(desenhar(df, colunas, titulo, cores),
                                width='stretch', key=titulo)


def normalizar_ticker_b3(ticker: object) -> str:
    """Converte um ticker digitado pelo usuario para o formato exibido na B3."""
    codigo = str(ticker).strip().upper()
    return codigo.removesuffix('.SA')


def ticker_b3_sugerido(codigo_base: str) -> str:
    """Retorna o ticker completo do codigo-base presente no diretorio de indicadores."""
    codigo = codigo_base.strip().lower()
    return TICKERS_B3.get(codigo, f'{codigo.upper()}.SA')


def ticker_yahoo(ticker: str) -> str:
    """Adiciona o sufixo da B3 exigido pelo Yahoo Finance."""
    return ticker if ticker.startswith('^') or ticker.endswith('.SA') else f'{ticker}.SA'


@st.cache_data(ttl=3600, show_spinner=False)
def carregar_precos_fechamento(tickers: tuple[str, ...], periodo: str) -> pd.DataFrame:
    """Baixa os fechamentos ajustados da B3 em uma unica consulta."""
    import yfinance as yf

    yahoo_tickers = tuple(ticker_yahoo(ticker) for ticker in tickers)
    yf.cache.set_cache_location(str(Path(tempfile.gettempdir()) / 'analise-fundamentalista-yfinance'))
    try:
        dados = yf.download(list(yahoo_tickers), period=periodo, interval='1d', auto_adjust=True,
                            multi_level_index=False, progress=False)
    except (KeyError, RequestsError, ValueError, YFException):
        return pd.DataFrame(columns=list(tickers))

    if dados.empty:
        return pd.DataFrame(columns=list(tickers))
    if isinstance(dados.columns, pd.MultiIndex):
        if 'Close' in dados.columns.get_level_values(0):
            fechamento = dados['Close']
        elif 'Close' in dados.columns.get_level_values(1):
            fechamento = dados.xs('Close', axis=1, level=1)
        else:
            return pd.DataFrame(columns=list(tickers))
    else:
        if 'Close' not in dados.columns:
            return pd.DataFrame(columns=list(tickers))
        fechamento = dados['Close']

    if isinstance(fechamento, pd.Series):
        fechamento = fechamento.to_frame(name=yahoo_tickers[0])
    fechamento = fechamento.reindex(columns=list(yahoo_tickers))
    return fechamento.rename(columns=dict(zip(yahoo_tickers, tickers))).dropna(how='all').dropna(how='all', axis=1)


def linhas_mercado(dados: pd.DataFrame, titulo: str, unidade: str, cores: dict,
                   formato: str = 'brl') -> go.Figure:
    """Desenha linhas de mercado com o mesmo estilo do painel fundamentalista."""
    fig = go.Figure()
    for indice, ticker_ in enumerate(dados.columns):
        valores = dados[ticker_]
        texto_hover = ([f'{valor:.2%}' if pd.notna(valor) else '—' for valor in valores]
                       if formato == 'pct'
                       else [f'R$ {valor:,.2f}' if pd.notna(valor) else '—' for valor in valores])
        fig.add_trace(go.Scatter(
            x=dados.index, y=valores, name=ticker_, mode='lines',
            line=dict(color=cores['serie'][indice], width=2),
            hovertemplate=f'<b>{ticker_}</b>: %{{customdata}}<extra></extra>',
            customdata=texto_hover,
        ))
    fig = _layout(fig, cores, titulo, unidade, len(dados.columns))
    fig.update_xaxes(type='date')
    if formato == 'pct':
        fig.update_yaxes(tickformat='.0%')
    return fig


def grafico_retornos_logaritmicos(precos: pd.DataFrame, cores: dict) -> go.Figure | None:
    """Exibe, por acao, histograma e linha dos retornos logaritmicos diarios."""
    retornos = np.log(precos / precos.shift(1))
    retornos = retornos.dropna(how='all')
    if retornos.empty:
        return None

    titulos = [item for ticker_ in retornos.columns for item in (ticker_, '')]
    fig = make_subplots(rows=len(retornos.columns), cols=2, subplot_titles=titulos, vertical_spacing=0.03)
    for indice, ticker_ in enumerate(retornos.columns, start=1):
        serie = retornos[ticker_].dropna()
        if serie.empty:
            continue
        cor = cores['serie'][(indice - 1) % len(cores['serie'])]
        fig.add_trace(go.Histogram(x=serie, nbinsx=150, histnorm='probability density', marker_color=cor),
                      row=indice, col=1)
        if serie.nunique() > 1:
            media, desvio = norm.fit(serie)
            valores_x = np.linspace(serie.min(), serie.max(), 100)
            fig.add_trace(go.Scatter(x=valores_x, y=norm.pdf(valores_x, media, desvio), mode='lines',
                                     line=dict(color=cores['negativo'], width=1), showlegend=False),
                          row=indice, col=1)
        fig.add_vline(x=0, line=dict(color=cores['negativo'], width=1), row=indice, col=1)
        fig.add_trace(go.Scatter(x=serie.index, y=serie, mode='lines', line=dict(color=cor, width=1.5),
                                 showlegend=False), row=indice, col=2)
        fig.add_hline(y=0, line=dict(color=cores['negativo'], width=1), row=indice, col=2)

    fig.update_layout(
        title=dict(text='Histogramas e grafico de linhas dos retornos logaritmicos',
                   font=dict(size=15, color=cores['texto']), x=0, xanchor='left'),
        height=max(420, 360 * len(retornos.columns)), margin=dict(l=8, r=24, t=56, b=8),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=cores['texto_2'], size=12), showlegend=False,
    )
    fig.update_xaxes(showgrid=False, linecolor=cores['grade'])
    fig.update_yaxes(showgrid=True, gridcolor=cores['grade'], zeroline=False)
    return fig


def coeficiente_hurst(serie: pd.Series) -> float:
    """Calcula o expoente de Hurst por regressao dos desvios das diferencas defasadas."""
    valores = serie.dropna().to_numpy(dtype=float)
    max_lag = min(100, len(valores) // 2)
    if max_lag < 3:
        return float('nan')
    lags = np.arange(2, max_lag)
    desvios = np.array([np.std(valores[lag:] - valores[:-lag]) for lag in lags])
    validos = desvios > 0
    if validos.sum() < 2:
        return float('nan')
    return float(np.polyfit(np.log(lags[validos]), np.log(desvios[validos]), 1)[0])


def tabela_risco_retorno(precos: pd.DataFrame, preco_mercado: pd.Series,
                          alpha: float = 0.05) -> pd.DataFrame:
    """Calcula beta, retorno e medidas de risco diarias de cada acao."""
    retorno_mercado = preco_mercado.pct_change().dropna()
    registros = []
    for ticker_ in precos.columns:
        serie = precos[ticker_].dropna()
        retornos = serie.pct_change().dropna()
        alinhados = pd.concat([retornos.rename('ativo'), retorno_mercado.rename('mercado')], axis=1).dropna()
        variancia_mercado = alinhados['mercado'].var()
        beta = (alinhados['ativo'].cov(alinhados['mercado']) / variancia_mercado
                if len(alinhados) > 1 and variancia_mercado else float('nan'))
        anos = (serie.index[-1] - serie.index[0]).days / 365.25 if len(serie) > 1 else 0
        cagr = (serie.iloc[-1] / serie.iloc[0]) ** (1 / anos) - 1 if anos > 0 and serie.iloc[0] > 0 else float('nan')
        maior_drawdown = (serie / serie.cummax() - 1).min()
        var = retornos.quantile(alpha)
        cvar = retornos[retornos <= var].mean()
        try:
            graus_liberdade, loc, escala_t = student_t.fit(retornos)
            quantil_t = student_t.ppf(alpha, graus_liberdade)
            var_t = loc + escala_t * quantil_t
            cvar_t = (loc - escala_t * (graus_liberdade + quantil_t ** 2)
                      * student_t.pdf(quantil_t, graus_liberdade) / (alpha * (graus_liberdade - 1)))
            if graus_liberdade <= 1:
                cvar_t = float('nan')
        except (ValueError, FloatingPointError):
            var_t, cvar_t = float('nan'), float('nan')
        registros.append({
            'Acao': ticker_, 'Beta': beta, 'Hurst': coeficiente_hurst(serie), 'CAGR': cagr * 100,
            'Maior drawdown': maior_drawdown * 100, 'VaR': var * 100, 'CVaR': cvar * 100,
            'VaR (student-t)': var_t * 100, 'CVaR (student-t)': cvar_t * 100,
        })
    return pd.DataFrame(registros)


def heatmap_retornos_anuais(precos: pd.DataFrame, cores: dict) -> go.Figure | None:
    """Mostra o retorno de cada ano em uma matriz de acoes e anos."""
    fechamentos_anuais = precos.resample('YE').last()
    retornos_anuais = (fechamentos_anuais.pct_change().iloc[1:] * 100).dropna(how='all')
    if retornos_anuais.empty:
        return None
    matriz = retornos_anuais.T
    anos = [str(data.year) for data in matriz.columns]
    texto = [[f'{valor:.1f}%' if pd.notna(valor) else '' for valor in linha] for linha in matriz.to_numpy()]
    fig = go.Figure(go.Heatmap(
        z=matriz.to_numpy(), x=anos, y=matriz.index, text=texto, texttemplate='%{text}',
        colorscale=[[0, cores['negativo']], [0.5, cores['grade']], [1, cores['positivo']]], zmid=0,
        colorbar=dict(title='%'), hovertemplate='<b>%{y}</b><br>%{x}: %{z:.2f}%<extra></extra>',
    ))
    fig.update_layout(title=dict(text='Heatmap dos retornos anuais', font=dict(size=15, color=cores['texto']),
                                 x=0, xanchor='left'), height=max(260, 44 * len(matriz) + 145),
                      margin=dict(l=8, r=64, t=56, b=8), paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)', font=dict(color=cores['texto_2'], size=12))
    fig.update_xaxes(side='bottom', type='category')
    fig.update_yaxes(type='category', autorange='reversed')
    return fig


def heatmap_correlacao(precos: pd.DataFrame, cores: dict) -> go.Figure | None:
    """Mostra a correlacao dos retornos diarios entre as acoes selecionadas."""
    correlacao = precos.pct_change().corr().dropna(how='all').dropna(how='all', axis=1)
    if correlacao.empty:
        return None
    texto = [[f'{valor:.2f}' if pd.notna(valor) else '' for valor in linha] for linha in correlacao.to_numpy()]
    fig = go.Figure(go.Heatmap(
        z=correlacao.to_numpy(), x=correlacao.columns, y=correlacao.index, text=texto, texttemplate='%{text}',
        colorscale=[[0, cores['negativo']], [0.5, cores['grade']], [1, cores['positivo']]],
        zmin=-1, zmax=1, zmid=0, colorbar=dict(title='Correlacao'),
        hovertemplate='<b>%{y} x %{x}</b><br>Correlacao: %{z:.2f}<extra></extra>',
    ))
    fig.update_layout(title=dict(text='Heatmap de correlacao', font=dict(size=15, color=cores['texto']),
                                 x=0, xanchor='left'), height=max(300, 48 * len(correlacao) + 145),
                      margin=dict(l=8, r=86, t=56, b=8), paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)', font=dict(color=cores['texto_2'], size=12))
    fig.update_xaxes(side='bottom', type='category')
    fig.update_yaxes(type='category', autorange='reversed')
    return fig


def grafico_risco_retorno(precos: pd.DataFrame, tabela: pd.DataFrame, cores: dict) -> go.Figure:
    """Relaciona volatilidade anualizada e CAGR das acoes selecionadas."""
    volatilidade = precos.pct_change().std() * np.sqrt(252) * 100
    dados = pd.DataFrame({
        'Acao': tabela['Acao'], 'Volatilidade anualizada': tabela['Acao'].map(volatilidade), 'CAGR': tabela['CAGR'],
    }).dropna()
    fig = go.Figure()
    for indice, linha in dados.reset_index(drop=True).iterrows():
        fig.add_trace(go.Scatter(
            x=[linha['Volatilidade anualizada']], y=[linha['CAGR']], mode='markers+text', name=linha['Acao'],
            text=[linha['Acao']], textposition='top center',
            marker=dict(size=13, color=cores['serie'][indice], line=dict(color=cores['texto'], width=1)),
            hovertemplate=(f"<b>{linha['Acao']}</b><br>Volatilidade anualizada: "
                           '%{x:.2f}%<br>CAGR: %{y:.2f}%<extra></extra>'),
        ))
    fig.update_layout(title=dict(text='Risco e retorno', font=dict(size=15, color=cores['texto']), x=0,
                                 xanchor='left'), height=430, margin=dict(l=8, r=24, t=56, b=8),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(color=cores['texto_2'], size=12), showlegend=False)
    fig.update_xaxes(title='Volatilidade anualizada (%)', showgrid=True, gridcolor=cores['grade'], zeroline=False)
    fig.update_yaxes(title='CAGR (%)', showgrid=True, gridcolor=cores['grade'], zeroline=True,
                     zerolinecolor=cores['negativo'], zerolinewidth=1)
    return fig


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
st.set_page_config(page_title='Indicadores fundamentalistas', page_icon='📊', layout='wide')

try:
    tema = st.context.theme.type
except Exception:
    tema = 'light'
cores = PALETA.get(tema, PALETA['light'])

# --- Seleção do arquivo: setor -> empresa -> periodicidade ------------------
st.sidebar.header('Empresa')
raiz = st.sidebar.text_input('Diretório dos indicadores', value=RAIZ_PADRAO)

if not Path(raiz).is_dir():
    st.error(f'Diretório não encontrado: `{raiz}`')
    st.stop()

catalogo = mapear(raiz)
if not catalogo:
    st.error(f'Nenhum arquivo no padrão `<setor>/<ticker>/<ticker>_indicadores_<tipo>.csv` '
             f'em `{raiz}`.')
    st.stop()

setor = st.sidebar.selectbox('Setor', sorted(catalogo), format_func=nome_setor)
ticker = st.sidebar.selectbox('Empresa', sorted(catalogo[setor]))
tipos = catalogo[setor][ticker]
tipo = st.sidebar.radio('Periodicidade', sorted(tipos), horizontal=True,
                        format_func=str.capitalize)
if st.sidebar.button('Atualizar lista de empresas', width='stretch'):
    mapear.clear()
    st.rerun()

arquivo = Path(tipos[tipo])
df = carregar(str(arquivo), arquivo.stat().st_mtime)
disponiveis = [c for c in METRICAS if c in df.columns]

st.sidebar.header('Período')
periodos = df['periodo'].tolist()
inicio, fim = st.sidebar.select_slider('Intervalo', options=periodos,
                                       value=(periodos[0], periodos[-1]),
                                       label_visibility='collapsed')
df = df[(df['periodo'] >= inicio) & (df['periodo'] <= fim)].reset_index(drop=True)

st.title(f'{ticker} — indicadores fundamentalistas')
st.caption(f'{nome_setor(setor)} · {tipo} · {len(df)} períodos '
           f'({df["periodo"].iloc[0]} a {df["periodo"].iloc[-1]}) · '
           f'valores monetários em R$ bilhões · {arquivo.name}')

# Resumo do último período, com variação contra o período anterior
st.subheader(f'Último período · {df["periodo"].iloc[-1]}')
destaques = [c for c in ('p/l', 'p/vp', 'ev_ebitda', 'roe', 'roic', 'margem_liquida',
                         'dl_ebitda', 'payout') if c in disponiveis]
for faixa in (destaques[:4], destaques[4:]):
    for coluna, nome in zip(st.columns(len(faixa)), faixa):
        m, serie = METRICAS[nome], escala(df, nome)
        atual = serie.iloc[-1]
        anterior = serie.iloc[-2] if len(serie) > 1 else None
        delta = None if anterior is None or pd.isna(anterior) or pd.isna(atual) else atual - anterior
        coluna.metric(m.label, fmt(atual, m.tipo),
                      None if delta is None else fmt(delta, m.tipo), help=m.ajuda or None)

abas = st.tabs(['Valuation', 'Rentabilidade', 'Endividamento', 'Fluxo de caixa',
                'Proventos', 'Explorar', 'Dados', 'Ranking', 'Indicadores gerais'])


def existentes(*colunas: str) -> list:
    """Mantém só as colunas presentes no arquivo aberto, na ordem pedida."""
    return [c for c in colunas if c in disponiveis]


with abas[0]:
    bloco(df, cores, [
        (linhas, existentes('p/l', 'p/vp', 'ev_ebitda'), 'Múltiplos de mercado'),
        (linhas, existentes('l/p'), 'Earnings yield (L/P)'),
        (barras, existentes('valor_mercado'), 'Valor de mercado'),
        (linhas, existentes('lpa', 'lpa_acum', 'vpa'), 'LPA e VPA'),
    ])

with abas[1]:
    bloco(df, cores, [
        (linhas, existentes('roe', 'roic'), 'ROE e ROIC'),
        (linhas, existentes('margem_liquida'), 'Margem líquida'),
        (barras, existentes('ebitda', 'ebitda_acum'), 'EBITDA'),
        (linhas, existentes('effective_tax_rate', 'reinvestment_rate'),
         'Alíquota efetiva de IR e taxa de reinvestimento'),
    ])

with abas[2]:
    bloco(df, cores, [
        (barras, existentes('divida_bruta', 'caixa_total', 'divida_liquida'),
         'Dívida bruta, caixa e dívida líquida'),
        (linhas, existentes('dl_ebitda', 'dl_pl'), 'Alavancagem'),
    ])

with abas[3]:
    bloco(df, cores, [
        (barras, existentes('fco', 'fci', 'fcf'), 'Fluxos de caixa (FCO, FCI, FCF)'),
        (barras, existentes('fco_nao_acum', 'fci_nao_acum', 'fcf_nao_acum'),
         'Fluxos de caixa no trimestre'),
        (barras, existentes('fco_acum', 'fci_acum', 'fcf_acum'),
         'Fluxos de caixa acumulados no ano'),
        (barras, existentes('free_cash_flow'), 'Free cash flow'),
        (barras, existentes('capex', 'capex_acum', 'net_capex', 'adjusted_net_capex'), 'Capex'),
        (barras, existentes('fcfe', 'fcff'), 'FCFE e FCFF'),
        (barras, existentes('working_capital', 'change_in_non_cash_wc'), 'Capital de giro'),
    ])

with abas[4]:
    bloco(df, cores, [
        (barras, existentes('proventos', 'buyback', 'buyback_nao_acum'), 'Proventos e recompras'),
        (linhas, existentes('payout'), 'Payout'),
    ])

with abas[5]:
    escolhidos = st.multiselect(
        'Indicadores (combine apenas indicadores de mesma unidade — o gráfico usa um único eixo)',
        options=disponiveis, default=disponiveis[:1],
        format_func=lambda c: f'{METRICAS[c].label} ({METRICAS[c].unidade})')
    unidades = {METRICAS[c].unidade for c in escolhidos}
    if not escolhidos:
        st.info('Escolha ao menos um indicador.')
    elif len(unidades) > 1:
        st.warning(f'Unidades diferentes selecionadas ({", ".join(sorted(unidades))}). '
                   'Escolha indicadores de mesma unidade.')
    else:
        forma = st.radio('Forma', ['Linha', 'Barra'], horizontal=True, label_visibility='collapsed')
        desenhar = linhas if forma == 'Linha' else barras
        titulo = ' · '.join(METRICAS[c].label for c in escolhidos[:8])
        st.plotly_chart(desenhar(df, escolhidos[:8], titulo, cores), width='stretch')

with abas[6]:
    tabela = pd.DataFrame({'Período': df['periodo']})
    for col in disponiveis:
        tabela[f'{METRICAS[col].label} ({METRICAS[col].unidade})'] = escala(df, col).round(2)
    st.dataframe(tabela.set_index('Período').T, width='stretch')
    st.download_button('Baixar CSV filtrado', df.to_csv(sep=';', index=False).encode('utf-8'),
                       file_name=arquivo.name, mime='text/csv')

with abas[7]:
    st.caption(f'Compara as empresas do catálogo pelos arquivos **{tipo}**. '
               f'{ticker} aparece destacado em azul nos gráficos.')
    col_setores, col_periodo = st.columns([3, 2])
    setores_sel = col_setores.multiselect('Setores', sorted(catalogo), default=sorted(catalogo),
                                          format_func=nome_setor)
    modo = col_periodo.radio('Período de comparação',
                             ['Último de cada empresa', 'Um período específico'])
    periodo_ranking = None
    if modo == 'Um período específico':
        opcoes = periodos_do_catalogo(catalogo, tipo, setores_sel)
        if opcoes:
            periodo_ranking = col_periodo.selectbox('Período', opcoes)

    dados = comparativo(catalogo, tipo, setores_sel, periodo_ranking)
    if dados.empty:
        st.info('Selecione ao menos um setor com empresas no período escolhido.')
    else:
        if periodo_ranking is None and dados['Período'].nunique() > 1:
            st.warning('As empresas têm últimos períodos diferentes '
                       f'({", ".join(sorted(dados["Período"].unique()))}). '
                       'Para comparar todas na mesma data, escolha um período específico.')
        fora = {c: sorted(dados.loc[dados[c] <= 0, 'Empresa']) for c in SO_POSITIVOS}
        fora = {METRICAS[c].label: e for c, e in fora.items() if e}
        if fora:
            st.caption('Fora do ranking por múltiplo negativo (prejuízo ou EBITDA negativo): '
                       + ' · '.join(f'{k}: {", ".join(v)}' for k, v in fora.items()))

        indicadores = list(RANKING)
        for i in range(0, len(indicadores), 2):
            for coluna, col_ind in zip(st.columns(2), indicadores[i:i + 2]):
                with coluna:
                    st.plotly_chart(ranking_barras(dados, col_ind, cores, ticker),
                                    width='stretch', key=f'rank_{col_ind}')

        st.subheader('Tabela comparativa')
        st.caption('Dívida líquida negativa (DL/EBITDA e DL/PL) significa caixa líquido. '
                   'Clique no cabeçalho para ordenar.')
        st.dataframe(
            dados.round(2),
            width='stretch', hide_index=True,
            column_config={c: st.column_config.NumberColumn(
                f'{METRICAS[c].label} ({METRICAS[c].unidade})', format='%.2f') for c in RANKING})

with abas[8]:
    acoes = {
        ticker_b3_sugerido(ticker_): nome_setor(setor_)
        for setor_, empresas in catalogo.items()
        for ticker_ in empresas
    }
    tickers = st.multiselect(
        'Ações', options=sorted(acoes), default=[ticker_b3_sugerido(ticker)], max_selections=len(cores['serie']),
        format_func=lambda ticker_: (f'{ticker_} - {acoes[ticker_]}' if ticker_ in acoes
                                     else f'{str(ticker_).upper()} (ticker manual)'),
        accept_new_options=True, placeholder='Selecione ou digite um ticker (ex.: PETR4)',
        key='indicadores_gerais_acoes_b3',
    )
    tickers = list({normalizar_ticker_b3(ticker_): None for ticker_ in tickers if str(ticker_).strip()})
    periodos_mercado = {'2 anos': '2y', '5 anos': '5y', '10 anos': '10y'}
    periodo_mercado = st.selectbox('Histórico de preços', list(periodos_mercado), index=0,
                                  key='indicadores_gerais_periodo')
    if not tickers:
        st.info('Escolha ao menos uma acao.')
    else:
        with st.spinner('Baixando precos das acoes selecionadas...'):
            precos_acoes = carregar_precos_fechamento(tuple(tickers), periodos_mercado[periodo_mercado])
        if precos_acoes.empty:
            st.warning('Nao foi possivel obter os precos das acoes selecionadas.')
        else:
            st.plotly_chart(linhas_mercado(precos_acoes, 'Precos das Acoes', 'R$', cores), width='stretch')
            retorno_acumulado = precos_acoes.apply(
                lambda serie: serie / serie.dropna().iloc[0] - 1 if serie.notna().any() else serie
            ).dropna(how='all')
            grafico_retorno = linhas_mercado(retorno_acumulado, 'Retorno acumulado', '%', cores, formato='pct')
            grafico_retorno.add_hline(y=0, line=dict(color=cores['negativo'], width=1.5))
            st.plotly_chart(grafico_retorno, width='stretch')
            heatmap = heatmap_retornos_anuais(precos_acoes, cores)
            if heatmap is None:
                st.info('O historico selecionado nao contem anos suficientes para calcular os retornos anuais.')
            else:
                st.plotly_chart(heatmap, width='stretch')
            momentum = (precos_acoes - precos_acoes.shift(252)).dropna(how='all')
            if momentum.empty:
                st.info('O historico selecionado nao contem 252 pregoes para calcular o momentum de um ano.')
            else:
                grafico_momentum = linhas_mercado(momentum, 'Momentum - 1 ano (252 pregoes)', 'R$', cores)
                grafico_momentum.add_hline(y=0, line=dict(color=cores['negativo'], width=1.5))
                st.plotly_chart(grafico_momentum, width='stretch')
            retornos_log = np.log(precos_acoes / precos_acoes.shift(1))
            volatilidade = (retornos_log.rolling(252, min_periods=252).std() * np.sqrt(252)).dropna(how='all')
            if volatilidade.empty:
                st.info('O historico selecionado nao contem 252 pregoes para calcular a volatilidade anualizada.')
            else:
                st.plotly_chart(linhas_mercado(volatilidade, 'Volatilidade anualizada (252 pregoes)', '%', cores,
                                                formato='pct'), width='stretch')
            volatilidade_movel = (retornos_log.rolling(60, min_periods=60).std() * np.sqrt(252)).dropna(how='all')
            if volatilidade_movel.empty:
                st.info('O historico selecionado nao contem 60 pregoes para calcular a volatilidade movel.')
            else:
                st.plotly_chart(linhas_mercado(volatilidade_movel, 'Volatilidade movel (60 pregoes)', '%', cores,
                                                formato='pct'), width='stretch')
            grafico_retornos = grafico_retornos_logaritmicos(precos_acoes, cores)
            if grafico_retornos is not None:
                st.plotly_chart(grafico_retornos, width='stretch')
            st.subheader('Indicadores de risco e retorno')
            st.caption('Beta contra o Ibovespa; CAGR do periodo selecionado; VaR e CVaR diarios a 95% de confianca.')
            with st.spinner('Calculando os indicadores de risco e retorno...'):
                precos_mercado = carregar_precos_fechamento(('^BVSP',), periodos_mercado[periodo_mercado])
            if precos_mercado.empty or precos_mercado['^BVSP'].dropna().empty:
                st.warning('Nao foi possivel obter os precos do Ibovespa para calcular o beta.')
            else:
                tabela_risco = tabela_risco_retorno(precos_acoes, precos_mercado['^BVSP'])
                st.dataframe(
                    tabela_risco, width='stretch', hide_index=True,
                    column_config={
                        'Beta': st.column_config.NumberColumn('Beta', format='%.2f'),
                        'Hurst': st.column_config.NumberColumn('Hurst', format='%.3f'),
                        'CAGR': st.column_config.NumberColumn('CAGR', format='%.2f%%'),
                        'Maior drawdown': st.column_config.NumberColumn('Maior drawdown', format='%.2f%%'),
                        'VaR': st.column_config.NumberColumn('VaR', format='%.2f%%'),
                        'CVaR': st.column_config.NumberColumn('CVaR', format='%.2f%%'),
                        'VaR (student-t)': st.column_config.NumberColumn('VaR (student-t)', format='%.2f%%'),
                        'CVaR (student-t)': st.column_config.NumberColumn('CVaR (student-t)', format='%.2f%%'),
                    },
                )
                st.caption('Hurst: abaixo de 0,5 indica reversao a media; perto de 0,5, comportamento aleatorio; acima de 0,5, persistencia de tendencia.')
                st.plotly_chart(grafico_risco_retorno(precos_acoes, tabela_risco, cores), width='stretch')
                heatmap_correlacoes = heatmap_correlacao(precos_acoes, cores)
                if heatmap_correlacoes is not None:
                    st.plotly_chart(heatmap_correlacoes, width='stretch')
