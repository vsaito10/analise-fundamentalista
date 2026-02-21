from datetime import datetime # noqa: F401
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.cluster.hierarchy as sch
import scipy.stats as sps
import statsmodels.api as sm
import yfinance as yf


# Funções de indicadores gerais de mercado
def volatilidade(df: pd.DataFrame, ticker: str, ano: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calcula a volatilidade anualizada, mensal e semanal.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame das ações.
    ticker: str
        Ticker da empresa.
    ano: str
        Período escolhido.

    Returns
    -------
    volatilidade_anual: pd.DataFrame
        DataFrame da volatilidade anualiazada.
    volatilidade_mensal: pd.DataFrame
        DataFrame da volatilidade mensal.
    volatilidade_semanal: pd.DataFrame
        DataFrame da volatilidade semanal.
    """
    # Calculando o retorno logarítmico
    log_return = np.log(df.loc[ano, ticker] / df.loc[ano, ticker].shift(1))

    # Calculando a volatilidade anualizada
    volatilidade_anual = np.std(log_return) * np.sqrt(252)

    # Calculando a volatilidade mensal
    volatilidade_mensal = np.std(log_return) * np.sqrt(12)

    # Calculando a volatilidade semanal
    volatilidade_semanal = np.std(log_return) * np.sqrt(52)

    return volatilidade_anual, volatilidade_mensal, volatilidade_semanal


def drawdown(returns: pd.DataFrame) -> float:
    """
    Calcula o drawdown.

    Parameters
    ----------
    returns: pd.DataFrame
        DataFrame dos retornos diário do ativo.

    Returns
    ----------
    float
        Ponto mínimo do drawdown.
    """
    # Calculando o retorno acumulado
    cumulative_returns = (1+returns).cumprod()

    # Calculando o pico
    peak = cumulative_returns.expanding(min_periods=1).max()

    # Calculando o drawdown
    drawdown = (cumulative_returns / peak) - 1

    return drawdown.min()


def cluster_corr(corr_array: pd.DataFrame|np.ndarray, inplace=False):
    """
    Reorganiza a matriz de correlação ('corr_array') para que grupos de variáveis altamente correlacionadas fiquem próximos uns dos outros.
    
    Parameters
    ----------
    corr_array: pd.DataFrame | np.ndarray
        Matriz NxN de correlação.
        
    Returns
    -------
    Matriz NxN de correlação com as colunas reorganizadas.

    NOTE: https://wil.yegelwel.com/cluster-correlation-matrix/
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    
    if not inplace:
        corr_array = corr_array.copy()
    
    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
        
    return corr_array[idx, :][:, idx]


def cagr(start_value: float, end_value: float, num_periods: int) -> float:
    """
    Calcula o CAGR.

    Parameters
    ----------
    start_value: float
        Preço inicial da ação.
    end_value: float
        Preço final da ação.
    num_periods: int
        Número de períodos.

    Returns
    -------
    float
        Número do CAGR.
    """
    return (end_value / start_value) ** (1 / (num_periods - 1)) - 1


def log_returns(df: pd.DataFrame):
    """
    Calcula os retornos logarítmicos para cada empresa em um DataFrame de preços de fechamento.
    
    Parameters
    ----------
    df: pd.DataFrame
        DataFrame contendo os preços de fechamento das empresas.
        
    Returns
    -------
    pd.DataFrame
        DataFrame contendo os retornos logarítmicos de cada empresa.
    """
    log_returns = np.log(df / df.shift(1))
    return log_returns


def calculate_beta(index: str, stock: str, period: str, interval: str, just_beta=False) -> float:
    """
    Função que calcula o Beta da ação.

    Parameters
    ----------
    index: str
        Índice de ações (S&P500, Nasdaq, Russel, IBOV e etc).
    stock: str
        Ticker da ação.
    period: str
        Período dos preços da ação.
    interval: str
        Intervalo dos preços da ação.
    just_beta: bool
        Retorna apenas o beta ou beta e ols.

    Returns
    -------
    beta: float
        Beta da ação.
    """
    index_data = yf.download(tickers=index, period=period, interval=interval, auto_adjust=True, multi_level_index=False)['Close']
    stock_data = yf.download(tickers=stock, period=period, interval=interval, auto_adjust=True, multi_level_index=False)['Close']

    index_data =  index_data.rename(index)
    stock_data =  stock_data.rename(stock)

    data = pd.merge(index_data, stock_data, left_index=True, right_index=True)
    data[f"{index}_return"] = data[index].pct_change()
    data[f"{stock}_return"] = data[stock].pct_change()
    data.dropna(inplace=True)
    X = data[f"{index}_return"].values
    y = data[f"{stock}_return"].values
    X = sm.add_constant(X)

    ols = sm.OLS(y, X).fit()

    beta = ols.params[1]

    if just_beta:
        return beta
    else:
        return beta, ols


def plot_indicator_scatter(ten_k: bool, lst_dfs: list, indicador: str, setor: str):
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    lst_dfs: list
        Lista que contém tuplas com o nome da empresa e o seu df dos indicadores fundamentalistas.
    indicador: str
        Nome do indicador fundamentalista.
    setor: str
        Nome do setor que as empresas atuam.

    Returns
    ----------
    Plot do gráfico de linha do indicador fundamentalista específico.
    """
    if ten_k:
        # Plotando o indicador específico
        fig = go.Figure()

        for empresa, indicadores in lst_dfs:
            fig.add_trace(go.Scatter(
                x=indicadores[indicador].index,
                y=indicadores[indicador].values,
                name=empresa
            ))

        # Adicionando uma linha horizontal
        fig.add_hline(y=0, line=dict(color='red', width=1), line_dash='dash')

        # Atualizando o layout
        fig.update_layout(
            height=700,
            width=1000,
            title_text=f'Empresas do setor de {setor} - {indicador}',
            template='seaborn'
        )
    
    else:
        # Plotando o indicador específico
        fig = go.Figure()

        for empresa, indicadores in lst_dfs:
            fig.add_trace(go.Scatter(
                x=indicadores[indicador].index,
                y=indicadores[indicador].values,
                name=empresa
            ))

        # Adicionando uma linha horizontal
        fig.add_hline(y=0, line=dict(color='red', width=1))

        # Atualizando o layout
        fig.update_layout(
            height=700,
            width=1000,
            title_text=f'Empresas do setor de {setor} - {indicador}',
            template='seaborn'
        )

    return fig.show()


def plot_indicator_bar(lst_dfs: list, indicador: str, setor: str):
    """
    Parameters
    ----------
    lst_dfs: list
        Lista que contém tuplas com o nome da empresa e o seu df dos indicadores fundamentalistas.
    indicador: str
        Nome do indicador fundamentalista.
    setor: str
        Nome do setor que as empresas atuam.

    Returns
    -------
    Plot do gráfico de barras do indicador fundamentalista específico.
    """
    # Plotando o indicador específico
    fig = go.Figure()

    for empresa, indicadores in lst_dfs:
        fig.add_trace(go.Bar(
            x=indicadores[indicador].index,
            y=indicadores[indicador].values,
            name=empresa
        ))

    # Adicionando uma linha horizontal
    fig.add_hline(y=0, line=dict(color='red', width=1), line_dash='dash')

    # Atualizando o layout
    fig.update_layout(
        height=700,
        width=1000,
        title_text=f'Empresas do setor de {setor} - {indicador}',
        template='seaborn'
    )
    
    return fig.show()


def plot_indicators_subplot_scatter(ten_k: bool, lst_dfs: list, indicadores: list, setor: str):
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    lst_dfs: list
        Lista que contém tuplas com o nome da empresa e o seu df dos indicadores fundamentalistas.
    indicadores: list
        Lista de nomes dos indicadores fundamentalistas.
    setor: str
        Nome do setor que as empresas atuam.

    Returns
    -------
    Subplot dos indicadores fundamentalistas.

    NOTE: caso eu adicione mais empresas do setor, eu tenho que atualizar a lista 'lst_empresas'.
    """
    if ten_k:
        # Criando subplots
        fig = make_subplots(
            rows=len(indicadores), 
            cols=1, 
            subplot_titles=indicadores, 
            vertical_spacing=0.02
        )

        for idx, indicador in enumerate(indicadores):
            row = idx + 1
            for empresa, indicadores_df in lst_dfs:
                fig.add_trace(go.Scatter(
                    x=indicadores_df[indicador].index,
                    y=indicadores_df[indicador].values,
                    name=empresa
                ), row=row, col=1)

        # Adicionando uma linha horizontal
        fig.update_yaxes(fixedrange=True)
        fig.add_hline(y=0, line=dict(color='red', width=1), line_dash='dash')

        # Atualizando o layout
        fig.update_layout(
            height=4000,
            width=1000,
            title_text=f'Empresas do setor de {setor}',
            template='seaborn',
            showlegend=False
        )
    
    else:
        # Criando subplots
        fig = make_subplots(
            rows=len(indicadores), 
            cols=1, 
            subplot_titles=indicadores, 
            vertical_spacing=0.02
        )

        for idx, indicador in enumerate(indicadores):
            row = idx + 1
            for empresa, indicadores_df in lst_dfs:
                fig.add_trace(go.Scatter(
                    x=indicadores_df[indicador].index,
                    y=indicadores_df[indicador].values,
                    name=empresa
                ), row=row, col=1)

        # Adicionando uma linha horizontal
        fig.update_yaxes(fixedrange=True)
        fig.add_hline(y=0, line=dict(color='red', width=1), line_dash='dash')

        # Atualizando o layout
        fig.update_layout(
            height=4000,
            width=1000,
            title_text=f'Empresas do setor de {setor}',
            template='seaborn',
            showlegend=False
        )

    return fig.show()


def plot_indicators_subplot_bar(lst_dfs: list, indicadores: list, setor: str):
    """
    Parameters
    ----------
    lst_dfs: list
        Lista que contém tuplas com o nome da empresa e o seu df dos indicadores fundamentalistas.
    indicadores: list
        Lista de nomes dos indicadores fundamentalistas.
    setor: str
        Nome do setor que as empresas atuam.

    Returns
    -------
    Subplot dos indicadores fundamentalistas.

    NOTE: caso eu adicione mais empresas do setor, eu tenho que atualizar a lista 'lst_empresas'.
    """
    # Criando subplots
    fig = make_subplots(
        rows=len(indicadores), 
        cols=1, 
        subplot_titles=indicadores, 
        vertical_spacing=0.05
    )

    for idx, indicador in enumerate(indicadores):
        row = idx + 1
        for empresa, indicadores_df in lst_dfs:
            fig.add_trace(go.Bar(
                x=indicadores_df[indicador].index,
                y=indicadores_df[indicador].values,
                name=empresa
            ), row=row, col=1)

    # Adicionando uma linha horizontal
    fig.update_yaxes(fixedrange=True)
    fig.add_hline(y=0, line=dict(color='red', width=1), line_dash='dash')

    # Atualizando o layout
    fig.update_layout(
        height=1500,
        width=1000,
        title_text=f'Empresas do setor de {setor}',
        template='seaborn',
        showlegend=False
    )

    return fig.show()


def plot_only_indicators_subplot_bar(lst_dfs: list, indicador: str, setor:str):
    """
    Parameters
    ----------
    lst_dfs: list
        Lista que contém tuplas com o nome da empresa e o seu df dos indicadores fundamentalistas.
    indicador: str
        Escolher um dos indicadores ('Valor de Mercado', 'Dívida Bruta', 'Caixa e Equivalentes' e 'Dívida Líquida').
    setor: str
        Nome do setor que as empresas atuam.
    
    Returns
    -------
    Subplot do gráfico de barras do indicador selecionado.

    NOTE: eu criei essa função, porque eu não consigo fazer um subplot de barras com os dados 10-q,
    porque eles possuem índices totalmente diferentes.
    """
    # Criando lista de títulos com o nome de cada empresa
    subplot_titles = [f'{empresa}' for empresa, _ in lst_dfs]

    # Criando subplots
    fig = make_subplots(
        rows=len(lst_dfs), 
        cols=1,
        subplot_titles=subplot_titles
        )

    # Iterando sobre os dfs p/ fazer o plot do indicador fundamentalista específico
    for i, (empresa, df) in enumerate(lst_dfs, start=1):
        fig.add_trace(go.Bar(
            x=df.index,
            y=df[indicador].values
        ), row=i, col=1)

    # Atualizando o layout
    fig.update_layout(
        height=1000,
        width=1000,
        title_text=f'Empresa do setor {setor} - {indicador}',
        template='seaborn',
        showlegend=False
    )

    return fig.show()


def plot_indicators_subplot_histogram(df_setor: pd.DataFrame, setor: str):
    """
    Parameters
    ----------
    df_setor: pd.DataFrame
        DataFrame que contém os preços de fechamento das empresas do setor selecionado.
    setor: str
        Nome do setor.
    
    Returns
    -------
    Subplot do histogramas dos retornos logarítmicos das empresas.
    """
    # Calculando o retorno logarítmico
    df_log_returns = log_returns(df_setor)
    df_log_returns = df_log_returns.dropna()

    # Lista com os nomes das empresas do setor p/ serem plotadas apenas nos subplots da 'col=1'
    # ['AMD', '', 'AVGO', '', 'INTC', '', 'MU', '', 'NVDA']
    lst_subtitles_col_1 = [item for empresa in df_log_returns.keys() for item in (empresa, '')]

    # Plotando o histograma dos retornos logarítmicos
    fig = make_subplots(
        rows=len(df_log_returns.columns), 
        cols=2, 
        subplot_titles=lst_subtitles_col_1,
        vertical_spacing=0.03
    )

    for idx, empresa in enumerate(df_log_returns):
        row = idx + 1
        # Plot histograma
        fig.add_trace(go.Histogram(
            x=df_log_returns[empresa],
            nbinsx=150,
            histnorm='probability density'
        ), row=row, col=1)

        # Calculando a média e o desvio-padrão da empresa para plotar a curva normal
        mu_norm, sig_norm = sps.norm.fit(df_log_returns[empresa])
        x_values = np.linspace(min(df_log_returns[empresa]), max(df_log_returns[empresa]), 100)
        y_values = sps.norm.pdf(x_values, mu_norm, sig_norm)

        # Adicionando a curva de distribuição normal como um traço
        fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', line_width=0.5, line=dict(color='red')), row=row, col=1)

        # Adicionando uma linha vertical no ponto 0
        fig.add_vline(x=0, line=dict(color='red', width=1), row=row, col=1)

        # Plot Scatter
        fig.add_trace(go.Scatter(
            x=df_log_returns.index,
            y=df_log_returns[empresa]
        ), row=row, col=2)

        # Adicionando uma linha horizontal
        fig.add_hline(y=0, line=dict(color='red', width=1), row=row, col=2)
    
    # Atualizando o layout
    fig.update_layout(
        height=3000,
        width=1200,
        title_text=f'Setor de {setor} - Histogramas e gráfico de linhas dos retornos logarítmicos',
        template='seaborn',
        showlegend=False
    )

    return fig.show()


def subplot_qqplot(df_setor: pd.DataFrame):
    """
    Parameters
    ----------
    df_setor: pd.DataFrame
        DataFrame que contém os preços de fechamento das empresas do setor selecionado.
    
    Returns
    -------
    Subplot do gráfico de qqplot das empresas.
    """
    # Calculando o retorno logarítmico
    df_log_returns = log_returns(df_setor)
    df_log_returns2 = df_log_returns.dropna()

    # Subplot do qqplot das empresas
    with plt.style.context('ggplot'):
        fig, axs = plt.subplots(len(df_log_returns2.columns), 1, figsize=(8, 25))

        for idx, empresa in enumerate(df_log_returns2.columns):
            axs[idx].set_title(f'QQ-plot {empresa}', fontsize=16)
            sm.qqplot(df_log_returns2[empresa], fit=True, line='45', ax=axs[idx])  

        plt.tight_layout() 
            
    return plt.show()


def plot_risk_return(df_setor: pd.DataFrame, setor: str):
    """
    Parameters
    ----------
    df_setor: pd.DataFrame
        DataFrame que contém os preços de fechamento das empresas do setor selecionado.
    setor: str
        Nome do setor.
    
    Returns
    -------
    Plot do gráfico da relação risco x retorno do setor selecionado.
    """
    # Calculando o retorno logarítmico
    df_log_returns = log_returns(df_setor)

    # Listas da média do retorno logarítmico e do desvio-padrão do retorno logarítmico
    lst_ret_mean = []
    lst_ret_std = []

    for empresa in df_log_returns:
        # Média do retorno logarítmico
        ret_mean = df_log_returns[empresa].dropna().mean() * 100
        # Desvio-padrão do retorno logarítmico médio
        ret_std = df_log_returns[empresa].dropna().std() * 100
        lst_ret_mean.append(ret_mean)
        lst_ret_std.append(ret_std)

    # Dataframe da relação risco x retorno
    df_risk_return = pd.DataFrame([lst_ret_mean, lst_ret_std], columns=df_log_returns.columns, index=['mean', 'std'])

    # Plotando o grafico da relação risco x retorno
    fig = go.Figure()

    for empresa in df_risk_return: 
        fig.add_trace(go.Scatter(
            x=[df_risk_return.loc['mean', empresa]],
            y=[df_risk_return.loc['std', empresa]],
            mode='markers',
            marker=dict(symbol='star', size=10),
            name=empresa
        ))

    # Atualizando o layout
    fig.update_layout(
        title=f'Setor de {setor} - Gráfico risco x retorno',
        xaxis=dict(title='Média Esperada Retorno Diário'),
        yaxis=dict(title='Risco Diário'),
        showlegend=True
    )

    return fig.show()


def VaR(df_setor: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parameters
    ----------
    df_setor: pd.DataFrame
        DataFrame que contém os preços de fechamento das empresas do setor selecionado.

    Returns
    -------
    df_var_hist: pd.DataFrame
        DataFrame do VaR histórico.
    df_var_param: pd.DataFrame
        DataFrame do VaR paramétrico.

    NOTE: é o quantil da distribuição de perda. Lembrando que aqui as perdas são positivas.
    Se um ativo (ou portfólio) possui um VaR(95)=2%, significa que o investidor tem 95% de certeza que as perdas não serão maiores do que 2%, 
    ou que há uma probabilidade de 5% das perdas excederem 2%.
    """
    # Calculando o retorno logarítmico
    df_log_returns = log_returns(df_setor)
    df_log_returns2 = df_log_returns.dropna()

    # VaR histórico
    lst_var_hist_99 = []
    lst_var_hist_95 = []
    lst_var_hist_90 = []

    for empresa in df_log_returns2:
        # VaR histórico 99 
        var_hist_99 = round(abs(np.percentile(df_log_returns2[empresa], 1)) * 100, 2)
        lst_var_hist_99.append(var_hist_99)
        # VaR histórico 95 
        var_hist_95 = round(abs(np.percentile(df_log_returns2[empresa], 5)) * 100, 2)
        lst_var_hist_95.append(var_hist_95)
        # VaR histórico 90 
        var_hist_90 = round(abs(np.percentile(df_log_returns2[empresa], 10)) * 100, 2)
        lst_var_hist_90.append(var_hist_90)

    # Criando um df do VaR histórico
    df_var_hist = pd.DataFrame([lst_var_hist_90, lst_var_hist_95, lst_var_hist_99], columns=df_log_returns2.columns).T
    df_var_hist = df_var_hist.rename(columns={0: 'VaR_hist_90', 1: 'VaR_hist_95', 2: 'VaR_hist_99'})

    # VaR paramétrico (gaussiano)
    lst_var_param_99 = []
    lst_var_param_95 = []
    lst_var_param_90 = []

    for empresa in df_log_returns2:
        # VaR paramétrico 99 
        var_99_param = round(abs(sps.norm.ppf(0.01, df_log_returns2[empresa].mean(), df_log_returns2[empresa].std())) * 100, 2)
        lst_var_param_99.append(var_99_param)
        # VaR paramétrico 95
        var_95_param = round(abs(sps.norm.ppf(0.05, df_log_returns2[empresa].mean(), df_log_returns2[empresa].std())) * 100, 2)
        lst_var_param_95.append(var_95_param)
        # VaR paramétrico 90
        var_90_param = round(abs(sps.norm.ppf(0.1, df_log_returns2[empresa].mean(), df_log_returns2[empresa].std())) * 100, 2)
        lst_var_param_90.append(var_90_param)

    # Criando um df do VaR paramétrico
    df_var_param = pd.DataFrame([lst_var_param_90, lst_var_param_95, lst_var_param_99], columns=df_log_returns2.columns).T
    df_var_param = df_var_param.rename(columns={0: 'VaR_param_90', 1: 'VaR_param_95', 2: 'VaR_param_99'})

    return df_var_hist, df_var_param


def CVaR(df_setor: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    df_setor: pd.DataFrame
        DataFrame que contém os preços de fechamento das empresas do setor selecionado.

    Returns
    -------
    df_cvar: pd.DataFrame
        DataFrame do CVaR.

    NOTE: é a média de todos os valores que excederam o VaR.
    """
    # Calculando o retorno logarítmico
    df_log_returns = log_returns(df_setor)
    df_log_returns2 = df_log_returns.dropna()

    # VaR paramétrico (gaussiano)
    df_var_param = VaR(df_setor)[1]

    # CVaR ou Expected Shortfall 
    cvar_99 = round(abs(df_log_returns2[df_log_returns2*100 <= -df_var_param['VaR_param_99']].mean() * 100), 2)
    cvar_95 = round(abs(df_log_returns2[df_log_returns2*100 <= -df_var_param['VaR_param_95']].mean() * 100), 2)
    cvar_90 = round(abs(df_log_returns2[df_log_returns2*100 <= -df_var_param['VaR_param_90']].mean() * 100), 2)

    # Criando um df do CVaR
    df_cvar = pd.DataFrame([cvar_90, cvar_95,cvar_99]).T
    df_cvar = df_cvar.rename(columns={0: 'CVaR_90', 1: 'CVaR_95', 2: 'CVaR_99'})

    return df_cvar


def VaR_CVaR_scipy(df_setor: pd.DataFrame, num_alpha: float, tipo_distrib: str) -> pd.DataFrame:
    """
    Parameters
    ----------
    df_setor: pd.DataFrame
        DataFrame que contém os preços de fechamento das empresas do setor selecionado.
    num_alpha: float
        Intervalo de confiança.
    tipo_distrib: str
        Tipo da distribuição (normal ou student-t).

    Returns
    -------
    df_var_cvar_t: pd.DataFrame
        DataFrame que contém o VaR e o CVaR.

    NOTE: a diferença dessa função para as funções 'VaR' e 'CVaR' é de que a média e o desvio-padrão são calculados pela função
    'sps.norm.fit'. As outras funções calculam média e o desvio-padrão com as funções '.mean()' e '.std()'.
    https://github.com/playgrdstar/expected_shortfall/blob/master/Expected%20Shortfall.ipynb
    """

    df_log_returns = log_returns(df_setor)
    df_log_returns2 = df_log_returns.dropna()

    # Intervalo de confiança 
    alpha = num_alpha

    if tipo_distrib == 'normal':
        # Calculando o 'mu' e 'sigma' de cada empresa
        lst_mu_norm = []
        lst_sig_norm = []

        for empresa in df_log_returns2:
            # Utilizando a função 'scipy.stats.fit' para extrair a média e a volatilidade mais próximo de uma distribuição normal do ativo
            mu_norm, sig_norm = sps.norm.fit(df_log_returns2[empresa])
            lst_mu_norm.append(mu_norm)
            lst_sig_norm.append(sig_norm)

        # Criando o df do 'mu' e 'sigma'
        df_mu_sig_norm = pd.DataFrame([lst_mu_norm, lst_sig_norm], columns=df_log_returns2.columns).T
        df_mu_sig_norm = df_mu_sig_norm.rename(columns={0:'mu_norm', 1:'sig_norm'})

        # Calculando o VaR e CVaR
        lst_var_norm_empresas = []
        lst_cvar_norm_empresas = []

        for empresa in df_mu_sig_norm.T:
            # Calculando o VaR normal
            var_norm_empresas = round((sps.norm.ppf(1-alpha)*df_mu_sig_norm.loc[empresa, 'sig_norm'] - df_mu_sig_norm.loc[empresa, 'mu_norm']) * 100, 2)
            lst_var_norm_empresas.append(var_norm_empresas)

            # Calculando o CVaR normal
            cvar_norm_empresa = round((alpha**-1 * sps.norm.pdf(sps.norm.ppf(alpha))*df_mu_sig_norm.loc[empresa, 'sig_norm'] - df_mu_sig_norm.loc[empresa, 'mu_norm']) * 100, 2)
            lst_cvar_norm_empresas.append(cvar_norm_empresa)

        # Criando a string do nº do intervalo de confiança -> (alpha=0.1 -> VaR_90), (alpha=0.05 -> VaR_95) e (alpha=0.01 -> VaR_99)
        num_alpha_str = str(int((1 - alpha) * 100))

        # Criando o df do VaR e CVaR
        df_var_cvar_norm = pd.DataFrame([lst_var_norm_empresas, lst_cvar_norm_empresas], columns=df_log_returns2.columns).T
        df_var_cvar_norm = df_var_cvar_norm.rename(columns={0: f'VaR_norm_{num_alpha_str}', 1: f'CVaR_norm_{num_alpha_str}'})

        return df_var_cvar_norm

    if tipo_distrib == 'student-t':
        # Calculando o 'mu', 'sigma' e 'nu' de cada empresa
        lst_mu_t = []
        lst_sig_t = []
        lst_nu = []

        for empresa in df_log_returns2:
            nu, mu_t, sig_t = sps.t.fit(df_log_returns2[empresa])
            nu = np.round(nu)
            lst_mu_t.append(mu_t)
            lst_sig_t.append(sig_t)
            lst_nu.append(nu)

        # Criando o df do 'mu', 'sigma' e 'nu'
        df_mu_sig_nu_t = pd.DataFrame([lst_mu_t, lst_sig_t, lst_nu], columns=df_log_returns2.columns).T
        df_mu_sig_nu_t = df_mu_sig_nu_t.rename(columns={0:'mu_t', 1:'sig_t', 2:'nu'})

        # Calculando o VaR e CVaR
        h = 1
        xanu = sps.t.ppf(alpha, nu)

        lst_var_t_empresas = []
        lst_cvar_norm_empresas = []

        for empresa in df_mu_sig_nu_t.T:
            # Calculando o VaR student-t
            var_t_empresas = np.sqrt((df_mu_sig_nu_t.loc[empresa, 'nu']-2) / df_mu_sig_nu_t.loc[empresa, 'nu']) * sps.t.ppf(1-alpha, df_mu_sig_nu_t.loc[empresa, 'nu'])*df_mu_sig_nu_t.loc[empresa, 'sig_t'] - h*df_mu_sig_nu_t.loc[empresa, 'mu_t']
            var_t_empresas = round(var_t_empresas * 100, 2)
            lst_var_t_empresas.append(var_t_empresas)

            # Calculando o CVaR student-t
            cvar_t_empresas = -1/alpha * (1-df_mu_sig_nu_t.loc[empresa, 'nu'])**(-1) * (df_mu_sig_nu_t.loc[empresa, 'nu']-2+xanu**2) * sps.t.pdf(xanu, df_mu_sig_nu_t.loc[empresa, 'nu'])*df_mu_sig_nu_t.loc[empresa, 'sig_t']  - h*df_mu_sig_nu_t.loc[empresa, 'mu_t']
            cvar_t_empresas = round(cvar_t_empresas * 100, 2)
            lst_cvar_norm_empresas.append(cvar_t_empresas)

        # Criando a string do nº do intervalo de confiança -> (alpha=0.1 -> VaR_90), (alpha=0.05 -> VaR_95) e (alpha=0.01 -> VaR_99)
        num_alpha_str = str(int((1 - alpha) * 100))

        # Criando o df do VaR e CVaR
        df_var_cvar_t = pd.DataFrame([lst_var_t_empresas, lst_cvar_norm_empresas], columns=df_log_returns2.columns).T
        df_var_cvar_t = df_var_cvar_t.rename(columns={0: f'VaR_t_{num_alpha_str}', 1: f'CVaR_t_{num_alpha_str}'})

        return df_var_cvar_t


def hurst_exponent(time_series: pd.DataFrame|pd.Series, max_lag: list) -> float:
    """
    Parameters
    ----------
    times_series: pd.DataFrame | pd.Series
        Série temporal.
    mas_lag: list
        Lista com o número de lags.

    Returns
    -------
    float
        Expoente de Hurst da série temporal.

    NOTE: Hurst < 0.5 - a série é anti-persistente (em reversão a média).
          Hurst = 0.5 - a série é aleatória.
          Hurst > 0.5 - a série é persistente (em tendência).

          https://quantbrasil.com.br/descobrindo-as-tendencias-do-ibovespa-utilizando-o-expoente-de-hurst/
    """
    lags = range(2, max_lag)

    # Variances of the lagged differences
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]

    # Calculate the slope of the log plot -> the Hurst Exponent
    reg = np.polyfit(np.log(lags), np.log(tau), 1)

    return reg[0]


# Funções fundamentalistas
def indicador_vm(ten_k: bool, price: pd.Series, df_is: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    price: pd.Series
        Série pandas dos preços do ativo.
    df_is: pd.DataFrame
        DataFrame do 'income statement' (DRE).

    Returns
    -------
    vm: pd.Series
        Série pandas do indicador valor de mercado.
    vm_10q: pd.Series
        Série pandas do indicador valor de mercado trimestral.

    NOTE: Eu estou multiplicando por 1000 para ficar na escala certa do valor de mercado.
    Contudo, ele ficará diferente da escala dos demais números do 'income_statement'. Por isso,
    eu tenho que dividir o 'valor de mercado' por 1000 no indicador EV/ebitda.
    """
    if ten_k:
        # Calculando o valor de mercado
        vm = (price.values * df_is.loc['Number of Shares - Basic']) * 1000
        return vm
    else:
        # Calculando o valor de mercado e retirando os dados de 2020
        vm_10q = (price.values * df_is.loc['Number of Shares - Basic'][0:len(df_is.columns)-3]) * 1000
        return vm_10q
    

def indicador_vm_2(ten_k: bool, price: pd.Series, df_is: pd.DataFrame) -> pd.Series:
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    price: pd.Series
        Série pandas dos preços do ativo.
    df_is: pd.DataFrame
        DataFrame do 'income statement' (DRE).

    Returns
    -------
    vm: pd.Series
        Série pandas do indicador valor de mercado.
    vm_10q: pd.Series
        Série pandas do indicador valor de mercado trimestral.

    NOTE: Eu não estou multiplicando por 1000, porque no income statement o valor do nº ações está na escala real.
    Apple e Vistra são empresas que eu utilizo essa função.
    """
    if ten_k:
        # Calculando o valor de mercado
        vm = price.values * df_is.loc['Number of Shares - Basic']
        return vm
    else:
        # Calculando o valor de mercado e retirando os dados de 2020
        vm_10q = price.values * df_is.loc['Number of Shares - Basic'][0:len(df_is.columns)-3]
        return vm_10q


def indicador_lpa(ten_k: bool, df_is: pd.DataFrame, idx: list) -> pd.Series|tuple[pd.Series, pd.Series]:
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    df_is: pd.DataFrame
        DataFrame do 'income statement' (DRE).
    idx: list
        Lista que contém as datas que serão o index do indicador.

    Returns
    -------
    lpa: pd.Series
        Série pandas do indicador LPA.
    lpa_acumulado: pd.Series
        Série pandas do indicador LPA acumulado trimestral.
    """
    if ten_k:
        # Calculand o LPA
        lpa = round(df_is.loc['Net Income'] / df_is.loc['Number of Shares - Basic'], 2)
        return lpa
    else:
        # LPA
        lpa_10q = round(df_is.loc['Net Income'] / df_is.loc['Number of Shares - Basic'], 2)
        # Calculando o LPA acumulado
        lst_lpa_acum_10q = []
        for i in range(len(lpa_10q) - 3):
            lpa_acum = lpa_10q[i:i+4].sum()  
            lst_lpa_acum_10q.append(round(lpa_acum,2))
        # Criando uma serie do lpa acumulado
        lpa_acumulado = pd.Series(lst_lpa_acum_10q, index=idx[0:len(idx)-3])
        return lpa_10q, lpa_acumulado
    

def indicador_lpa_2(ten_k: bool, df_is: pd.DataFrame, idx: list) -> pd.Series|tuple[pd.Series, pd.Series]:
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    df_is: pd.DataFrame
        DataFrame do 'income statement' (DRE).
    idx: list
        Lista que contém as datas que serão o index do indicador.

    Returns
    -------
    lpa: pd.Series
        Série pandas do indicador LPA.
    lpa_acumulado: pd.Series
        Série pandas do indicador LPA acumulado trimestral.

    NOTE: Eu estou dividindo por 1000, porque no income statement o valor do nº ações está numa escala diferente.
    Apple e Vistra são empresas que eu utilizo essa função.
    """
    if ten_k:
        # Calculand o LPA
        lpa = round(df_is.loc['Net Income'] / (df_is.loc['Number of Shares - Basic']/1000), 2)
        return lpa
    else:
        # LPA
        lpa_10q = round(df_is.loc['Net Income'] / (df_is.loc['Number of Shares - Basic']/1000), 2)
        # Calculando o LPA acumulado
        lst_lpa_acum_10q = []
        for i in range(len(lpa_10q) - 3):
            lpa_acum = lpa_10q[i:i+4].sum()  
            lst_lpa_acum_10q.append(round(lpa_acum,2))
        # Criando uma serie do lpa acumulado
        lpa_acumulado = pd.Series(lst_lpa_acum_10q, index=idx[0:len(idx)-3])
        return lpa_10q, lpa_acumulado


def indicador_p_lpa(ten_k: bool, price: pd.Series, lpa: pd.Series, idx: list) -> pd.Series:
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    price: pd.Series
        Série pandas dos preços do ativo.
    lpa: pd.Series
        Série pandas do indicador LPA.
    idx: list
        Lista que contém as datas que serão o index do indicador.

    Returns
    -------
    p_lpa: pd.Series
        Série pandas do indicador P/L.
    p_lpa_10q: pd.Series
        Série pandas do indicador P/L trimestral.
    """
    # Garante que são numéricos
    price = pd.to_numeric(price, errors='coerce')
    lpa = pd.to_numeric(lpa, errors='coerce')

    # Evita divisão por zero ou LPA negativo
    lpa_adjusted = lpa.copy()
    lpa_adjusted[lpa_adjusted <= 0] = np.nan  

    # Calculando o P/L
    indicador_p_lpa = (price / lpa_adjusted).round(2)

    if ten_k:
        p_lpa = pd.Series(indicador_p_lpa, index=idx).fillna(0)
        return p_lpa

    else:
        p_lpa_10q = pd.Series(indicador_p_lpa, index=idx[0:len(idx)-3]).fillna(0)
        return p_lpa_10q


def indicador_p_lpa_2(ten_k: bool, vm: pd.Series, df_is: pd.DataFrame) -> pd.Series:
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    vm: pd.Series
        Série pandas do indicador valor de mercado.
    df_is: pd.DataFrame
        DataFrame do 'income statement' (DRE).

    Returns
    -------
    p_lpa: pd.Series
        Série pandas do indicador P/L.
    p_lpa_10q: pd.Series
        Série pandas do indicador P/L trimestral.

    NOTE: o 'valor de mercado' tem que dividir por 1000, porque ele está na escala diferente
    dos demais números do 'income_statement'.
    """
    if ten_k:
        # Calculando o P/L
        p_lpa_2 = round((vm.values/1000) / (df_is.loc['Net Income']), 2)

        return p_lpa_2

    else:
        # Calculando o P/L
        p_lpa_2_10q = round((vm.values/1000) / (df_is.loc['Net Income'][:-3]), 2)

        return p_lpa_2_10q


def indicador_lpa_p(ten_k: bool, price: pd.Series, lpa: pd.Series, idx: list) -> pd.Series:
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    price: pd.Series
        Série pandas dos preços do ativo.
    lpa: pd.Series
        Série pandas do indicador LPA.
    idx: list
        Lista que contém as datas que serão o index do indicador.

    Returns
    -------
    lpa_p: pd.Series
        Série pandas do indicador L/P (Earnings Yield).
    lpa_p_10q: pd.Series
        Série pandas do indicador L/P (Earnings Yield) trimestral.
    """
    # Garante que são numéricos
    price = pd.to_numeric(price, errors='coerce')
    lpa = pd.to_numeric(lpa, errors='coerce')

    # Calculando o L/P (Earnings Yield)
    indicador_lpa_p = ((lpa / price) * 100).round(2)

    if ten_k:
        # Transformando o indicador em uma serie pandas
        lpa_p = pd.Series(indicador_lpa_p, index=idx) 
        return lpa_p
    else:
        # Transformando o indicador em uma serie pandas
        lpa_p_10q = pd.Series(indicador_lpa_p, index=idx[0:len(idx)-3]) 
        return lpa_p_10q
    

def indicador_depreciacao(ten_k: bool, df_cf: pd.Series, first_quarter: str, idx: list) -> pd.Series|tuple[pd.Series, pd.Series, pd.Series]:
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    df_cf: pd.DataFrame
        DataFrame do 'cash flow statement' (fluxo de caixa).
    first_quarter: str
        Indicar qual é o número do 1Q da empresa.
    idx: list
        Lista que contém as datas que serão o index do indicador.

    Returns
    -------
    depreciacao: pd.Series
        Série pandas do indicador depreciação.
    depreciacao_not_accum_10q: pd.Series
        Série pandas do indicador depreciação trimestral não acumulada.
    depreciacao_not_accum_10q_sliced: pd.Series
        Série pandas do indicador depreciação trimestral não acumulada cortada.
    depreciacao_accum_10q: pd.Series
        Série pandas do indicador depreciação trimestral acumulada.
    """
    if ten_k:
        # Calculando a depreciação
        depreciacao = df_cf.loc['Depreciation, Depletion and Amortization']
        return depreciacao
    else:
        # Calculando a depreciação 
        depreciacao_10q = df_cf.loc['Depreciation, Depletion and Amortization']
        
        # Calculando a depreciação não acumulado do 2Q e 3Q
        lst_depreciacao_not_accum = {}

        for current_idx, item in depreciacao_10q.items():
            month, year = current_idx.month, current_idx.year

            # Condição para as empresas que fazem o lançamento nos meses 02, 05 e 08
            if first_quarter == '02': 
                if month == 5:
                    # Calculando a depreciação não acumulada do mês 05, -> "mês 05" - "mês 02" 
                    depreciacao_05 = item - (depreciacao_10q[f'2-{year}'].values[0])
                    lst_depreciacao_not_accum[current_idx] = depreciacao_05

                if month == 8:
                    # Calculando a depreciação não acumulada do mês 08 -> "mês 08" - ("mês 05" - "mês 02") - "mês 02"        
                    depreciacao_08 = item - (depreciacao_10q[f'5-{year}'].values[0] - depreciacao_10q[f'2-{year}'].values[0]) - depreciacao_10q[f'2-{year}'].values[0]
                    lst_depreciacao_not_accum[current_idx] = depreciacao_08

                # Contem apenas as depreciações não acumuladas dos meses 05 e 08
                depreciacao_not_accum = pd.Series(lst_depreciacao_not_accum, dtype=float)
                # Adicionando a depreciação do mês 02 nesta serie de depreciações não acumuladas 
                depreciacao_10q_feb = depreciacao_10q[depreciacao_10q.index.month == 2]
                depreciacao_not_accum_10q = pd.concat([depreciacao_not_accum, depreciacao_10q_feb], sort=False)
                depreciacao_not_accum_10q = depreciacao_not_accum_10q.sort_index(ascending=False)

                # Condição para as empresas que fazem o lançamento nos meses 03, 06 e 09
            if first_quarter == '03': 
                if month == 6:
                    # Calculando a depreciação não acumulada do mês 06 -> "mês 06" - "mês 03"
                    depreciacao_06 = item - (depreciacao_10q[f'3-{year}'].values[0])
                    lst_depreciacao_not_accum[current_idx] = depreciacao_06

                if month == 9:
                    # Calculando a depreciação não acumulada do mês 09 -> "mês 09" - ("mês 06" - "mês 03") - "mês 03"
                    depreciacao_09 = item - (depreciacao_10q[f'6-{year}'].values[0] - depreciacao_10q[f'3-{year}'].values[0]) - depreciacao_10q[f'3-{year}'].values[0]
                    lst_depreciacao_not_accum[current_idx] = depreciacao_09
                        
                # Contem apenas as depreciações não acumuladas dos meses 06 e 09
                depreciacao_not_accum = pd.Series(lst_depreciacao_not_accum, dtype=float)
                # Adicionando a depreciação do mês 03 nesta serie de depreciações não acumuladas 
                depreciacao_10q_mar = depreciacao_10q[depreciacao_10q.index.month == 3]
                depreciacao_not_accum_10q = pd.concat([depreciacao_not_accum, depreciacao_10q_mar], sort=False) 
                # Ordenando a serie para que o index fique igual aos outros indicadores
                depreciacao_not_accum_10q = depreciacao_not_accum_10q.sort_index(ascending=False)

            # Condição para as empresas que fazem o lançamento nos meses 04, 07 e 10
            if first_quarter == '04': 
                if month == 7:
                    # Calculando a depreciação não acumulada do mês 07, -> "mês 07" - "mês 04" 
                    depreciacao_07 = item - (depreciacao_10q[f'4-{year}'].values[0])
                    lst_depreciacao_not_accum[current_idx] = depreciacao_07

                if month == 10:
                    # Calculando a depreciação não acumulada do mês 10 -> "mês 10" - ("mês 07" - "mês 04") - "mês 04"        
                    depreciacao_10 = item - (depreciacao_10q[f'7-{year}'].values[0] - depreciacao_10q[f'4-{year}'].values[0]) - depreciacao_10q[f'4-{year}'].values[0]
                    lst_depreciacao_not_accum[current_idx] = depreciacao_10

                # Contem apenas as depreciações não acumuladas dos meses 07 e 10
                depreciacao_not_accum = pd.Series(lst_depreciacao_not_accum, dtype=float)
                # Adicionando a depreciação do mês 04 nesta serie de depreciações não acumuladas 
                depreciacao_10q_apr = depreciacao_10q[depreciacao_10q.index.month == 4]
                depreciacao_not_accum_10q = pd.concat([depreciacao_not_accum, depreciacao_10q_apr], sort=False)
                depreciacao_not_accum_10q = depreciacao_not_accum_10q.sort_index(ascending=False)

            # Condição para as empresas que fazem o lançamento nos meses 09, 12 e 03
            if first_quarter == '09':
                if month == 12:
                    # Calculando a depreciação não acumulada do mês 12 -> "mês 12" - "mês 09"
                    depreciacao_12 = item - (depreciacao_10q[f'9-{year}'].values[0])
                    lst_depreciacao_not_accum[current_idx] = depreciacao_12

                if month == 3:
                    # Calculando o ano anterior, porque o mês 3 já está no próximo ano
                    year_before = year - 1 

                    # Calculando a depreciação não acumulada do mês 03 -> "mês 03 do próximo ano" - ("mês 12" - "mês 09") - "mês 09"
                    depreciacao_03 = item - (depreciacao_10q[f'12-{year_before}'].values[0] - depreciacao_10q[f'9-{year_before}'].values[0]) - depreciacao_10q[f'9-{year_before}'].values[0]
                    lst_depreciacao_not_accum[current_idx] = depreciacao_03

                # Contem apenas as depreciações não acumuladas dos meses 12 e 3
                depreciacao_not_accum = pd.Series(lst_depreciacao_not_accum, dtype=float)
                # Adicionando a depreciação do mês 09 nesta serie de depreciações não acumuladas 
                depreciacao_10q_sep = depreciacao_10q[depreciacao_10q.index.month == 9]
                depreciacao_not_accum_10q = pd.concat([depreciacao_not_accum, depreciacao_10q_sep], sort=False)
                # Ordenando a serie para que o index fique igual aos outros indicadores
                depreciacao_not_accum_10q = depreciacao_not_accum_10q.sort_index(ascending=False)

            # Condição para as empresas que fazem o lançamento nos meses 12, 03 e 06
            if first_quarter == '12':
                if month == 3:
                    # Calculando o ano anterior, porque o 1Q é o mês 12
                    year_before = year - 1
                    # Calculando a depreciação não acumulada do mês 03 -> "mês 03" - "mês 12 do ano anterior"
                    depreciacao_03 = item - (depreciacao_10q[f'12-{year_before}'].values[0])
                    lst_depreciacao_not_accum[current_idx] = depreciacao_03
        
                if month == 6:
                    # Calculando o ano anterior, porque o 1Q é o mês 12
                    year_before = year - 1
                    # Calculando a depreciação não acumulada do mês 06 -> "mês 06" - ("mês 03" - "mês 12 do ano anterior") - "mês 12 do ano anterior"
                    depreciacao_06 = item - (depreciacao_10q[f'3-{year}'].values[0] - depreciacao_10q[f'12-{year_before}'].values[0]) - depreciacao_10q[f'12-{year_before}'].values[0]
                    lst_depreciacao_not_accum[current_idx] = depreciacao_06

                # Contem apenas as depreciações não acumuladas dos meses 03 e 06
                depreciacao_not_accum = pd.Series(lst_depreciacao_not_accum, dtype=float)
                # Adicionando a depreciação do mês 12 nesta serie de depreciações não acumuladas 
                depreciacao_10q_dec = depreciacao_10q[depreciacao_10q.index.month == 12]
                depreciacao_not_accum_10q = pd.concat([depreciacao_not_accum, depreciacao_10q_dec], sort=False)
                depreciacao_not_accum_10q = depreciacao_not_accum_10q.sort_index(ascending=False)

        # Retirando os dados de 2020
        depreciacao_not_accum_10q_sliced = depreciacao_not_accum_10q[0:len(depreciacao_not_accum_10q)-3]

        # Calculando a depreciação acumulada
        lst_depreciacao_accum_10q = []
        for i in range(len(depreciacao_not_accum_10q) - 3):
            capex_accum = depreciacao_not_accum_10q[i:i+4].sum()  
            lst_depreciacao_accum_10q.append(capex_accum)
        # Criando uma serie da depreciação acumulada
        depreciacao_accum_10q = pd.Series(lst_depreciacao_accum_10q, index=idx[0:len(idx)-3])

        return depreciacao_not_accum_10q, depreciacao_not_accum_10q_sliced, depreciacao_accum_10q


def indicador_ebitda(ten_k: bool, df_is: pd.DataFrame, idx: list, depreciacao: pd.Series) -> pd.Series|tuple[pd.Series, pd.Series, pd.Series]:
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    df_is: pd.DataFrame
        DataFrame do 'income statement' (DRE).
    idx: list
        Lista que contém as datas que serão o index do indicador.
    depreciacao: pd.Series
        Série pandas da depreciação não acumulada.

    Returns
    -------
    ebitda: pd.Series
        Série pandas do indicador ebitda.
    ebitda_10q: pd.Series
        Série pandas do indicador ebitda trimestral.
    ebitda_sliced: pd.Series
        Série pandas do indicador ebitda trimestral, excluindo os dados de 2020.
    ebitda_accum_10q: pd.Series
        Série pandas do indicador ebitda trimestral acumulado.

    NOTE: para os dados de 10-Q, eu retorno dois tipos de ebitda. O ebitda sem cortes serve 
    para calcular o ebitda acumulado no indicador EV/EBITDA. O ebitda cortado é para concatenar
    na tabela final dos indicadores.
    """
    if ten_k:
        # Calculando o EBITDA
        ebitda = df_is.loc['Operating Income'] + depreciacao
        return ebitda
    
    else:
        # Calculando o EBITDA
        ebitda_10q = df_is.loc['Operating Income'] + depreciacao
        # Retirando os dados de 2020
        ebitda_sliced = ebitda_10q[0:len(ebitda_10q)-3]
        # Calculando o EBITDA acumulado
        lst_ebitda_acum_10q = []
        for i in range(len(ebitda_10q) - 3):
            ebitda_accum = ebitda_10q[i:i+4].sum()  
            lst_ebitda_acum_10q.append(ebitda_accum)
        # Criando uma serie do ebitda acumulado
        ebitda_accum_10q = pd.Series(lst_ebitda_acum_10q, index=idx[0:len(idx)-3])
        return ebitda_10q, ebitda_sliced, ebitda_accum_10q
    

def indicador_ml(ten_k: bool, df_is: pd.DataFrame) -> pd.Series:
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    df_is: pd.DataFrame
        DataFrame do 'income statement' (DRE).

    Returns
    -------
    m_liq: pd.Series
        Série pandas do indicador margem líquida.
    m_liq_10q: pd.Series
        Série pandas do indicador margem líquida trimestral.
    """
    # Evita divisão por zero ou LPA negativo
    revenues_adjusted = df_is.loc['Revenues'].copy()
    revenues_adjusted[revenues_adjusted <= 0] = np.nan  

    if ten_k:
        # Calculando a margem líquida
        m_liq = np.round((df_is.loc['Net Income'] / revenues_adjusted) * 100, 2).fillna(0)
        return m_liq
    else:
        # Calculando a margem líquida
        m_liq_10q = np.round((df_is.loc['Net Income'] / revenues_adjusted) * 100, 2).fillna(0)
        # Retirando os dados de 2020
        m_liq_10q = m_liq_10q[0:len(m_liq_10q)-3]
        return m_liq_10q

    
def indicador_vpa(ten_k: bool, df_bs: pd.DataFrame, df_is: pd.DataFrame) -> pd.Series:
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    df_bs: pd.DataFrame
        DataFrame do 'balance sheet' (balanço patrimonial).
    df_is: pd.DataFrame
        DataFrame do 'income statement' (DRE).

    Returns
    -------
    vpa: pd.Series
        Série pandas do indicador vpa.
    vpa_10q: pd.Series
        Série pandas do indicador vpa acumulado trimestral.
    """
    if ten_k:
        # Calculando o VPA
        vpa = np.round(df_bs.loc['Total Shareholders Equity'] / df_is.loc['Number of Shares - Basic'], 2)
        return vpa
    else:
        # Calculando o VPA
        vpa_10q = np.round(df_bs.loc['Total Shareholders Equity'] / df_is.loc['Number of Shares - Basic'], 2)
        # Retirando os dados de 2020
        vpa_10q = vpa_10q[0:len(vpa_10q)-3]
        return vpa_10q
    

def indicador_vpa_2(ten_k: bool, df_bs: pd.DataFrame, df_is: pd.DataFrame) -> pd.Series:
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    df_bs: pd.DataFrame
        DataFrame do 'balance sheet' (balanço patrimonial).
    df_is: pd.DataFrame
        DataFrame do 'income statement' (DRE).

    Returns
    -------
    vpa: pd.Series
        Série pandas do indicador vpa.
    vpa_10q: pd.Series
        Série pandas do indicador vpa acumulado trimestral.
    NOTE: Eu estou dividindo por 1000, porque no income statement o valor do nº ações está numa escala diferente.
    Apple e Vistra são empresas que eu utilizo essa função.
    """
    if ten_k:
        # Calculando o VPA
        vpa = np.round(df_bs.loc['Total Shareholders Equity'] / (df_is.loc['Number of Shares - Basic']/1000), 2)
        return vpa
    else:
        # Calculando o VPA
        vpa_10q = np.round(df_bs.loc['Total Shareholders Equity'] / (df_is.loc['Number of Shares - Basic']/1000), 2)
        # Retirando os dados de 2020
        vpa_10q = vpa_10q[0:len(vpa_10q)-3]
        return vpa_10q


def indicador_p_vpa(ten_k: bool, price: pd.Series, vpa: pd.Series, idx: list) -> pd.Series:
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    price: pd.Series
        Série pandas dos preços do ativo.
    vpa: pd.Series
        Série pandas do indicador VPA.
    idx: list
        Lista que contém as datas que serão o index do indicador.

    Returns
    -------
    p_vpa: pd.Series
        Série pandas do indicador P/VPA.
    p_vpa_10q: pd.Series
        Série pandas do indicador P/VPA trimestral.
    """
    # Garante que são numéricos
    price = pd.to_numeric(price, errors='coerce')
    vpa = pd.to_numeric(vpa, errors='coerce')

    # Evita divisão por zero ou VPA negativo
    vpa_adjusted = vpa.copy()
    vpa_adjusted[vpa_adjusted <= 0] = np.nan

    # Calculando o P/VPA
    indicador_p_vpa = (price / vpa_adjusted).round(2)

    if ten_k:
        p_vpa = pd.Series(indicador_p_vpa, index=idx).fillna(0) 
        return p_vpa
    else:
        p_vpa_10q = pd.Series(indicador_p_vpa, index=idx[0:len(idx)-3]).fillna(0)
        return p_vpa_10q 


def indicador_caixa(ten_k: bool, df_bs: pd.DataFrame, lst_itens_caixa: list) -> pd.Series:
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    df_bs: pd.DataFrame
        DataFrame do 'balance sheet' (balanço patrimonial).
    lst_itens_caixa: list
        Lista dos itens do 'balance sheet' (balanço patrimonial) que formam o caixa e equivalentes.

    Returns
    -------
    caixa: pd.Series
        Série pandas do indicador caixa e equivalentes.
    caixa_10q: pd.Series
        Série pandas do indicador caixa e equivalentes trimestral.
    """
    if ten_k:
        # Calculando o caixa e equivalentes para 10-K
        caixa = abs(df_bs.loc[lst_itens_caixa]).sum()
        return caixa
    else:
        # Calculando o caixa e equivalentes para 10-Q
        caixa_10q = abs(df_bs.loc[lst_itens_caixa]).sum()
        # Retirando os dados de 2020
        caixa_10q = caixa_10q[0:len(caixa_10q)-3]
        return caixa_10q
    

def indicador_divida_bruta(ten_k: bool, df_bs: pd.DataFrame, lst_itens_divida: list) -> pd.Series:
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    df_bs: pd.DataFrame
        DataFrame do 'balance sheet' (balanço patrimonial).
    lst_itens_divida: list
        Lista dos itens do 'balance sheet' (balanço patrimonial) que formam a dívida bruta.

    Returns
    -------
    div_bruta: pd.Series
        Série pandas do indicador dívida bruta.
    div_bruta_10q: pd.Series
        Série pandas do indicador dívida bruta trimestral.
    """
    if ten_k:
        # Calculando a dívida bruta
        div_bruta = abs(df_bs.loc[lst_itens_divida]).sum()
        return div_bruta
    else:
        # Calculando a dívida bruta
        div_bruta_10q = abs(df_bs.loc[lst_itens_divida]).sum()
        # Retirando os dados de 2020
        div_bruta_10q = div_bruta_10q[0:len(div_bruta_10q)-3]
        return div_bruta_10q
    

def indicador_divida_liquida(ten_k: bool, caixa: pd.Series, div_bruta: pd.Series, idx: list) -> pd.Series:
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    caixa: pd.Series
        Série pandas do indicador caixa e equivalentes.
    div_bruta: pd.Series
        Série pandas do indicador dívida bruta.
    idx: list
        Lista que contém as datas que serão o index do indicador.

    Returns
    -------
    div_liq: pd.Series
        Série pandas do indicador dívida líquida.
    div_liq_10q: pd.Series
        Série pandas do indicador dívida líquida trimestral.
    """
    if ten_k:
        # Calculando a dívida líquida
        div_liq = div_bruta.values - caixa.values
        # Transformando em uma série pandas
        div_liq = pd.Series(div_liq, index=idx) 
        return div_liq
    else:
        #  Calculando a dívida líquida
        div_liq_10q = div_bruta.values - caixa.values
        # Transformando em uma série pandas
        div_liq_10q = pd.Series(div_liq_10q, index=idx[0:len(idx)-3]) 
        return div_liq_10q
    

def indicador_dl_ebitda(ten_k:bool, ebitda: pd.Series, div_liq: pd.Series, idx: list) -> pd.Series:
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    ebitda: pd.Series
        Série pandas do indicador ebitda.
    div_liq: pd.Series
        Série pandas do indicador dívida líquida.
    idx: list
        Lista que contém as datas que serão o index do indicador.

    Returns
    -------
    dl_ebitda: pd.Series
        Série pandas do indicador dívida líquida/ebitda.
    dl_ebitda: pd.Series
        Série pandas do indicador dívida líquida/ebitda trimestral.
    """
    if ten_k:
        # Calculando a Dívida Líquida / EBITDA
        dl_ebitda = np.round(div_liq.values / ebitda, 2) 
        return dl_ebitda
    else:
        # Calculando a Dívida Líquida / EBITDA
        dl_ebitda_10q = np.round(div_liq.values / ebitda.values, 2) 
        # Transformando em uma série pandas
        dl_ebitda_10q = pd.Series(dl_ebitda_10q , index=idx[0:len(idx)-3]) 
        return dl_ebitda_10q


def indicador_dl_pl(ten_k: bool, div_liq: pd.Series, df_bs: pd.DataFrame) -> pd.Series:
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    div_liq: pd.Series
        Série pandas do indicador dívida líquida.
    df_bs: pd.DataFrame
        DataFrame do 'balance sheet' (balanço patrimonial).

    Returns
    -------
    dl_pl: pd.Series
        Série pandas do indicador dívida líquida/pl.
    dl_pl_10q: pd.Series
        Série pandas do indicador dívida líquida/pl trimestral.
    """
    if ten_k:
        # Calculando a Dívida Líquida / PL
        dl_pl = np.round(div_liq.values / df_bs.loc['Total Shareholders Equity'], 2) 
        return dl_pl
    else:
        # Calculando a Dívida Líquida / PL
        dl_pl_10q = np.round(div_liq.values / df_bs.loc['Total Shareholders Equity'][0:len(df_bs.columns)-3], 2) 
        return dl_pl_10q


def indicador_ev_ebitda(vm: pd.Series, div_liq: pd.Series, ebitda: pd.Series) -> pd.Series:
    """
    Parameters
    ----------
    vm: pd.Series
        Série pandas do indicador valor de mercado.
    div_liq: pd.Series
        Série pandas do indicador dívida líquida.
    ebitda: pd.Series
        Série pandas do indicador ebitda.

    Returns
    -------
    ev_ebitda: pd.Series
        Série pandas do indicador ev/ebitda.

    NOTE: o 'valor de mercado' tem que dividir por 1000, porque ele está na escala diferente
    dos demais números do 'income_statement'.
    """
    # Calculando o EV/EBITDA
    ev = (vm.values/1000) + abs(div_liq)
    ev_ebitda = np.round(ev.values / ebitda, 2)
    return ev_ebitda


def indicador_roe(ten_k: bool, df_is: pd.Series, df_bs: pd.Series, idx: list) -> pd.Series:
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    df_is: pd.DataFrame
        DataFrame do 'income statement' (DRE).
    df_bs: pd.DataFrame
        DataFrame do 'balance sheet' (balanço patrimonial).
    idx: list
        Lista que contém as datas que serão o index do indicador.

    Returns
    -------
    roe: pd.Series
        Série pandas do indicador roe.
    roe_10q: pd.Series
        Série pandas do indicador roe trimestral.
    """
    if ten_k:
        # Calculando o ROE
        roe = round((df_is.loc['Net Income'] / df_bs.loc['Total Shareholders Equity']) * 100, 2)
        return roe
    else:
        # Calculando o lucro líquido acumulado
        lst_lucro_liq_acum_10q = []
        for i in range(len(df_is.loc['Net Income']) - 3):
            lucro_liq_acum = df_is.loc['Net Income'][i:i+4].sum()  
            lst_lucro_liq_acum_10q.append(lucro_liq_acum)
        # Criando um df do lucro líquido acumulado
        lucro_liq_acum_final = pd.Series(lst_lucro_liq_acum_10q, index=idx[0:len(idx)-3])

        # Calculando o ROE
        roe_10q = round((lucro_liq_acum_final.values / df_bs.loc['Total Shareholders Equity'][0:len(df_bs.columns)-3]) * 100, 2)
        return roe_10q


def indicador_roic(ten_k:bool, df_is: pd.Series, df_bs: pd.Series, div_bruta: pd.Series, idx: list) -> pd.Series:
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    df_is: pd.DataFrame
        DataFrame do 'income statement' (DRE).
    df_bs: pd.DataFrame
        DataFrame do 'balance sheet' (balanço patrimonial).
    div_bruta: pd.Series 
        Série pandas do indicador dívida bruta.
    idx: list
        Lista que contém as datas que serão o index do indicador.

    Returns
    -------
    roic: pd.Series 
        Série pandas do indicador roic.
    roic_10q: pd.Series 
        Série pandas do indicador roic trimestral.
    """
    if ten_k:
        # Calculando o ROIC
        noplat = df_is.loc['Operating Income'] - df_is.loc['Provision for Income Taxes']
        cap_investido = df_bs.loc['Total Shareholders Equity'] + div_bruta
        roic = np.round((noplat.values / cap_investido.values) * 100, 2)
        # Transformando em uma série pandas
        roic = pd.Series(roic, index=idx) 
        return roic
    else:
        # Calculando o lucro operacional acumulado
        lst_lucro_op_acum_10q = []
        for i in range(len(df_is.loc['Operating Income']) - 3):
            lucro_op_acum = df_is.loc['Operating Income'][i:i+4].sum()  
            lst_lucro_op_acum_10q.append(lucro_op_acum)
        # Criando um df do lucro operacional acumulado
        lucro_op_acum_final = pd.Series(lst_lucro_op_acum_10q, index=idx[0:len(idx)-3])

        # # Calculando o imposto acumulado
        lst_imposto_acum_10q = []
        for i in range(len(df_is.loc['Provision for Income Taxes']) - 3):
            imposto_acum = df_is.loc['Provision for Income Taxes'][i:i+4].sum()  
            lst_imposto_acum_10q .append(imposto_acum)
        # Criando um df do imposto acumulado
        imposto_acum_final = pd.Series(lst_imposto_acum_10q, index=idx[0:len(idx)-3])

        # Calculando o ROIC 
        noplat_10q = lucro_op_acum_final - imposto_acum_final
        cap_investido_10q = df_bs.loc['Total Shareholders Equity'][0:len(df_bs.columns)-3] + div_bruta
        roic_10q = np.round((noplat_10q.values / cap_investido_10q.values) * 100, 2)
        # Transformando em uma série pandas
        roic_10q = pd.Series(roic_10q, index=idx[0:len(idx)-3]) 
        return roic_10q


def indicador_fco(ten_k: bool, df_cf: pd.Series, first_quarter: str) -> pd.Series:
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    df_cf: pd.DataFrame
        DataFrame do 'cash flow statement' (fluxo de caixa).
    first_quarter: str
        Indicar qual é o número do 1Q da empresa.

    Returns
    -------
    fco: pd.Series
        Série pandas do indicador fco.
    fco_10q: pd.Series
        Série pandas do indicador fco trimestral.
    """
    if ten_k:
        # Calculando o FCO
        fco = df_cf.loc['Net Cash Provided by (Used in) Operating Activities']
        return fco
    else:
        # Calculando o FCO e retirando os dados de 2020
        fco_10q = df_cf.loc['Net Cash Provided by (Used in) Operating Activities'][0:len(df_cf.columns)-3]
        
        # Calculando o FCO não acumulado do 2Q e 3Q
        lst_fco_not_accum = {}

        for idx, item in fco_10q.items():
            month, year = idx.month, idx.year

            # Condição para as empresas que fazem o lançamento nos meses 02, 05 e 08
            if first_quarter == '02': 
                if month == 5:
                    # Calculando o FCO não acumulado do mês 05, -> "mês 05" - "mês 02" 
                    fco_05 = item - (fco_10q[f'2-{year}'].values[0])
                    lst_fco_not_accum[idx] = fco_05

                if month == 8:
                    # Calculando o FCO não acumulado do mês 08 -> "mês 08" - ("mês 05" - "mês 02") - "mês 02"        
                    fco_08 = item - (fco_10q[f'5-{year}'].values[0] - fco_10q[f'2-{year}'].values[0]) - fco_10q[f'2-{year}'].values[0]
                    lst_fco_not_accum[idx] = fco_08

                # Contem apenas os FCOs não acumulados dos meses 05 e 08
                fco_not_accum = pd.Series(lst_fco_not_accum, dtype=float)
                # Adicionando o FCO do mês 02 nesta serie de FCOs não acumulados 
                fco_10q_feb = fco_10q[fco_10q.index.month == 2]
                fco_not_accum_10q = pd.concat([fco_not_accum, fco_10q_feb], sort=False)
                # Ordenando a serie para que o index fique igual aos outros indicadores
                fco_not_accum_10q = fco_not_accum_10q.sort_index(ascending=False)

                # Condição para as empresas que fazem o lançamento nos meses 03, 06 e 09
            if first_quarter == '03': 
                if month == 6:
                    # Calculando o FCO não acumulado do mês 06 -> "mês 06" - "mês 03"
                    fco_06 = item - (fco_10q[f'3-{year}'].values[0])
                    lst_fco_not_accum[idx] = fco_06

                if month == 9:
                    # Calculando o FCO não acumulado do mês 09 -> "mês 09" - ("mês 06" - "mês 03") - "mês 03"
                    fco_09 = item - (fco_10q[f'6-{year}'].values[0] - fco_10q[f'3-{year}'].values[0]) - fco_10q[f'3-{year}'].values[0]
                    lst_fco_not_accum[idx] = fco_09
                    
                # Contem apenas os FCOs não acumulados dos meses 06 e 09
                fco_not_accum = pd.Series(lst_fco_not_accum, dtype=float)
                # Adicionando o FCO do mês 03 nesta serie de FCOs não acumulados 
                fco_10q_mar = fco_10q[fco_10q.index.month == 3]
                fco_not_accum_10q = pd.concat([fco_not_accum, fco_10q_mar], sort=False) 
                # Ordenando a serie para que o index fique igual aos outros indicadores
                fco_not_accum_10q = fco_not_accum_10q.sort_index(ascending=False)

            # Condição para as empresas que fazem o lançamento nos meses 04, 07 e 10
            if first_quarter == '04': 
                if month == 7:
                    # Calculando o FCO não acumulado do mês 07, -> "mês 07" - "mês 04" 
                    fco_07 = item - (fco_10q[f'4-{year}'].values[0])
                    lst_fco_not_accum[idx] = fco_07

                if month == 10:
                    # Calculando o FCO não acumulado do mês 10 -> "mês 10" - ("mês 07" - "mês 04") - "mês 04"        
                    fco_10 = item - (fco_10q[f'7-{year}'].values[0] - fco_10q[f'4-{year}'].values[0]) - fco_10q[f'4-{year}'].values[0]
                    lst_fco_not_accum[idx] = fco_10

                # Contem apenas os FCOs não acumulados dos meses 07 e 10
                fco_not_accum = pd.Series(lst_fco_not_accum, dtype=float)
                # Adicionando o FCO do mês 04 nesta serie de FCOs não acumulados 
                fco_10q_apr = fco_10q[fco_10q.index.month == 4]
                fco_not_accum_10q = pd.concat([fco_not_accum, fco_10q_apr], sort=False)
                # Ordenando a serie para que o index fique igual aos outros indicadores
                fco_not_accum_10q = fco_not_accum_10q.sort_index(ascending=False)

            # Condição para as empresas que fazem o lançamento nos meses 09, 12 e 03
            if first_quarter == '09':
                if month == 12:
                    # Calculando o FCO não acumulado do mês 12 -> "mês 12" - "mês 09"
                    fco_12 = item - (fco_10q[f'9-{year}'].values[0])
                    lst_fco_not_accum[idx] = fco_12

                if month == 3:
                    # Calculando o ano anterior, porque o mês 3 já está no próximo ano
                    year_before = year - 1 
                    # Calculando o FCO não acumulado do mês 03 -> "mês 03 do próximo ano" - ("mês 12" - "mês 09") - "mês 09"
                    fco_03 = item - (fco_10q[f'12-{year_before}'].values[0] - fco_10q[f'9-{year_before}'].values[0]) - fco_10q[f'9-{year_before}'].values[0]
                    lst_fco_not_accum[idx] = fco_03

                # Contem apenas os FCO não acumulados dos meses 12 e 3
                fco_not_accum = pd.Series(lst_fco_not_accum, dtype=float)
                # Adicionando o FCO do mês 09 nesta serie de FCO não acumulados
                fco_10q_sep = fco_10q[fco_10q.index.month == 9]
                fco_not_accum_10q = pd.concat([fco_not_accum, fco_10q_sep], sort=False)
                # Ordenando a serie para que o index fique igual aos outros indicadores
                fco_not_accum_10q = fco_not_accum_10q.sort_index(ascending=False)

            # Condição para as empresas que fazem o lançamento nos meses 12, 03 e 06
            if first_quarter == '12': 
                if month == 3:
                    # Calculando o ano anterior, porque o 1Q é o mês 12
                    year_before = year - 1
                    # Calculando o FCO não acumulado do mês 03 -> "mês 03" - "mês 12 do ano anterior"
                    fco_03 = item - (fco_10q[f'12-{year_before}'].values[0])
                    lst_fco_not_accum[idx] = fco_03

                if month == 6:
                    # Calculando o ano anterior, porque o 1Q é o mês 12
                    year_before = year - 1
                    # Calculando o FCO não acumulado do mês 06 -> "mês 06" - ("mês 03" - "mês 12 do ano anterior") - "mês 12 do ano anterior"
                    fco_06 = item - (fco_10q[f'3-{year}'].values[0] - fco_10q[f'12-{year_before}'].values[0]) - fco_10q[f'12-{year_before}'].values[0]
                    lst_fco_not_accum[idx] = fco_06

                # Contem apenas os FCOs não acumulados dos meses 03 e 06
                fco_not_accum = pd.Series(lst_fco_not_accum, dtype=float)
                # Adicionando o FCO do mês 12 nesta serie de FCOs não acumulados 
                fco_10q_dec = fco_10q[fco_10q.index.month == 12]
                fco_not_accum_10q = pd.concat([fco_not_accum, fco_10q_dec], sort=False)
                # Ordenando a serie para que o index fique igual aos outros indicadores
                fco_not_accum_10q = fco_not_accum_10q.sort_index(ascending=False)
                
        return fco_not_accum_10q
    

def indicador_fci(ten_k: bool, df_cf: pd.Series, first_quarter: str) -> pd.Series:
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    df_cf: pd.DataFrame
        DataFrame do 'cash flow statement' (fluxo de caixa).
    first_quarter: str
        Indicar qual é o número do 1Q da empresa.

    Returns
    -------
    fci: pd.Series
        Série pandas do indicador fci.
    fci_10q: pd.Series
        Série pandas do indicador fci trimestral.
    """
    if ten_k:
        # Calculando o FCI
        fci = df_cf.loc['Net Cash Provided by (Used in) Investing Activities']
        return fci
    else:
        # Calculando o FCI e retirando os dados de 2020
        fci_10q = df_cf.loc['Net Cash Provided by (Used in) Investing Activities'][0:len(df_cf.columns)-3]

        # Calculando o FCI não acumulado do 2Q e 3Q
        lst_fci_not_accum = {}

        for idx, item in fci_10q.items():
            month, year = idx.month, idx.year

            # Condição para as empresas que fazem o lançamento nos meses 02, 05 e 08
            if first_quarter == '02': 
                if month == 5:
                    # Calculando o FCI não acumulado do mês 05, -> "mês 05" - "mês 02" 
                    fci_05 = item - (fci_10q[f'2-{year}'].values[0])
                    lst_fci_not_accum[idx] = fci_05

                if month == 8:
                    # Calculando o FCI não acumulado do mês 08 -> "mês 08" - ("mês 05" - "mês 02") - "mês 02"        
                    fci_08 = item - (fci_10q[f'5-{year}'].values[0] - fci_10q[f'2-{year}'].values[0]) - fci_10q[f'2-{year}'].values[0]
                    lst_fci_not_accum[idx] = fci_08

                # Contem apenas os FCIs não acumulados dos meses 05 e 08
                fci_not_accum = pd.Series(lst_fci_not_accum, dtype=float)
                # Adicionando o FCI do mês 02 nesta serie de FCIs não acumulados 
                fci_10q_feb = fci_10q[fci_10q.index.month == 2]
                fci_not_accum_10q = pd.concat([fci_not_accum, fci_10q_feb], sort=False)
                # Ordenando a serie para que o index fique igual aos outros indicadores
                fci_not_accum_10q = fci_not_accum_10q.sort_index(ascending=False)

            # Condição para as empresas que fazem o lançamento nos meses 03, 06 e 09
            if first_quarter == '03': 
                if month == 6:
                    # Calculando o FCI não acumulado do mês 06 -> "mês 06" - "mês 03"
                    fci_06 = item - (fci_10q[f'3-{year}'].values[0])
                    lst_fci_not_accum[idx] = fci_06

                if month == 9:
                    # Calculando o FCI não acumulado do mês 09 -> "mês 09" - ("mês 06" - "mês 03") - "mês 03"
                    fci_09 = item - (fci_10q[f'6-{year}'].values[0] - fci_10q[f'3-{year}'].values[0]) - fci_10q[f'3-{year}'].values[0]
                    lst_fci_not_accum[idx] = fci_09
                    
                # Contem apenas os FCIs não acumulados dos meses 06 e 09
                fci_not_accum = pd.Series(lst_fci_not_accum, dtype=float)
                # Adicionando o FCI do mês 03 nesta serie de FCIs não acumulados 
                fci_10q_mar = fci_10q[fci_10q.index.month == 3]
                fci_not_accum_10q = pd.concat([fci_not_accum, fci_10q_mar], sort=False) 
                # Ordenando a serie para que o index fique igual aos outros indicadores
                fci_not_accum_10q = fci_not_accum_10q.sort_index(ascending=False)

            # Condição para as empresas que fazem o lançamento nos meses 04, 07 e 10
            if first_quarter == '04': 
                if month == 7:
                    # Calculando o FCI não acumulado do mês 07, -> "mês 07" - "mês 04" 
                    fci_07 = item - (fci_10q[f'4-{year}'].values[0])
                    lst_fci_not_accum[idx] = fci_07

                if month == 10:
                    # Calculando o FCI não acumulado do mês 10 -> "mês 10" - ("mês 07" - "mês 04") - "mês 04"        
                    fci_10 = item - (fci_10q[f'7-{year}'].values[0] - fci_10q[f'4-{year}'].values[0]) - fci_10q[f'4-{year}'].values[0]
                    lst_fci_not_accum[idx] = fci_10

                # Contem apenas os FCIs não acumulados dos meses 07 e 10
                fci_not_accum = pd.Series(lst_fci_not_accum, dtype=float)
                # Adicionando o FCI do mês 04 nesta serie de FCIs não acumulados 
                fci_10q_apr = fci_10q[fci_10q.index.month == 4]
                fci_not_accum_10q = pd.concat([fci_not_accum, fci_10q_apr], sort=False)
                # Ordenando a serie para que o index fique igual aos outros indicadores
                fci_not_accum_10q = fci_not_accum_10q.sort_index(ascending=False)

            # Condição para as empresas que fazem o lançamento nos meses 09, 12 e 03
            if first_quarter == '09':
                if month == 12:
                    # Calculando o FCI não acumulado do mês 12 -> "mês 12" - "mês 09"
                    fci_12 = item - (fci_10q[f'9-{year}'].values[0])
                    lst_fci_not_accum[idx] = fci_12

                if month == 3:
                    # Calculando o ano anterior, porque o mês 3 já está no próximo ano
                    year_before = year - 1 

                    # Calculando o FCI não acumulado do mês 03 -> "mês 03 do próximo ano" - ("mês 12" - "mês 09") - "mês 09"
                    fci_03 = item - (fci_10q[f'12-{year_before}'].values[0] - fci_10q[f'9-{year_before}'].values[0]) - fci_10q[f'9-{year_before}'].values[0]
                    lst_fci_not_accum[idx] = fci_03

                # Contem apenas os FCI não acumulados dos meses 12 e 3
                fci_not_accum = pd.Series(lst_fci_not_accum, dtype=float)
                # Adicionando o FCI do mês 09 nesta serie de FCI não acumulados
                fci_10q_sep = fci_10q[fci_10q.index.month == 9]
                fci_not_accum_10q = pd.concat([fci_not_accum, fci_10q_sep], sort=False)
                # Ordenando a serie para que o index fique igual aos outros indicadores
                fci_not_accum_10q = fci_not_accum_10q.sort_index(ascending=False)

            # Condição para as empresas que fazem o lançamento nos meses 12, 03 e 06
            if first_quarter == '12': 
                if month == 3:
                    # Calculando o ano anterior, porque o 1Q é o mês 12
                    year_before = year - 1
                    # Calculando o FCI não acumulado do mês 03 -> "mês 03" - "mês 12 do ano anterior"
                    fci_03 = item - (fci_10q[f'12-{year_before}'].values[0])
                    lst_fci_not_accum[idx] = fci_03

                if month == 6:
                    # Calculando o ano anterior, porque o 1Q é o mês 12
                    year_before = year - 1
                    # Calculando o FCI não acumulado do mês 06 -> "mês 06" - ("mês 03" - "mês 12 do ano anterior") - "mês 12 do ano anterior"
                    fci_06 = item - (fci_10q[f'3-{year}'].values[0] - fci_10q[f'12-{year_before}'].values[0]) - fci_10q[f'12-{year_before}'].values[0]
                    lst_fci_not_accum[idx] = fci_06

                # Contem apenas os FCIs não acumulados dos meses 03 e 06
                fci_not_accum = pd.Series(lst_fci_not_accum, dtype=float)
                # Adicionando o FCI do mês 12 nesta serie de FCIs não acumulados 
                fci_10q_dec = fci_10q[fci_10q.index.month == 12]
                fci_not_accum_10q = pd.concat([fci_not_accum, fci_10q_dec], sort=False)
                # Ordenando a serie para que o index fique igual aos outros indicadores
                fci_not_accum_10q = fci_not_accum_10q.sort_index(ascending=False)
                
        return fci_not_accum_10q


def indicador_fcf(ten_k: bool, df_cf: pd.Series, first_quarter: str) -> pd.Series:
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    df_cf: pd.DataFrame
        DataFrame do 'cash flow statement' (fluxo de caixa).
    first_quarter: str
        Indicar qual é o número do 1Q da empresa.

    Returns
    -------
    fcf: pd.Series
        Série pandas do indicador fcf.
    fcf_10q: pd.Series
        Série pandas do indicador fcf trimestral.
    """
    if ten_k:
        # Calculando o FCF
        fcf = df_cf.loc['Net Cash Provided by (Used in) Financing Activities']
        return fcf
    else:
        # Calculando o FCF e retirando os dados de 2020
        fcf_10q = df_cf.loc['Net Cash Provided by (Used in) Financing Activities'][0:len(df_cf.columns)-3]

        # Calculando o FCF não acumulado do 2Q e 3Q
        lst_fcf_not_accum = {}

        for idx, item in fcf_10q.items():
            month, year = idx.month, idx.year

            # Condição para as empresas que fazem o lançamento nos meses 02, 05 e 08
            if first_quarter == '02': 
                if month == 5:
                    # Calculando o FCF não acumulado do mês 05, -> "mês 05" - "mês 02" 
                    fcf_05 = item - (fcf_10q[f'2-{year}'].values[0])
                    lst_fcf_not_accum[idx] = fcf_05

                if month == 8:
                    # Calculando o FCF não acumulado do mês 08 -> "mês 08" - ("mês 05" - "mês 02") - "mês 02"        
                    fcf_08 = item - (fcf_10q[f'5-{year}'].values[0] - fcf_10q[f'2-{year}'].values[0]) - fcf_10q[f'2-{year}'].values[0]
                    lst_fcf_not_accum[idx] = fcf_08

                # Contem apenas os FCFs não acumulados dos meses 05 e 08
                fcf_not_accum = pd.Series(lst_fcf_not_accum, dtype=float)
                # Adicionando o FCF do mês 02 nesta serie de FCFs não acumulados 
                fcf_10q_feb = fcf_10q[fcf_10q.index.month == 2]
                fcf_not_accum_10q = pd.concat([fcf_not_accum, fcf_10q_feb], sort=False)
                 # Ordenando a serie para que o index fique igual aos outros indicadores
                fcf_not_accum_10q = fcf_not_accum_10q.sort_index(ascending=False)

            # Condição para as empresas que fazem o lançamento nos meses 03, 06 e 09
            if first_quarter == '03': 
                if month == 6:
                    # Calculando o FCF não acumulado do mês 06 -> "mês 06" - "mês 03"
                    fcf_06 = item - (fcf_10q[f'3-{year}'].values[0])
                    lst_fcf_not_accum[idx] = fcf_06

                if month == 9:
                    # Calculando o FCF não acumulado do mês 09 -> "mês 09" - ("mês 06" - "mês 03") - "mês 03"
                    fcf_09 = item - (fcf_10q[f'6-{year}'].values[0] - fcf_10q[f'3-{year}'].values[0]) - fcf_10q[f'3-{year}'].values[0]
                    lst_fcf_not_accum[idx] = fcf_09
                    
                # Contem apenas os FCFs não acumulados dos meses 06 e 09
                fcf_not_accum = pd.Series(lst_fcf_not_accum, dtype=float)
                # Adicionando o FCI do mês 03 nesta serie de FCIs não acumulados 
                fcf_10q_mar = fcf_10q[fcf_10q.index.month == 3]
                fcf_not_accum_10q = pd.concat([fcf_not_accum, fcf_10q_mar], sort=False) 
                # Ordenando a serie para que o index fique igual aos outros indicadores
                fcf_not_accum_10q = fcf_not_accum_10q.sort_index(ascending=False)

            # Condição para as empresas que fazem o lançamento nos meses 04, 07 e 10
            if first_quarter == '04': 
                if month == 7:
                    # Calculando o FCF não acumulado do mês 07, -> "mês 07" - "mês 04" 
                    fcf_07 = item - (fcf_10q[f'4-{year}'].values[0])
                    lst_fcf_not_accum[idx] = fcf_07

                if month == 10:
                    # Calculando o FCF não acumulado do mês 10 -> "mês 10" - ("mês 07" - "mês 04") - "mês 04"        
                    fcf_10 = item - (fcf_10q[f'7-{year}'].values[0] - fcf_10q[f'4-{year}'].values[0]) - fcf_10q[f'4-{year}'].values[0]
                    lst_fcf_not_accum[idx] = fcf_10

                # Contem apenas os FCFs não acumulados dos meses 07 e 10
                fcf_not_accum = pd.Series(lst_fcf_not_accum, dtype=float)
                # Adicionando o FCF do mês 04 nesta serie de FCFs não acumulados 
                fcf_10q_apr = fcf_10q[fcf_10q.index.month == 4]
                fcf_not_accum_10q = pd.concat([fcf_not_accum, fcf_10q_apr], sort=False)
                # Ordenando a serie para que o index fique igual aos outros indicadores
                fcf_not_accum_10q = fcf_not_accum_10q.sort_index(ascending=False)


            # Condição para as empresas que fazem o lançamento nos meses 09, 12 e 03
            if first_quarter == '09':
                if month == 12:
                    # Calculando o FCF não acumulado do mês 12 -> "mês 12" - "mês 09"
                    fcf_12 = item - (fcf_10q[f'9-{year}'].values[0])
                    lst_fcf_not_accum[idx] = fcf_12

                if month == 3:
                    # Calculando o ano anterior, porque o mês 3 já está no próximo ano
                    year_before = year - 1 

                    # Calculando o FCF não acumulado do mês 03 -> "mês 03 do próximo ano" - ("mês 12" - "mês 09") - "mês 09"
                    fcf_03 = item - (fcf_10q[f'12-{year_before}'].values[0] - fcf_10q[f'9-{year_before}'].values[0]) - fcf_10q[f'9-{year_before}'].values[0]
                    lst_fcf_not_accum[idx] = fcf_03

                # Contem apenas os FCF não acumulados dos meses 12 e 3
                fcf_not_accum = pd.Series(lst_fcf_not_accum, dtype=float)
                # Adicionando o FCF do mês 09 nesta serie de FCF não acumulados
                fcf_10q_sep = fcf_10q[fcf_10q.index.month == 9]
                fcf_not_accum_10q = pd.concat([fcf_not_accum, fcf_10q_sep], sort=False)
                # Ordenando a serie para que o index fique igual aos outros indicadores
                fcf_not_accum_10q = fcf_not_accum_10q.sort_index(ascending=False)

            # Condição para as empresas que fazem o lançamento nos meses 12, 03 e 06
            if first_quarter == '12': 
                if month == 3:
                    # Calculando o ano anterior, porque o 1Q é o mês 12
                    year_before = year - 1
                    # Calculando o FCF não acumulado do mês 03 -> "mês 03" - "mês 12 do ano anterior"
                    fcf_03 = item - (fcf_10q[f'12-{year_before}'].values[0])
                    lst_fcf_not_accum[idx] = fcf_03

                if month == 6:
                    # Calculando o ano anterior, porque o 1Q é o mês 12
                    year_before = year - 1
                    # Calculando o FCF não acumulado do mês 06 -> "mês 06" - ("mês 03" - "mês 12 do ano anterior") - "mês 12 do ano anterior"
                    fcf_06 = item - (fcf_10q[f'3-{year}'].values[0] - fcf_10q[f'12-{year_before}'].values[0]) - fcf_10q[f'12-{year_before}'].values[0]
                    lst_fcf_not_accum[idx] = fcf_06

                # Contem apenas os FCFs não acumulados dos meses 03 e 06
                fcf_not_accum = pd.Series(lst_fcf_not_accum, dtype=float)
                # Adicionando o FCF do mês 12 nesta serie de FCFs não acumulados 
                fcf_10q_dec = fcf_10q[fcf_10q.index.month == 12]
                fcf_not_accum_10q = pd.concat([fcf_not_accum, fcf_10q_dec], sort=False)
                # Ordenando a serie para que o index fique igual aos outros indicadores
                fcf_not_accum_10q = fcf_not_accum_10q.sort_index(ascending=False)
                
        return fcf_not_accum_10q


def indicador_capex(ten_k: bool, df_cf: pd.Series, first_quarter: str, lst_itens: list, idx: list) -> pd.Series|tuple[pd.Series, pd.Series, pd.Series]:
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    df_cf: pd.DataFrame
        DataFrame do 'cash flow statement' (fluxo de caixa).
    first_quarter: str
        Indicar qual é o número do 1Q da empresa.
    lst_itens: list
        Lista dos itens do 'cash flow statement' (fluxo de caixa) que formam o capex.
    idx: list
        Lista que contém as datas que serão o index do indicador.

    Returns
    -------
    capex: pd.Series
        Série pandas do indicador capex.
    capex_not_accum_10q: pd.Series
        Série pandas do indicador capex trimestral não acumulado.
    capex_not_accum_10q_sliced: pd.Series
        Série pandas do indicador capex trimestral não acumulado cortado.
    capex_accum_10q: pd.Series
        Série pandas do indicador capex trimestral acumulado.
    """
    if ten_k:
        # Calculando o capex
        capex = abs(df_cf.loc[lst_itens].sum())
        return capex
    else:
        # Calculando o capex 
        capex_10q = abs(df_cf.loc[lst_itens].sum())
        
        # Calculando o capex não acumulado do 2Q e 3Q
        lst_capex_not_accum = {}

        for current_idx, item in capex_10q.items():
            month, year = current_idx.month, current_idx.year

            # Condição para as empresas que fazem o lançamento nos meses 02, 05 e 08
            if first_quarter == '02': 
                if month == 5:
                    # Calculando o capex não acumulado do mês 05, -> "mês 05" - "mês 02" 
                    capex_05 = item - (capex_10q[f'2-{year}'].values[0])
                    lst_capex_not_accum[current_idx] = capex_05

                if month == 8:
                    # Calculando o capex não acumulado do mês 08 -> "mês 08" - ("mês 05" - "mês 02") - "mês 02"        
                    capex_08 = item - (capex_10q[f'5-{year}'].values[0] - capex_10q[f'2-{year}'].values[0]) - capex_10q[f'2-{year}'].values[0]
                    lst_capex_not_accum[current_idx] = capex_08

                # Contem apenas os capex não acumulados dos meses 05 e 08
                capex_not_accum = pd.Series(lst_capex_not_accum, dtype=float)
                # Adicionando o capex do mês 02 nesta serie de capex não acumulados 
                capex_10q_feb = capex_10q[capex_10q.index.month == 2]
                capex_not_accum_10q = pd.concat([capex_not_accum, capex_10q_feb], sort=False)
                capex_not_accum_10q = capex_not_accum_10q.sort_index(ascending=False)

                # Condição para as empresas que fazem o lançamento nos meses 03, 06 e 09
            if first_quarter == '03': 
                if month == 6:
                    # Calculando o capex não acumulado do mês 06 -> "mês 06" - "mês 03"
                    capex_06 = item - (capex_10q[f'3-{year}'].values[0])
                    lst_capex_not_accum[current_idx] = capex_06

                if month == 9:
                    # Calculando o capex não acumulado do mês 09 -> "mês 09" - ("mês 06" - "mês 03") - "mês 03"
                    capex_09 = item - (capex_10q[f'6-{year}'].values[0] - capex_10q[f'3-{year}'].values[0]) - capex_10q[f'3-{year}'].values[0]
                    lst_capex_not_accum[current_idx] = capex_09
                        
                # Contem apenas os capex não acumulados dos meses 06 e 09
                capex_not_accum = pd.Series(lst_capex_not_accum, dtype=float)
                # Adicionando o capex do mês 03 nesta serie de capex não acumulados 
                capex_10q_mar = capex_10q[capex_10q.index.month == 3]
                capex_not_accum_10q = pd.concat([capex_not_accum, capex_10q_mar], sort=False) 
                # Ordenando a serie para que o index fique igual aos outros indicadores
                capex_not_accum_10q = capex_not_accum_10q.sort_index(ascending=False)

            # Condição para as empresas que fazem o lançamento nos meses 04, 07 e 10
            if first_quarter == '04': 
                if month == 7:
                    # Calculando o capex não acumulado do mês 07, -> "mês 07" - "mês 04" 
                    capex_07 = item - (capex_10q[f'4-{year}'].values[0])
                    lst_capex_not_accum[current_idx] = capex_07

                if month == 10:
                    # Calculando o capex não acumulado do mês 10 -> "mês 10" - ("mês 07" - "mês 04") - "mês 04"        
                    capex_10 = item - (capex_10q[f'7-{year}'].values[0] - capex_10q[f'4-{year}'].values[0]) - capex_10q[f'4-{year}'].values[0]
                    lst_capex_not_accum[current_idx] = capex_10

                # Contem apenas os capex não acumulados dos meses 07 e 10
                capex_not_accum = pd.Series(lst_capex_not_accum, dtype=float)
                # Adicionando o capex do mês 04 nesta serie de capex não acumulados 
                capex_10q_apr = capex_10q[capex_10q.index.month == 4]
                capex_not_accum_10q = pd.concat([capex_not_accum, capex_10q_apr], sort=False)
                capex_not_accum_10q = capex_not_accum_10q.sort_index(ascending=False)

            # Condição para as empresas que fazem o lançamento nos meses 09, 12 e 03
            if first_quarter == '09':
                if month == 12:
                    # Calculando o capex não acumulado do mês 12 -> "mês 12" - "mês 09"
                    capex_12 = item - (capex_10q[f'9-{year}'].values[0])
                    lst_capex_not_accum[current_idx] = capex_12

                if month == 3:
                    # Calculando o ano anterior, porque o mês 3 já está no próximo ano
                    year_before = year - 1 

                    # Calculando o capex não acumulado do mês 03 -> "mês 03 do próximo ano" - ("mês 12" - "mês 09") - "mês 09"
                    capex_03 = item - (capex_10q[f'12-{year_before}'].values[0] - capex_10q[f'9-{year_before}'].values[0]) - capex_10q[f'9-{year_before}'].values[0]
                    lst_capex_not_accum[current_idx] = capex_03

                # Contem apenas os capex não acumulados dos meses 12 e 3
                capex_not_accum = pd.Series(lst_capex_not_accum, dtype=float)
                # Adicionando o capex do mês 09 nesta serie de capex não acumulados
                capex_10q_sep = capex_10q[capex_10q.index.month == 9]
                capex_not_accum_10q = pd.concat([capex_not_accum, capex_10q_sep], sort=False)
                # Ordenando a serie para que o index fique igual aos outros indicadores
                capex_not_accum_10q = capex_not_accum_10q.sort_index(ascending=False)

            # Condição para as empresas que fazem o lançamento nos meses 12, 03 e 06
            if first_quarter == '12':
                if month == 3:
                    # Calculando o ano anterior, porque o 1Q é o mês 12
                    year_before = year - 1
                    # Calculando o capex não acumulado do mês 03 -> "mês 03" - "mês 12 do ano anterior"
                    capex_03 = item - (capex_10q[f'12-{year_before}'].values[0])
                    lst_capex_not_accum[current_idx] = capex_03
        
                if month == 6:
                    # Calculando o ano anterior, porque o 1Q é o mês 12
                    year_before = year - 1
                    # Calculando o capex não acumulad0 do mês 06 -> "mês 06" - ("mês 03" - "mês 12 do ano anterior") - "mês 12 do ano anterior"
                    capex_06 = item - (capex_10q[f'3-{year}'].values[0] - capex_10q[f'12-{year_before}'].values[0]) - capex_10q[f'12-{year_before}'].values[0]
                    lst_capex_not_accum[current_idx] = capex_06

                # Contem apenas o capex não acumulados dos meses 03 e 06
                capex_not_accum = pd.Series(lst_capex_not_accum, dtype=float)
                # Adicionando o capex do mês 12 nesta serie de capex não acumulados 
                capex_10q_dec = capex_10q[capex_10q.index.month == 12]
                capex_not_accum_10q = pd.concat([capex_not_accum, capex_10q_dec], sort=False)
                capex_not_accum_10q = capex_not_accum_10q.sort_index(ascending=False)

        # Retirando os dados de 2020
        capex_not_accum_10q_sliced = capex_not_accum_10q[0:len(capex_not_accum_10q)-3]

        # Calculando o capex acumulado
        lst_capex_accum_10q = []
        for i in range(len(capex_not_accum_10q) - 3):
            capex_accum = capex_not_accum_10q[i:i+4].sum()  
            lst_capex_accum_10q.append(capex_accum)
        # Criando uma serie do capex acumulado
        capex_accum_10q = pd.Series(lst_capex_accum_10q, index=idx[0:len(idx)-3])

        return capex_not_accum_10q, capex_not_accum_10q_sliced, capex_accum_10q


def indicador_free_cash_flow(fco: pd.Series, capex: pd.Series) -> pd.Series:
    """
    Parameters
    ----------
    fco: pd.Series
        Série pandas do indicador fco.
    capex: pd.Series
        Série pandas do indicador capex.

    Returns
    -------
    free_cash_flow: pd.Series
        Série pandas do indicador free cash flow.
    """
    # Calculando o Free Cash Flow
    free_cash_flow = fco - capex
    return free_cash_flow


def indicador_rd(ten_k: bool, df_is: pd.Series, idx: list) -> pd.Series:
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    df_is: pd.DataFrame
        DataFrame do 'income statement' (DRE).
    idx: list
        Lista que contém as datas que serão o index do indicador.

    Returns
    -------
    rd_expense: pd.Series
        Série pandas do indicador R&D.
    rd_expense_10q: pd.Series
        Série pandas do indicador capex trimestral.
    """
    if ten_k:
        # Calculando o R&D
        rd_expense = abs(df_is.loc['Research and Development'])
        return rd_expense
    
    else:
        # Calculando o R&D e retirando os dados de 2020
        rd_expense_10q = abs(df_is.loc['Research and Development'])

        # Retirando os dados de 2020
        rd_expense_sliced = rd_expense_10q[0:len(rd_expense_10q)-3]

        # Calculando o R&D acumulado
        lst_rd_expense_acum_10q = []
        for i in range(len(rd_expense_10q) - 3):
            rd_expense_accum = rd_expense_10q[i:i+4].sum()  
            lst_rd_expense_acum_10q.append(rd_expense_accum)
        # Criando uma serie do R&D acumulado
        rd_expense_accum_10q = pd.Series(lst_rd_expense_acum_10q, index=idx[0:len(idx)-3])

        return rd_expense_10q, rd_expense_sliced, rd_expense_accum_10q


def indicador_net_capex(capex: pd.Series, depreciation: pd.Series) -> pd.Series:
    """
    Parameters
    ----------
    capex: pd.Series
        Série pandas do indicador capex.
    depreciation: pd.Series
        Série pandas do indicador depreciação.

    Returns
    -------
    net_capex: pd.Series
        Série pandas do indicador net capex.
    """
    # Calculando o Net Capex
    net_capex = capex - depreciation
    return net_capex


def indicador_adj_net_capex(net_capex: pd.Series, rd_expense: pd.Series) -> pd.Series:
    """
    Parameters
    ----------
    net_capex: pd.Series
        Série pandas do indicador net capex.
    rd_expense: pd.Series
        Série pandas do indicador R&D.

    Returns
    -------
    adj_net_capex: pd.Series
        Série pandas do indicador net capex ajustado.
    """
    # Calculando o Net Capex
    adj_net_capex = net_capex + rd_expense
    return adj_net_capex


def indicador_working_capital(ten_k: bool, df_bs: pd.Series) -> pd.Series:
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    df_bs: pd.DataFrame
        DataFrame do 'balance sheet' (balanço patrimonial).

    Returns
    -------
    working_capital: pd.Series
        Série pandas do indicador working capital (capital de giro).
    working_capital_10q: pd.Series
        Série pandas do indicador working capital (capital de giro) trimestral.
    """
    if ten_k:
        # Calculando o R&D
        working_capital = df_bs.loc['Assets, Current'] - df_bs.loc['Liabilities, Current']
        return working_capital
    else:
        # Calculando o R&D e retirando os dados de 2020
        working_capital_10q = df_bs.loc['Assets, Current'][0:len(df_bs.columns)-3] - df_bs.loc['Liabilities, Current'][0:len(df_bs.columns)-3]
        return working_capital_10q


def indicador_change_non_cash_wc(ten_k: bool, df_cf: pd.Series, lst_itens: list, first_quarter: str) -> pd.Series|tuple[pd.Series, pd.Series]:
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    df_cf: pd.DataFrame
        DataFrame do 'cash flow statement' (fluxo de caixa).
    lst_itens: list
        Lista dos itens do 'cash flow statement' (fluxo de caixa) que formam o change non cash flow.
    first_quarter: str
        Indicar qual é o número do 1Q da empresa.

    Returns
    -------
    change_non_cash_wc: pd.Series
        Série pandas do indicador change non cash working capital.
    change_non_cash_wc_not_accum_10q: pd.Series
        Série pandas do indicador change non cash working capital trimestral não acumulado.
    change_non_cash_wc_not_accum_10q_sliced: pd.Series
        Série pandas do indicador change non cash working capital trimestral acumulado.
    """
    if ten_k:
        # Calculando o change non-cash working capital
        change_non_cash_wc = df_cf.loc[lst_itens].sum() * -1
        return change_non_cash_wc
    
    else:
        # Calculando o change non-cash working capital 
        change_non_cash_wc_10q = df_cf.loc[lst_itens].sum() * -1
        # Calculando o change non-cash working capital não acumulado do 2Q e 3Q
        lst_change_non_cash_wc_not_accum = {}

        for idx, item in change_non_cash_wc_10q.items():
            month, year = idx.month, idx.year

            # Condição para as empresas que fazem o lançamento nos meses 02, 05 e 08
            if first_quarter == '02': 
                if month == 5:
                    # Calculando o change non-cash working capital não acumulado do mês 05, -> "mês 05" - "mês 02" 
                    change_non_cash_wc_05 = item - (change_non_cash_wc_10q[f'2-{year}'].values[0])
                    lst_change_non_cash_wc_not_accum[idx] = change_non_cash_wc_05

                if month == 8:
                    # Calculando o change non-cash working capital não acumulado do mês 08 -> "mês 08" - ("mês 05" - "mês 02") - "mês 02"        
                    change_non_cash_wc_08 = item - (change_non_cash_wc_10q[f'5-{year}'].values[0] - change_non_cash_wc_10q[f'2-{year}'].values[0]) - change_non_cash_wc_10q[f'2-{year}'].values[0]
                    lst_change_non_cash_wc_not_accum[idx] = change_non_cash_wc_08

                # Contem apenas os change non-cash working capital não acumulados dos meses 05 e 08
                change_non_cash_wc_not_accum = pd.Series(lst_change_non_cash_wc_not_accum, dtype=float)
                # Adicionando o change non-cash working capital do mês 02 nesta serie de change non-cash working capital não acumulados 
                change_non_cash_wc_10q_feb = change_non_cash_wc_10q[change_non_cash_wc_10q.index.month == 2]
                change_non_cash_wc_not_accum_10q = pd.concat([change_non_cash_wc_not_accum, change_non_cash_wc_10q_feb], sort=False)
                change_non_cash_wc_not_accum_10q = change_non_cash_wc_not_accum_10q.sort_index(ascending=False)

                # Condição para as empresas que fazem o lançamento nos meses 03, 06 e 09
            if first_quarter == '03': 
                if month == 6:
                    # Calculando o change non-cash working capital não acumulado do mês 06 -> "mês 06" - "mês 03"
                    change_non_cash_wc_06 = item - (change_non_cash_wc_10q[f'3-{year}'].values[0])
                    lst_change_non_cash_wc_not_accum[idx] = change_non_cash_wc_06

                if month == 9:
                    # Calculando o change non-cash working capital não acumulado do mês 09 -> "mês 09" - ("mês 06" - "mês 03") - "mês 03"
                    change_non_cash_wc_09 = item - (change_non_cash_wc_10q[f'6-{year}'].values[0] - change_non_cash_wc_10q[f'3-{year}'].values[0]) - change_non_cash_wc_10q[f'3-{year}'].values[0]
                    lst_change_non_cash_wc_not_accum[idx] = change_non_cash_wc_09
                        
                # Contem apenas os change non-cash working capital não acumulados dos meses 06 e 09
                change_non_cash_wc_not_accum = pd.Series(lst_change_non_cash_wc_not_accum, dtype=float)
                # Adicionando o change non-cash working capital do mês 03 nesta serie de change non-cash working capital não acumulados 
                change_non_cash_wc_10q_mar = change_non_cash_wc_10q[change_non_cash_wc_10q.index.month == 3]
                change_non_cash_wc_not_accum_10q = pd.concat([change_non_cash_wc_not_accum, change_non_cash_wc_10q_mar], sort=False) 
                # Ordenando a serie para que o index fique igual aos outros indicadores
                change_non_cash_wc_not_accum_10q = change_non_cash_wc_not_accum_10q.sort_index(ascending=False)

            # Condição para as empresas que fazem o lançamento nos meses 04, 07 e 10
            if first_quarter == '04': 
                if month == 7:
                    # Calculando o change non-cash working capital não acumulado do mês 07, -> "mês 07" - "mês 04" 
                    change_non_cash_wc_07 = item - (change_non_cash_wc_10q[f'4-{year}'].values[0])
                    lst_change_non_cash_wc_not_accum[idx] = change_non_cash_wc_07

                if month == 10:
                    # Calculando o change non-cash working capital não acumulado do mês 10 -> "mês 10" - ("mês 07" - "mês 04") - "mês 04"        
                    change_non_cash_wc_10 = item - (change_non_cash_wc_10q[f'7-{year}'].values[0] - change_non_cash_wc_10q[f'4-{year}'].values[0]) - change_non_cash_wc_10q[f'4-{year}'].values[0]
                    lst_change_non_cash_wc_not_accum[idx] = change_non_cash_wc_10

                # Contem apenas os change non-cash working capital não acumulados dos meses 07 e 10
                change_non_cash_wc_not_accum = pd.Series(lst_change_non_cash_wc_not_accum, dtype=float)
                # Adicionando o change non-cash working capital do mês 04 nesta serie de change non-cash working capital não acumulados 
                change_non_cash_wc_10q_apr = change_non_cash_wc_10q[change_non_cash_wc_10q.index.month == 4]
                change_non_cash_wc_not_accum_10q = pd.concat([change_non_cash_wc_not_accum, change_non_cash_wc_10q_apr], sort=False)
                change_non_cash_wc_not_accum_10q = change_non_cash_wc_not_accum_10q.sort_index(ascending=False)

            # Condição para as empresas que fazem o lançamento nos meses 09, 12 e 03
            if first_quarter == '09':
                if month == 12:
                    # Calculando o change non-cash working capital não acumulado do mês 12 -> "mês 12" - "mês 09"
                    change_non_cash_wc_12 = item - (change_non_cash_wc_10q[f'9-{year}'].values[0])
                    lst_change_non_cash_wc_not_accum[idx] = change_non_cash_wc_12

                if month == 3:
                    # Calculando o ano anterior, porque o mês 3 já está no próximo ano
                    year_before = year - 1 

                    # Calculando o change non-cash working capital não acumulado do mês 03 -> "mês 03 do próximo ano" - ("mês 12" - "mês 09") - "mês 09"
                    change_non_cash_wc_03 = item - (change_non_cash_wc_10q[f'12-{year_before}'].values[0] - change_non_cash_wc_10q[f'9-{year_before}'].values[0]) - change_non_cash_wc_10q[f'9-{year_before}'].values[0]
                    lst_change_non_cash_wc_not_accum[idx] = change_non_cash_wc_03

                # Contem apenas os change non-cash working capital não acumulados dos meses 12 e 3
                change_non_cash_wc_not_accum = pd.Series(lst_change_non_cash_wc_not_accum, dtype=float)
                # Adicionando os change non-cash working capital do mês 09 nesta serie de change non-cash working capital não acumulados
                change_non_cash_wc_10q_sep = change_non_cash_wc_10q[change_non_cash_wc_10q.index.month == 9]
                change_non_cash_wc_not_accum_10q = pd.concat([change_non_cash_wc_not_accum, change_non_cash_wc_10q_sep], sort=False)
                # Ordenando a serie para que o index fique igual aos outros indicadores
                change_non_cash_wc_not_accum_10q = change_non_cash_wc_not_accum_10q.sort_index(ascending=False)

            # Condição para as empresas que fazem o lançamento nos meses 12, 03 e 06
            if first_quarter == '12':
                if month == 3:
                    # Calculando o ano anterior, porque o 1Q é o mês 12
                    year_before = year - 1
                    # Calculando o change non-cash working capital não acumulado do mês 03 -> "mês 03" - "mês 12 do ano anterior"
                    change_non_cash_wc_03 = item - (change_non_cash_wc_10q[f'12-{year_before}'].values[0])
                    lst_change_non_cash_wc_not_accum[idx] = change_non_cash_wc_03
        
                if month == 6:
                    # Calculando o ano anterior, porque o 1Q é o mês 12
                    year_before = year - 1
                    # Calculando o change non-cash working capital não acumulad0 do mês 06 -> "mês 06" - ("mês 03" - "mês 12 do ano anterior") - "mês 12 do ano anterior"
                    capex_06 = item - (change_non_cash_wc_10q[f'3-{year}'].values[0] - change_non_cash_wc_10q[f'12-{year_before}'].values[0]) - change_non_cash_wc_10q[f'12-{year_before}'].values[0]
                    lst_change_non_cash_wc_not_accum[idx] = capex_06

                # Contem apenas o change non-cash working capital não acumulados dos meses 03 e 06
                change_non_cash_wc_not_accum = pd.Series(lst_change_non_cash_wc_not_accum, dtype=float)
                # Adicionando o change non-cash working capital do mês 12 nesta serie de change non-cash working capital não acumulados 
                change_non_cash_wc_10q_dec = change_non_cash_wc_10q[change_non_cash_wc_10q.index.month == 12]
                change_non_cash_wc_not_accum_10q = pd.concat([change_non_cash_wc_not_accum, change_non_cash_wc_10q_dec], sort=False)
                change_non_cash_wc_not_accum_10q = change_non_cash_wc_not_accum_10q.sort_index(ascending=False)

        # Retirando os dados de 2020
        change_non_cash_wc_not_accum_10q_sliced = change_non_cash_wc_not_accum_10q[0:len(change_non_cash_wc_not_accum_10q)-3]
        return change_non_cash_wc_not_accum_10q, change_non_cash_wc_not_accum_10q_sliced


def indicador_reinvestment_rate(ten_k: bool, net_capex: pd.Series, change_non_cash_wc: pd.Series, df_is: pd.DataFrame, idx:list) -> pd.Series:
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    net_capex: pd.Series
        Série pandas do indicador net capex.
    change_non_cash_wc: pd.Series
        Série pandas do indicador change_non_cash_wc.
    df_is: pd.DataFrame
        DataFrame do 'income statement' (DRE).
    idx: list
        Lista que contém as datas que serão o index do indicador.

    Returns
    -------
    reinvestment_rate: pd.Series
        Série pandas do indicador reinvestment rate.
    reinvestment_rate_10q: pd.Series
        Série pandas do indicador reinvestment rate trimestral.
    """
    if ten_k:
        # Calculando o reinvestment rate
        reinvestment_rate = round((net_capex + change_non_cash_wc) / (df_is.loc['Operating Income'] - df_is.loc['Provision for Income Taxes']), 4) * 100
        return reinvestment_rate
    else:
        # Calculando o EBIT acumulado
        ebit_10q = df_is.loc['Operating Income']
        lst_ebit_acum_10q = []
        for i in range(len(ebit_10q) - 3):
            ebit_acum = ebit_10q[i:i+4].sum()  
            lst_ebit_acum_10q.append(round(ebit_acum,2))
        # Criando uma serie do EBIT acumulado
        ebit_10q_acum = pd.Series(lst_ebit_acum_10q, index=idx[0:len(idx)-3])

        # Calculando o IR acumulado
        ir_expense_10q = df_is.loc['Provision for Income Taxes']
        lst_ir_expense_acum_10q = []
        for i in range(len(ir_expense_10q) - 3):
            ir_expense_acum = ir_expense_10q[i:i+4].sum()  
            lst_ir_expense_acum_10q.append(round(ir_expense_acum,2))
        # Criando uma serie do IR acumulado
        ir_expense_10q_acum = pd.Series(lst_ir_expense_acum_10q, index=idx[0:len(idx)-3])
        
        # Calculando o reinvestment rate
        reinvestment_rate_10q = round((net_capex + change_non_cash_wc[0:-3]) / (ebit_10q_acum - ir_expense_10q_acum), 4) * 100
        return reinvestment_rate_10q


def indicador_new_borrowing(
    ten_k: bool, 
    lst_itens_new_borrowing: list, 
    df_cf: pd.DataFrame, 
    first_quarter: str, 
    idx: list
) -> pd.Series|tuple[pd.Series, pd.Series, pd.Series]:
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    lst_itens_new_borrowing: list
        Lista dos itens do 'cash flow statement' (fluxo de caixa) que formam o new borrowing.
    df_cf: pd.DataFrame
        DataFrame do 'cash flow statement' (fluxo de caixa).
    first_quarter: str
        Indicar qual é o número do 1Q da empresa.
    idx: list
        Lista que contém as datas que serão o index do indicador.

    Returns
    -------
    new borrowing: pd.Series
        Série pandas do indicador new borrowing.
    new_borrowing_not_accum_10q: pd.Series
        Série pandas do indicador new borrowing trimestral não acumulado.
    new_borrowing_not_accum_10q_sliced: pd.Series
        Série pandas do indicador new borrowing trimestral não acumulado cortado.
    new_borrowing_accum_10q: pd.Series
        Série pandas do indicador new borrowing trimestral acumulado.
    """
    if ten_k:
        # Calculando o new borring
        new_borrowing = abs(df_cf.loc[lst_itens_new_borrowing]).sum()
        return new_borrowing
    
    else:
        # Calculando o new borring
        new_borrowing_10q = abs(df_cf.loc[lst_itens_new_borrowing]).sum()
        # Calculando o new borrowing não acumulado do 2Q e 3Q
        lst_new_borrowing_not_accum = {}

        for current_idx, item in new_borrowing_10q.items():
            month, year = current_idx.month, current_idx.year

            # Condição para as empresas que fazem o lançamento nos meses 02, 05 e 08
            if first_quarter == '02': 
                if month == 5:
                    # Calculando o new borrowing não acumulado do mês 05, -> "mês 05" - "mês 02" 
                    new_borrowing_05 = item - (new_borrowing_10q[f'2-{year}'].values[0])
                    lst_new_borrowing_not_accum[current_idx] = new_borrowing_05

                if month == 8:
                    # Calculando o new borrowing não acumulado do mês 08 -> "mês 08" - ("mês 05" - "mês 02") - "mês 02"        
                    new_borrowing_08 = item - (new_borrowing_10q[f'5-{year}'].values[0] - new_borrowing_10q[f'2-{year}'].values[0]) - new_borrowing_10q[f'2-{year}'].values[0]
                    lst_new_borrowing_not_accum[current_idx] = new_borrowing_08

                # Contem apenas os new borrowing não acumulados dos meses 05 e 08
                new_borrowing_not_accum = pd.Series(lst_new_borrowing_not_accum, dtype=float)
                # Adicionando o new borrowing do mês 02 nesta serie de new borrowing não acumulados 
                new_borrowing_10q_feb = new_borrowing_10q[new_borrowing_10q.index.month == 2]
                new_borrowing_not_accum_10q = pd.concat([new_borrowing_not_accum, new_borrowing_10q_feb], sort=False)
                # Ordenando a serie para que o index fique igual aos outros indicadores
                new_borrowing_not_accum_10q = new_borrowing_not_accum_10q.sort_index(ascending=False)

                # Condição para as empresas que fazem o lançamento nos meses 03, 06 e 09
            if first_quarter == '03': 
                if month == 6:
                    # Calculando o new borrowing não acumulado do mês 06 -> "mês 06" - "mês 03"
                    new_borrowing_06 = item - (new_borrowing_10q[f'3-{year}'].values[0])
                    lst_new_borrowing_not_accum[current_idx] = new_borrowing_06

                if month == 9:
                    # Calculando o new borrowing não acumulado do mês 09 -> "mês 09" - ("mês 06" - "mês 03") - "mês 03"
                    new_borrowing_09 = item - (new_borrowing_10q[f'6-{year}'].values[0] - new_borrowing_10q[f'3-{year}'].values[0]) - new_borrowing_10q[f'3-{year}'].values[0]
                    lst_new_borrowing_not_accum[current_idx] = new_borrowing_09
                        
                # Contem apenas os new borrowing não acumulados dos meses 06 e 09
                new_borrowing_not_accum = pd.Series(lst_new_borrowing_not_accum, dtype=float)
                # Adicionando o new borrowing do mês 03 nesta serie de new borrowing não acumulados 
                new_borrowing_10q_mar = new_borrowing_10q[new_borrowing_10q.index.month == 3]
                new_borrowing_not_accum_10q = pd.concat([new_borrowing_not_accum, new_borrowing_10q_mar], sort=False) 
                # Ordenando a serie para que o index fique igual aos outros indicadores
                new_borrowing_not_accum_10q = new_borrowing_not_accum_10q.sort_index(ascending=False)

            # Condição para as empresas que fazem o lançamento nos meses 04, 07 e 10
            if first_quarter == '04': 
                if month == 7:
                    # Calculando o new borrowing não acumulado do mês 07, -> "mês 07" - "mês 04" 
                    new_borrowing_07 = item - (new_borrowing_10q[f'4-{year}'].values[0])
                    lst_new_borrowing_not_accum[current_idx] = new_borrowing_07

                if month == 10:
                    # Calculando o new borrowing não acumulado do mês 10 -> "mês 10" - ("mês 07" - "mês 04") - "mês 04"        
                    new_borrowing_10 = item - (new_borrowing_10q[f'7-{year}'].values[0] - new_borrowing_10q[f'4-{year}'].values[0]) - new_borrowing_10q[f'4-{year}'].values[0]
                    lst_new_borrowing_not_accum[current_idx] = new_borrowing_10

                # Contem apenas os new borrowing não acumulados dos meses 07 e 10
                new_borrowing_not_accum = pd.Series(lst_new_borrowing_not_accum, dtype=float)
                # Adicionando o new borrowing do mês 04 nesta serie de new borrowing não acumulados 
                new_borrowing_10q_apr = new_borrowing_10q[new_borrowing_10q.index.month == 4]
                new_borrowing_not_accum_10q = pd.concat([new_borrowing_not_accum, new_borrowing_10q_apr], sort=False)
                # Ordenando a serie para que o index fique igual aos outros indicadores
                new_borrowing_not_accum_10q = new_borrowing_not_accum_10q.sort_index(ascending=False)

            # Condição para as empresas que fazem o lançamento nos meses 09, 12 e 03
            if first_quarter == '09':
                if month == 12:
                    # Calculando o new borrowing não acumulado do mês 12 -> "mês 12" - "mês 09"
                    new_borrowing_12 = item - (new_borrowing_10q[f'9-{year}'].values[0])
                    lst_new_borrowing_not_accum[current_idx] = new_borrowing_12

                if month == 3:
                    # Calculando o ano anterior, porque o mês 3 já está no próximo ano
                    year_before = year - 1 

                    # Calculando o new borrowing não acumulado do mês 03 -> "mês 03 do próximo ano" - ("mês 12" - "mês 09") - "mês 09"
                    new_borrowing_03 = item - (new_borrowing_10q[f'12-{year_before}'].values[0] - new_borrowing_10q[f'9-{year_before}'].values[0]) - new_borrowing_10q[f'9-{year_before}'].values[0]
                    lst_new_borrowing_not_accum[current_idx] = new_borrowing_03

                # Contem apenas os new borrowing não acumulados dos meses 12 e 3
                new_borrowing_not_accum = pd.Series(lst_new_borrowing_not_accum, dtype=float)
                # Adicionando os new borrowing do mês 09 nesta serie de new borrowing não acumulados
                new_borrowing_10q_sep = new_borrowing_10q[new_borrowing_10q.index.month == 9]
                new_borrowing_not_accum_10q = pd.concat([new_borrowing_not_accum, new_borrowing_10q_sep], sort=False)
                # Ordenando a serie para que o index fique igual aos outros indicadores
                new_borrowing_not_accum_10q = new_borrowing_not_accum_10q.sort_index(ascending=False)

            # Condição para as empresas que fazem o lançamento nos meses 12, 03 e 06
            if first_quarter == '12':
                if month == 3:
                    # Calculando o ano anterior, porque o 1Q é o mês 12
                    year_before = year - 1
                    # Calculando o new borrowing não acumulado do mês 03 -> "mês 03" - "mês 12 do ano anterior"
                    new_borrowing_03 = item - (new_borrowing_10q[f'12-{year_before}'].values[0])
                    lst_new_borrowing_not_accum[current_idx] = new_borrowing_03
        
                if month == 6:
                    # Calculando o ano anterior, porque o 1Q é o mês 12
                    year_before = year - 1
                    # Calculando o new borrowing não acumulad0 do mês 06 -> "mês 06" - ("mês 03" - "mês 12 do ano anterior") - "mês 12 do ano anterior"
                    new_borrowing_06 = item - (new_borrowing_10q[f'3-{year}'].values[0] - new_borrowing_10q[f'12-{year_before}'].values[0]) - new_borrowing_10q[f'12-{year_before}'].values[0]
                    lst_new_borrowing_not_accum[current_idx] = new_borrowing_06

                # Contem apenas o new borrowing não acumulados dos meses 03 e 06
                new_borrowing_not_accum = pd.Series(lst_new_borrowing_not_accum, dtype=float)
                # Adicionando o new borrowing do mês 12 nesta serie de new borrowing não acumulados 
                new_borrowing_10q_dec = new_borrowing_10q[new_borrowing_10q.index.month == 12]
                new_borrowing_not_accum_10q = pd.concat([new_borrowing_not_accum, new_borrowing_10q_dec], sort=False)
                # Ordenando a serie para que o index fique igual aos outros indicadores
                new_borrowing_not_accum_10q = new_borrowing_not_accum_10q.sort_index(ascending=False)

        # Retirando os dados de 2020
        new_borrowing_not_accum_10q_sliced = new_borrowing_not_accum_10q[0:len(new_borrowing_not_accum_10q)-3]

        # Calculando o new borrowing acumulado
        lst_new_borrowing_accum_10q = []
        for i in range(len(new_borrowing_not_accum_10q) - 3):
            new_borrowing_accum = new_borrowing_not_accum_10q[i:i+4].sum()  
            lst_new_borrowing_accum_10q.append(new_borrowing_accum)
        # Criando uma serie do new borrowing acumulado
        new_borrowing_accum_10q = pd.Series(lst_new_borrowing_accum_10q, index=idx[0:len(idx)-3])

        return new_borrowing_not_accum_10q, new_borrowing_not_accum_10q_sliced, new_borrowing_accum_10q
    

def indicador_debt_paid(
    ten_k: bool, 
    lst_itens_debt_paid: list, 
    df_cf: pd.DataFrame, 
    first_quarter: str, 
    idx: list
) -> pd.Series|tuple[pd.Series, pd.Series, pd.Series]:
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    lst_itens_debt_paid: list
        Lista dos itens do 'cash flow statement' (fluxo de caixa) que formam o debt paid.
    df_cf: pd.DataFrame
        DataFrame do 'cash flow statement' (fluxo de caixa).
    first_quarter: str
        Indicar qual é o número do 1Q da empresa.
    idx: list
        Lista que contém as datas que serão o index do indicador.

    Returns
    -------
    debt_paid: pd.Series
        Série pandas do indicador debt paid.
    debt_paid_not_accum_10q: pd.Series
        Série pandas do indicador debt paid trimestral não acumulado.
    debt_paid_not_accum_10q_sliced: pd.Series
        Série pandas do indicador debt paid trimestral não acumulado cortado.
    debt_paid_accum_10q: pd.Series
        Série pandas do indicador debt paid trimestral acumulado.
    """
    if ten_k:
        # Calculando o debt paid
        debt_paid = abs(df_cf.loc[lst_itens_debt_paid]).sum()
        return debt_paid
    
    else:
        # Calculando o debt paid trimestral
        debt_paid_10q = abs(df_cf.loc[lst_itens_debt_paid]).sum()
        # Calculando o debt paid não acumulado do 2Q e 3Q
        lst_debt_paid_not_accum = {}

        for current_idx, item in debt_paid_10q.items():
            month, year = current_idx.month, current_idx.year

            # Condição para as empresas que fazem o lançamento nos meses 02, 05 e 08
            if first_quarter == '02': 
                if month == 5:
                    # Calculando o debt paid não acumulado do mês 05, -> "mês 05" - "mês 02" 
                    debt_paid_05 = item - (debt_paid_10q[f'2-{year}'].values[0])
                    lst_debt_paid_not_accum[current_idx] = debt_paid_05

                if month == 8:
                    # Calculando o debt paid não acumulado do mês 08 -> "mês 08" - ("mês 05" - "mês 02") - "mês 02"        
                    debt_paid_08 = item - (debt_paid_10q[f'5-{year}'].values[0] - debt_paid_10q[f'2-{year}'].values[0]) - debt_paid_10q[f'2-{year}'].values[0]
                    lst_debt_paid_not_accum[current_idx] = debt_paid_08

                # Contem apenas os debt paid não acumulados dos meses 05 e 08
                debt_paid_not_accum = pd.Series(lst_debt_paid_not_accum, dtype=float)
                # Adicionando o debt paid do mês 02 nesta serie de debt paid não acumulados 
                debt_paid_10q_feb = debt_paid_10q[debt_paid_10q.index.month == 2]
                debt_paid_not_accum_10q = pd.concat([debt_paid_not_accum, debt_paid_10q_feb], sort=False)
                # Ordenando a serie para que o index fique igual aos outros indicadores
                debt_paid_not_accum_10q = debt_paid_not_accum_10q.sort_index(ascending=False)

            # Condição para as empresas que fazem o lançamento nos meses 03, 06 e 09
            if first_quarter == '03': 
                if month == 6:
                    # Calculando o debt paid não acumulado do mês 06 -> "mês 06" - "mês 03"
                    debt_paid_06 = item - (debt_paid_10q[f'3-{year}'].values[0])
                    lst_debt_paid_not_accum[current_idx] = debt_paid_06

                if month == 9:
                    # Calculando o debt paid não acumulado do mês 09 -> "mês 09" - ("mês 06" - "mês 03") - "mês 03"
                    debt_paid_09 = item - (debt_paid_10q[f'6-{year}'].values[0] - debt_paid_10q[f'3-{year}'].values[0]) - debt_paid_10q[f'3-{year}'].values[0]
                    lst_debt_paid_not_accum[current_idx] = debt_paid_09
                        
                # Contem apenas os debt paid não acumulados dos meses 06 e 09
                debt_paid_not_accum = pd.Series(lst_debt_paid_not_accum, dtype=float)
                # Adicionando o debt paid do mês 03 nesta serie de debt paid não acumulados 
                debt_paid_10q_mar = debt_paid_10q[debt_paid_10q.index.month == 3]
                debt_paid_not_accum_10q = pd.concat([debt_paid_not_accum, debt_paid_10q_mar], sort=False) 
                # Ordenando a serie para que o index fique igual aos outros indicadores
                debt_paid_not_accum_10q = debt_paid_not_accum_10q.sort_index(ascending=False)

            # Condição para as empresas que fazem o lançamento nos meses 04, 07 e 10
            if first_quarter == '04': 
                if month == 7:
                    # Calculando o debt paid não acumulado do mês 07, -> "mês 07" - "mês 04" 
                    debt_paid_07 = item - (debt_paid_10q[f'4-{year}'].values[0])
                    lst_debt_paid_not_accum[current_idx] = debt_paid_07

                if month == 10:
                    # Calculando o debt paid não acumulado do mês 10 -> "mês 10" - ("mês 07" - "mês 04") - "mês 04"        
                    debt_paid_10 = item - (debt_paid_10q[f'7-{year}'].values[0] - debt_paid_10q[f'4-{year}'].values[0]) - debt_paid_10q[f'4-{year}'].values[0]
                    lst_debt_paid_not_accum[current_idx] = debt_paid_10

                # Contem apenas os debt paid não acumulados dos meses 07 e 10
                debt_paid_not_accum = pd.Series(lst_debt_paid_not_accum, dtype=float)
                # Adicionando o debt paid do mês 04 nesta serie de debt paid não acumulados 
                debt_paid_10q_apr = debt_paid_10q[debt_paid_10q.index.month == 4]
                debt_paid_not_accum_10q = pd.concat([debt_paid_not_accum, debt_paid_10q_apr], sort=False)
                # Ordenando a serie para que o index fique igual aos outros indicadores
                debt_paid_not_accum_10q = debt_paid_not_accum_10q.sort_index(ascending=False)

            # Condição para as empresas que fazem o lançamento nos meses 09, 12 e 03
            if first_quarter == '09':
                if month == 12:
                    # Calculando o debt paid não acumulado do mês 12 -> "mês 12" - "mês 09"
                    debt_paid_12 = item - (debt_paid_10q[f'9-{year}'].values[0])
                    lst_debt_paid_not_accum[current_idx] = debt_paid_12

                if month == 3:
                    # Calculando o ano anterior, porque o mês 3 já está no próximo ano
                    year_before = year - 1 
                    # Calculando o debt paid não acumulado do mês 03 -> "mês 03 do próximo ano" - ("mês 12" - "mês 09") - "mês 09"
                    debt_paid_03 = item - (debt_paid_10q[f'12-{year_before}'].values[0] - debt_paid_10q[f'9-{year_before}'].values[0]) - debt_paid_10q[f'9-{year_before}'].values[0]
                    lst_debt_paid_not_accum[current_idx] = debt_paid_03

                # Contem apenas os debt paid não acumulados dos meses 12 e 3
                debt_paid_not_accum = pd.Series(lst_debt_paid_not_accum, dtype=float)
                # Adicionando os debt paid do mês 09 nesta serie de debt paid não acumulados
                debt_paid_10q_sep = debt_paid_10q[debt_paid_10q.index.month == 9]
                debt_paid_not_accum_10q = pd.concat([debt_paid_not_accum, debt_paid_10q_sep], sort=False)
                # Ordenando a serie para que o index fique igual aos outros indicadores
                debt_paid_not_accum_10q = debt_paid_not_accum_10q.sort_index(ascending=False)

            # Condição para as empresas que fazem o lançamento nos meses 12, 03 e 06
            if first_quarter == '12':
                if month == 3:
                    # Calculando o ano anterior, porque o 1Q é o mês 12
                    year_before = year - 1
                    # Calculando o debt paid não acumulado do mês 03 -> "mês 03" - "mês 12 do ano anterior"
                    debt_paid_03 = item - (debt_paid_10q[f'12-{year_before}'].values[0])
                    lst_debt_paid_not_accum[current_idx] = debt_paid_03
        
                if month == 6:
                    # Calculando o ano anterior, porque o 1Q é o mês 12
                    year_before = year - 1
                    # Calculando o debt paid não acumulado do mês 06 -> "mês 06" - ("mês 03" - "mês 12 do ano anterior") - "mês 12 do ano anterior"
                    debt_paid_06 = item - (debt_paid_10q[f'3-{year}'].values[0] - debt_paid_10q[f'12-{year_before}'].values[0]) - debt_paid_10q[f'12-{year_before}'].values[0]
                    lst_debt_paid_not_accum[current_idx] = debt_paid_06

                # Contem apenas o debt paid não acumulados dos meses 03 e 06
                debt_paid_not_accum = pd.Series(lst_debt_paid_not_accum, dtype=float)
                # Adicionando o debt paid do mês 12 nesta serie de debt paid não acumulados 
                debt_paid_10q_dec = debt_paid_10q[debt_paid_10q.index.month == 12]
                debt_paid_not_accum_10q = pd.concat([debt_paid_not_accum, debt_paid_10q_dec], sort=False)
                # Ordenando a serie para que o index fique igual aos outros indicadores
                debt_paid_not_accum_10q = debt_paid_not_accum_10q.sort_index(ascending=False)

        # Retirando os dados de 2020
        debt_paid_not_accum_10q_sliced = debt_paid_not_accum_10q[0:len(debt_paid_not_accum_10q)-3]

        # Calculando o debt paid acumulado
        lst_debt_paid_accum_10q = []
        for i in range(len(debt_paid_not_accum_10q) - 3):
            debt_paid_accum = debt_paid_not_accum_10q[i:i+4].sum()  
            lst_debt_paid_accum_10q.append(debt_paid_accum)
        # Criando uma serie do debt paid acumulado
        debt_paid_accum_10q = pd.Series(lst_debt_paid_accum_10q, index=idx[0:len(idx)-3])

        return debt_paid_not_accum_10q, debt_paid_not_accum_10q_sliced, debt_paid_accum_10q


def indicador_taxes_operating_income(ten_k: bool, df_is: pd.DataFrame, idx: list) -> pd.Series:
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    df_is: pd.DataFrame
        DataFrame do 'income statement' (DRE).
    idx: list   
        Lista que contém as datas que serão o index do indicador.

    Returns
    -------
    taxes_operating_income: pd.Series
        Série pandas do indicador taxes on operating income.
    taxes_operating_income_10q: pd.Series
        Série pandas do indicador taxes on operating income trimestral.
    """
    if ten_k:
        # Calculando a effective tax rate
        effective_tax_rate = df_is.loc['Provision for Income Taxes'] / df_is.loc['Income Before Provision for Income Taxes']

        # Calculando o taxes on operating income
        taxes_operating_income = round(df_is.loc['Operating Income'] * effective_tax_rate, 0)

        # Substituindo os valores negativos por zero
        taxes_operating_income = taxes_operating_income.clip(lower=0)

        return taxes_operating_income
    
    else:
        # Calculando a effective tax rate
        effective_tax_rate_10q = df_is.loc['Provision for Income Taxes'][0:-3] / df_is.loc['Income Before Provision for Income Taxes'][0:-3]

        # Calculando o EBIT acumulado
        ebit_10q = df_is.loc['Operating Income']

        lst_ebit_acum_10q = []
        for i in range(len(ebit_10q) - 3):
            ebit_accum = ebit_10q[i:i+4].sum()  
            lst_ebit_acum_10q.append(ebit_accum)
        # Criando uma serie do ebit acumulado
        ebit_accum_10q = pd.Series(lst_ebit_acum_10q, index=idx[0:len(idx)-3])

        # Calculando o taxes on operating income
        taxes_operating_income_10q = round(ebit_accum_10q * effective_tax_rate_10q, 0)

        # Substituindo os valores negativos por zero
        taxes_operating_income_10q = taxes_operating_income_10q.clip(lower=0)

        return taxes_operating_income_10q


def indicador_fcfe(
    
    ten_k: bool, 
    depreciacao: pd.Series, 
    capex: pd.Series, 
    change_non_cash_wc: pd.Series, 
    new_borrowing: pd.Series, 
    debt_paid: pd.Series, 
    df_is: pd.DataFrame,
    idx: list
) -> pd.Series:
    """
    Parameters:
    ten_k: se o arquivo for 10-K (True), se for 10-Q (False).
    depreciacao: serie pandas do indicador depreciação.
    capex: serie pandas do indicador capex.
    change_non_cash_wc: serie pandas do indicador change non cash working capital.
    new_borrowing: serie pandas do indicador new borrowing.
    debt_paid: serie pandas do indicador debt paid.
    df_is: dataframe do 'income statement' (DRE).
    idx: lista que contém as datas que serão o index do indicador.

    Returns:
    fcfe: serie pandas do indicador free cash flow to equity (FCFE).
    fcfe_10q: serie pandas do indicador free cash flow to equity (FCFE) trimestral.
    """
    if ten_k:
        # Calculando o fcfe
        fcfe = df_is.loc['Net Income'] + depreciacao - capex - change_non_cash_wc + (new_borrowing - debt_paid)
        return fcfe
    
    else:
        # Calculando o lucro líquido acumulado
        lucro_liq_10q = df_is.loc['Net Income']

        lst_lucro_liq_accum_10q = []
        for i in range(len(lucro_liq_10q) - 3):
            debt_paid_accum = lucro_liq_10q[i:i+4].sum()  
            lst_lucro_liq_accum_10q.append(debt_paid_accum)
        # Criando uma serie do lucro liquido acumulado
        lucro_liq_accum_10q = pd.Series(lst_lucro_liq_accum_10q, index=idx[0:len(idx)-3])

        # Calculando o fcfe trimestral
        fcfe_10q = lucro_liq_accum_10q + depreciacao - capex - change_non_cash_wc + (new_borrowing - debt_paid)
        return fcfe_10q


def indicador_fcff(
    ten_k: bool, 
    taxes_operating_income: pd.Series,
    depreciacao: pd.Series, 
    capex: pd.Series, 
    change_non_cash_wc: pd.Series, 
    df_is: pd.DataFrame,
    idx: list
) -> pd.Series:
    """
    Parameters:
    ten_k: se o arquivo for 10-K (True), se for 10-Q (False).
    taxes_operating_income: serie pandas do indicador taxes on operating income.
    depreciacao: serie pandas do indicador depreciação.
    capex: serie pandas do indicador capex.
    change_non_cash_wc: serie pandas do indicador change non cash working capital.
    df_is: dataframe do 'income statement' (DRE).
    idx: lista que contém as datas que serão o index do indicador.
    
    Returns:
    fcff: serie pandas do indicador free cash flow to firm (FCFF).
    fcff_10q: serie pandas do indicador free cash flow to firm (FCFF) trimestral.
    """
    if ten_k:
        # Calculando o fcff
        fcff = df_is.loc['Operating Income'] - taxes_operating_income + depreciacao - capex - change_non_cash_wc 
        return fcff
    
    else:
        # Calculando o EBIT acumulado
        ebit_10q = df_is.loc['Operating Income']

        lst_ebit_acum_10q = []
        for i in range(len(ebit_10q) - 3):
            ebit_accum = ebit_10q[i:i+4].sum()  
            lst_ebit_acum_10q.append(ebit_accum)
        # Criando uma serie do ebit acumulado
        ebit_accum_10q = pd.Series(lst_ebit_acum_10q, index=idx[0:len(idx)-3])

        # Calculando o fcff trimestral
        fcff_10q = ebit_accum_10q - taxes_operating_income + depreciacao - capex - change_non_cash_wc
        return fcff_10q


def indicador_div_nao_acum(df_cf: pd.DataFrame, first_quarter: str) -> pd.Series:
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    df_cf: pd.DataFrame
        DataFrame do 'cash flow statement' (fluxo de caixa).
    first_quarter: str
        Indicar qual é o número do 1Q da empresa.

    Returns
    -------
    div_not_accum_10q: pd.Series
        Série pandas do dividendo não acumulado.
    """
    # Transformando os valores dos dividendos em valores absolutos
    dividends_10q = abs(df_cf.loc['Payments of Dividends'][0:len(df_cf.columns)-3])

    # Calculando o dividendo não acumulado do 2Q e 3Q
    lst_div_not_accum = {}

    for idx, item in dividends_10q.items():
        month, year = idx.month, idx.year

        # Condição para as empresas que fazem o lançamento nos meses 02, 05 e 08
        if first_quarter == '02': 
            if month == 5:
                # Calculando o dividendo não acumulado do mês 05, -> "mês 05" - "mês 02" 
                dividends_05 = item - (dividends_10q[f'2-{year}'].values[0])
                lst_div_not_accum[idx] = dividends_05

            if month == 8:
                # Calculando o dividendo não acumulado do mês 08 -> "mês 08" - ("mês 05" - "mês 02") - "mês 02"        
                dividends_08 = item - (dividends_10q[f'5-{year}'].values[0] - dividends_10q[f'2-{year}'].values[0]) - dividends_10q[f'2-{year}'].values[0]
                lst_div_not_accum[idx] = dividends_08

            # Contem apenas os dividendos não acumulados dos meses 05 e 08
            div_not_accum = pd.Series(lst_div_not_accum, dtype=float)
            # Adicionando os dividendos do mês 02 nesta serie de dividendos não acumulados 
            dividends_10q_feb = dividends_10q[dividends_10q.index.month == 2]
            div_not_accum_10q = pd.concat([div_not_accum, dividends_10q_feb], sort=False)
            # Ordenando a serie para que o index fique igual aos outros indicadores
            div_not_accum_10q = div_not_accum_10q.sort_index(ascending=False)

        # Condição para as empresas que fazem o lançamento nos meses 03, 06 e 09
        if first_quarter == '03': 
            if month == 6:
                # Calculando o dividendo não acumulado do mês 06, -> "mês 06" - "mês 03" 
                dividends_06 = item - (dividends_10q[f'3-{year}'].values[0])
                lst_div_not_accum[idx] = dividends_06

            if month == 9:
                # Calculando o dividendo não acumulado do mês 09 -> "mês 09" - ("mês 06" - "mês 03") - "mês 03"        
                dividends_09 = item - (dividends_10q[f'6-{year}'].values[0] - dividends_10q[f'3-{year}'].values[0]) - dividends_10q[f'3-{year}'].values[0]
                lst_div_not_accum[idx] = dividends_09

            # Contem apenas os dividendos não acumulados dos meses 06 e 09
            div_not_accum = pd.Series(lst_div_not_accum, dtype=float)
            # Adicionando os dividendos do mês 03 nesta serie de dividendos não acumulados 
            dividends_10q_mar = dividends_10q[dividends_10q.index.month == 3]
            div_not_accum_10q = pd.concat([div_not_accum, dividends_10q_mar], sort=False)
            # Ordenando a serie para que o index fique igual aos outros indicadores
            div_not_accum_10q = div_not_accum_10q.sort_index(ascending=False)

        # Condição para as empresas que fazem o lançamento nos meses 04, 07 e 10
        if first_quarter == '04': 
            if month == 7:
                # Calculando o dividendo não acumulado do mês 07, -> "mês 07" - "mês 04" 
                dividends_07 = item - (dividends_10q[f'4-{year}'].values[0])
                lst_div_not_accum[idx] = dividends_07

            if month == 10:
                # Calculando o dividendo não acumulado do mês 10 -> "mês 10" - ("mês 07" - "mês 04") - "mês 04"        
                dividends_10 = item - (dividends_10q[f'7-{year}'].values[0] - dividends_10q[f'4-{year}'].values[0]) - dividends_10q[f'4-{year}'].values[0]
                lst_div_not_accum[idx] = dividends_10

            # Contem apenas os dividendos não acumulados dos meses 07 e 10
            div_not_accum = pd.Series(lst_div_not_accum, dtype=float)
            # Adicionando os dividendos do mês 04 nesta serie de dividendos não acumulados 
            dividends_10q_apr = dividends_10q[dividends_10q.index.month == 4]
            div_not_accum_10q = pd.concat([div_not_accum, dividends_10q_apr], sort=False)
            # Ordenando a serie para que o index fique igual aos outros indicadores
            div_not_accum_10q = div_not_accum_10q.sort_index(ascending=False)

        # Condição para as empresas que fazem o lançamento nos meses 09, 12 e 03
        if first_quarter == '09':
            if month == 12:
                # Calculando o dividendo não acumulado do mês 12 -> "mês 12" - "mês 09"
                dividends_12 = item - (dividends_10q[f'9-{year}'].values[0])
                lst_div_not_accum[idx] = dividends_12

            if month == 3:
                # Calculando o ano anterior, porque o mês 3 já está no próximo ano
                year_before = year - 1 
                # Calculando o dividendo não acumulado do mês 03 -> "mês 03 do próximo ano" - ("mês 12" - "mês 09") - "mês 09"
                dividends_03 = item - (dividends_10q[f'12-{year_before}'].values[0] - dividends_10q[f'9-{year_before}'].values[0]) - dividends_10q[f'9-{year_before}'].values[0]
                lst_div_not_accum[idx] = dividends_03

            # Contem apenas os dividendos não acumulados dos meses 12 e 3
            buyback_not_accum = pd.Series(lst_div_not_accum, dtype=float)
            # Adicionando os dividendos do mês 09 nesta serie de dividendos não acumulados
            dividends_10q_sep = dividends_10q[dividends_10q.index.month == 9]
            div_not_accum_10q = pd.concat([buyback_not_accum, dividends_10q_sep], sort=False)
            # Ordenando a serie para que o index fique igual aos outros indicadores
            div_not_accum_10q = div_not_accum_10q.sort_index(ascending=False)

        # Condição para as empresas que fazem o lançamento nos meses 12, 03 e 06
        if first_quarter == '12':
            if month == 3:
                # Calculando o ano anterior, porque o 1Q é o mês 12
                year_before = year - 1
                # Calculando o dividendo não acumulado do mês 03 -> "mês 03" - "mês 12 do ano anterior"
                dividends_03 = item - (dividends_10q[f'12-{year_before}'].values[0])
                lst_div_not_accum[idx] = dividends_03
    
            if month == 6:
                # Calculando o ano anterior, porque o 1Q é o mês 12
                year_before = year - 1
                # Calculando o dividendo não acumulado do mês 06 -> "mês 06" - ("mês 03" - "mês 12 do ano anterior") - "mês 12 do ano anterior"
                dividends_06 = item - (dividends_10q[f'3-{year}'].values[0] - dividends_10q[f'12-{year_before}'].values[0]) - dividends_10q[f'12-{year_before}'].values[0]
                lst_div_not_accum[idx] = dividends_06

            # Contem apenas os dividendos não acumulados dos meses 03 e 06
            div_not_accum = pd.Series(lst_div_not_accum, dtype=float)
            # Adicionando o dividendo do mês 12 nesta serie de dividendos não acumulados 
            dividends_10q_dec = dividends_10q[dividends_10q.index.month == 12]
            div_not_accum_10q = pd.concat([div_not_accum, dividends_10q_dec], sort=False)
            # Ordenando a serie para que o index fique igual aos outros indicadores
            div_not_accum_10q = div_not_accum_10q.sort_index(ascending=False)

    return div_not_accum_10q


def indicador_dpa(ten_k: bool, df_cf: pd.Series, df_is: pd.Series, div_nao_acum: pd.Series) -> pd.Series:
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    df_cf: pd.DataFrame
        DataFrame do 'cash flow statement' (fluxo de caixa).
    df_is: pd.DataFrame
        DataFrame do 'income statement' (DRE).
    
    Returns
    -------
    dpa: pd.Series
        Série pandas do indicador dpa.
    dpa_10q: pd.Series
        Série pandas do indicador dpa trimestral.
    """
    if ten_k:
        # Calculando o Dividendo por Ação
        dpa = np.round(abs(df_cf.loc['Payments of Dividends']) / df_is.loc['Number of Shares - Basic'], 2)
        return dpa
    else:
        # Calculando o Dividendo por Ação
        dpa_10q = np.round(div_nao_acum / df_is.loc['Number of Shares - Basic'][0:len(df_is.columns)-3], 2)
        return dpa_10q
    

def indicador_dpa_2(ten_k: bool, df_cf: pd.Series, df_is: pd.Series, div_nao_acum: pd.Series) -> pd.Series:
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    df_cf: pd.DataFrame
        DataFrame do 'cash flow statement' (fluxo de caixa).
    df_is: pd.DataFrame
        DataFrame do 'income statement' (DRE).
    
    Returns
    -------
    dpa: pd.Series
        Série pandas do indicador dpa.
    dpa_10q: pd.Series
        Série pandas do indicador dpa trimestral.
    NOTE: Eu estou dividindo por 1000, porque no income statement o valor do nº ações está numa escala diferente.
    Apple e Vistra são empresas que eu utilizo essa função.
    """
    if ten_k:
        # Calculando o Dividendo por Ação
        dpa = np.round(abs(df_cf.loc['Payments of Dividends']) / (df_is.loc['Number of Shares - Basic']/1000), 2)
        return dpa
    else:
        # Calculando o Dividendo por Ação
        dpa_10q = np.round(div_nao_acum / (df_is.loc['Number of Shares - Basic'][0:len(df_is.columns)-3]/1000), 2)
        return dpa_10q
    

def indicador_payout(ten_k: bool, df_cf: pd.Series, df_is: pd.Series, div_nao_acum: pd.Series) -> pd.Series:
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    df_cf: pd.DataFrame
        DataFrame do 'cash flow statement' (fluxo de caixa).
    df_is: pd.DataFrame
        DataFrame do 'income statement' (DRE).
    div_nao_acum: pd.Series
        Série pandas do dividendo não acumulado.
    
    Returns
    -------
    payout: pd.Series
        Série pandas do indicador payout.
    payout_10q: pd.Series
        Série pandas do indicador payout trimestral.
    """
    if ten_k:
        # Calculando o Payout
        payout = np.round((abs(df_cf.loc['Payments of Dividends']) / df_is.loc['Net Income']) * 100, 2)
        return payout
    else:
        # Calculando o Payout e retirando os dados de 2020
        payout_10q = np.round((div_nao_acum/ df_is.loc['Net Income'][0:len(df_cf.columns)-3]) * 100, 2)
        return payout_10q


def indicador_buyback(ten_k: bool, df_cf: pd.Series, first_quarter:str) -> pd.Series:
    """
    Parameters
    ----------
    ten_k: bool
        Se o arquivo for 10-K (True), se for 10-Q (False).
    df_cf: pd.DataFrame
        DataFrame do 'cash flow statement' (fluxo de caixa).
    first_quarter: str
        Indicar qual é o número do 1Q da empresa.

    Returns
    -------
    buyback: pd.Series
        Série pandas do indicador buyback.
    buyback_not_accum_10q: pd.Series
        Série pandas do indicador buyback trimestral.
    """
    if ten_k:
        # Calculando o buyback
        buyback = abs(df_cf.loc['Payments for Repurchase of Common Stock'])
        return buyback
    else:
        # Calculando o buyback
        buyback_10q = abs(df_cf.loc['Payments for Repurchase of Common Stock'][0:len(df_cf.columns)-3])
        # Calculando o buyback não acumulado do 2Q e 3Q 
        lst_buyback_not_accum = {}
        lst_months_buyback = []

        for idx, item in buyback_10q.items():
            month, year = idx.month, idx.year
            lst_months_buyback.append(month)

            # Condição para as empresas que fazem o lançamento nos meses 02, 05 e 08
            if first_quarter == '02': 
                if month == 5:
                    # Calculando o buyback não acumulado do mês 06 -> "mês 05" - "mês 02"
                    buyback_05 = item - (buyback_10q[f'2-{year}'].values[0])
                    lst_buyback_not_accum[idx] = buyback_05

                if month == 8:
                    # Calculando o buyback não acumulado do mês 08 -> "mês 08" - ("mês 05" - "mês 02") - "mês 02"
                    buyback_08 = item - (buyback_10q[f'5-{year}'].values[0] - buyback_10q[f'2-{year}'].values[0]) - buyback_10q[f'2-{year}'].values[0]
                    lst_buyback_not_accum[idx] = buyback_08
                    
                # Contem apenas os buybacks não acumulados dos meses 05 e 08
                buyback_not_accum = pd.Series(lst_buyback_not_accum, dtype=float)
                # Adicionando os buybacks do mês 02 nesta serie de buybacks não acumulados 
                buyback_10q_feb = buyback_10q[buyback_10q.index.month == 2]
                buyback_not_accum_10q = pd.concat([buyback_not_accum, buyback_10q_feb], sort=False) 
                # Ordenando a serie para que o index fique igual aos outros indicadores
                buyback_not_accum_10q = buyback_not_accum_10q.sort_index(ascending=False)

            # Condição para as empresas que fazem o lançamento nos meses 03, 06 e 09
            if first_quarter == '03': 
                if month == 6:
                    # Calculando o buyback não acumulado do mês 06 -> "mês 06" - "mês 03"
                    buyback_06 = item - (buyback_10q[f'3-{year}'].values[0])
                    lst_buyback_not_accum[idx] = buyback_06

                if month == 9:
                    # Calculando o buyback não acumulado do mês 09 -> "mês 09" - ("mês 06" - "mês 03") - "mês 03"
                    buyback_09 = item - (buyback_10q[f'6-{year}'].values[0] - buyback_10q[f'3-{year}'].values[0]) - buyback_10q[f'3-{year}'].values[0]
                    lst_buyback_not_accum[idx] = buyback_09
                    
                # Contem apenas os buybacks não acumulados dos meses 06 e 09
                buyback_not_accum = pd.Series(lst_buyback_not_accum, dtype=float)
                # Adicionando os buybacks do mês 03 nesta serie de buybacks não acumulados 
                buyback_10q_mar = buyback_10q[buyback_10q.index.month == 3]
                buyback_not_accum_10q = pd.concat([buyback_not_accum, buyback_10q_mar], sort=False) 
                # Ordenando a serie para que o index fique igual aos outros indicadores
                buyback_not_accum_10q = buyback_not_accum_10q.sort_index(ascending=False)

            # Condição para as empresas que fazem o lançamento nos meses 04, 07 e 10
            if first_quarter == '04': 
                if month == 7:
                    # Calculando o buyback não acumulado do mês 07 -> "mês 07" - "mês 04"
                    buyback_07 = item - (buyback_10q[f'4-{year}'].values[0])
                    lst_buyback_not_accum[idx] = buyback_07

                if month == 10:
                    # Calculando o buyback não acumulado do mês 10 -> "mês 10" - ("mês 07" - "mês 04") - "mês 04"
                    buyback_10 = item - (buyback_10q[f'7-{year}'].values[0] - buyback_10q[f'4-{year}'].values[0]) - buyback_10q[f'4-{year}'].values[0]
                    lst_buyback_not_accum[idx] = buyback_10

                # Contem apenas os buybacks não acumulados dos meses 07 e 10
                buyback_not_accum = pd.Series(lst_buyback_not_accum, dtype=float)
                # Adicionando os buybacks do mês 04 nesta serie de buybacks não acumulados 
                buyback_10q_apr = buyback_10q[buyback_10q.index.month == 4]
                buyback_not_accum_10q = pd.concat([buyback_not_accum, buyback_10q_apr], sort=False)
                # Ordenando a serie para que o index fique igual aos outros indicadores
                buyback_not_accum_10q = buyback_not_accum_10q.sort_index(ascending=False)

            # Condição para as empresas que fazem o lançamento nos meses 09, 12 e 03
            if first_quarter == '09':
                if month == 12:
                    # Calculando o buyback não acumulado do mês 12 -> "mês 12" - "mês 09"
                    buyback_12 = item - (buyback_10q[f'9-{year}'].values[0])
                    lst_buyback_not_accum[idx] = buyback_12

                if month == 3:
                    # Calculando o ano anterior, porque o mês 3 já está no próximo ano
                    year_before = year - 1 

                    # Calculando o buyback não acumulado do mês 03 -> "mês 03 do próximo ano" - ("mês 12" - "mês 09") - "mês 09"
                    buyback_03 = item - (buyback_10q[f'12-{year_before}'].values[0] - buyback_10q[f'9-{year_before}'].values[0]) - buyback_10q[f'9-{year_before}'].values[0]
                    lst_buyback_not_accum[idx] = buyback_03

                # Contem apenas os buybacks não acumulados dos meses 12 e 3
                buyback_not_accum = pd.Series(lst_buyback_not_accum, dtype=float)
                # Adicionando os buybacks do mês 09 nesta serie de buybacks não acumulados
                buyback_10q_sep = buyback_10q[buyback_10q.index.month == 9]
                buyback_not_accum_10q = pd.concat([buyback_not_accum, buyback_10q_sep], sort=False)
                # Ordenando a serie para que o index fique igual aos outros indicadores
                buyback_not_accum_10q = buyback_not_accum_10q.sort_index(ascending=False)    

            # Condição para as empresas que fazem o lançamento nos meses 12, 03 e 06
            if first_quarter == '12':
                if month == 3:
                    # Calculando o ano anterior, porque o 1Q é o mês 12
                    year_before = year - 1

                    # Calculando o buyback não acumulado do mês 03 -> "mês 03" - "mês 12 do ano anterior"
                    buyback_03 = item - (buyback_10q[f'12-{year_before}'].values[0])
                    lst_buyback_not_accum[idx] = buyback_03
        
                if month == 6:
                    # Calculando o ano anterior, porque o 1Q é o mês 12
                    year_before = year - 1 
                    
                    # Calculando o buyback não acumulado do mês 06 -> "mês 06" - ("mês 03" - "mês 12 do ano anterior") - "mês 12 do ano anterior"
                    buyback_06 = item - (buyback_10q[f'03-{year}'].values[0] - buyback_10q[f'12-{year_before}'].values[0]) - buyback_10q[f'12-{year_before}'].values[0]
                    lst_buyback_not_accum[idx] = buyback_06

                # Contem apenas os buybacks não acumulados dos meses 03 e 06
                buyback_not_accum = pd.Series(lst_buyback_not_accum, dtype=float)
                # Adicionando os buybacks do mês 12 nesta serie de buybacks não acumulados 
                buyback_10q_dec = buyback_10q[buyback_10q.index.month == 12]
                buyback_not_accum_10q = pd.concat([buyback_not_accum, buyback_10q_dec], sort=False)
                # Ordenando a serie para que o index fique igual aos outros indicadores
                buyback_not_accum_10q = buyback_not_accum_10q.sort_index(ascending=False)

    return buyback_not_accum_10q