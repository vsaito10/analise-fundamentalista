from datetime import datetime, timedelta # noqa: F401
from functools import reduce
from pathlib import Path # noqa: F401
from plotly.subplots import make_subplots
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm
import matplotlib.pyplot as plt # noqa: F401
import numpy as np
import pandas as pd
import polars as pl 
import scipy.cluster.hierarchy as sch
import plotly.express as px # noqa: F401
import plotly.graph_objects as go
import statsmodels.api as sm
import seaborn as sns # noqa: F401
import yfinance as yf


def filtro_etf(path: str) -> pd.DataFrame:
    """
    Função que filtra a composição da carteira do ETF (IBOV e SMAL11).

    Parameters
    ----------
    path: str
        Caminho do arquivo do ETF.
    
    Returns
    -------
    pd.DataFrame
        DataFrame do ETF.
    """
    # Abrindo o arquivo da carteira teórica
    df = pd.read_csv(
        path, 
        sep=';',
        encoding='ISO-8859-1', 
        engine='python', 
        header=1, # cabeçalho vira a 2º linha da tabela
        skipfooter=2, # pula as duas últimas linhas da tabela
        index_col=False,
        dtype={'Qtde. Teórica': str} # No IEE, quando eu abria o arquivo, ele estava retirando dois zeros dos números da coluna 'Qtde. Teórica'
    )
    
    # Convertendo o dtypes das colunas
    df['Código'] = df['Código'].astype(str)
    df['Ação'] = df['Ação'].astype(str)
    df['Tipo'] = df['Tipo'].astype(str)
    df['Qtde. Teórica'] = df['Qtde. Teórica'].str.replace('.', '', regex=False)
    df['Qtde. Teórica'] = df['Qtde. Teórica'].astype(float)
    df['Part. (%)'] = df['Part. (%)'].str.replace(',', '.', regex=False)
    df['Part. (%)'] = df['Part. (%)'].astype(float)

    return df


def filtro_planilha_statusinvest(path: str)-> pd.DataFrame:
  """
  Função que filtra o arquivo de indicadores fundamentalistas do Statusinvest.

  Parameters
  ----------
  path: str
    Caminho do arquivo do Statusinvest.
  
  Returns
  -------
  pl.DataFrame
    DataFrame do Statusinvest.
  """
  df = pd.read_csv(path, sep=';')

  # Retirando os espaços (frente e trás) da string das colunas do df
  df.columns = df.columns.str.strip()

  # Selecionando as colunas necessárias
  df = df[['TICKER', 'PRECO', 'DY', 'P/L', 
           'P/VP', 'MARGEM BRUTA', 'MARG. LIQUIDA', 'EV/EBIT', 
           'DIVIDA LIQUIDA / EBIT', 'DIV. LIQ. / PATRI.', 'PSR', 'ROE', 
           'ROIC', 'LIQUIDEZ MEDIA DIARIA', 'LPA', 'VPA', 'VALOR DE MERCADO']]
  
  # Retirando as ações que não possuem liquidez diária
  df.dropna(axis=0, subset=['LIQUIDEZ MEDIA DIARIA'], inplace=True)

  # Retirando tickers que não são mais negociados
  df= df[(df['TICKER'] != 'HGTX3') & 
         (df['TICKER'] != 'LAME3') &
         (df['TICKER'] != 'LAME4') &
         (df['TICKER'] != 'OMGE3') & 
         (df['TICKER'] != 'GNDI3') &
         (df['TICKER'] != 'BIDI11') &
         (df['TICKER'] != 'BIDI3') &
         (df['TICKER'] != 'BIDI4') & 
         (df['TICKER'] != 'CESP6') &
         (df['TICKER'] != 'POWE3') &
         (df['TICKER'] != 'PNVL4') &
         (df['TICKER'] != 'LCAM3') &
         (df['TICKER'] != 'CESP3') & 
         (df['TICKER'] != 'CESP5') &
         (df['TICKER'] != 'EEEL3') & 
         (df['TICKER'] != 'EEEL4') &
         (df['TICKER'] != 'MOSI3') &
         (df['TICKER'] != 'IGTA3') &
         (df['TICKER'] != 'BKBR3') & 
         (df['TICKER'] != 'DMMO3') &    
         (df['TICKER'] != 'DMMO11') &
         (df['TICKER'] != 'PARD3') &
         (df['TICKER'] != 'MEGA3') &
         (df['TICKER'] != 'QVQP3B') 
  ]

  # Criando a coluna de segmento
  conditions = [
    # AGUA E SANEAMENTO
    (df['TICKER'] == 'AMBP3') | (df['TICKER'] == 'CASN3')	| (df['TICKER'] == 'CASN4')	| 
    (df['TICKER'] == 'CSMG3') | (df['TICKER'] == 'ORVR3') | (df['TICKER'] == 'SAPR11')| 
    (df['TICKER'] == 'SAPR3') | (df['TICKER'] == 'SAPR4') | (df['TICKER'] == 'SBSP3'),

    # AGRICULTURA
    (df['TICKER'] == 'AGRO3') | (df['TICKER'] == 'AGXY3') | (df['TICKER'] == 'APTI4') |
    (df['TICKER'] == 'FRTA3') | (df['TICKER'] == 'LAND3') | (df['TICKER'] == 'SLCE3') | 
    (df['TICKER'] == 'SOJA3') | (df['TICKER'] == 'TESA3') | (df['TICKER'] == 'TTEN3'),

    #ALIMENTOS DIVERSOS
    (df['TICKER'] == 'CAML3') | (df['TICKER'] == 'JOPA3') | (df['TICKER'] == 'FICT3') |
    (df['TICKER'] == 'JOPA4') | (df['TICKER'] == 'MDIA3') |(df['TICKER'] == 'ODER4'),

    # ALUGUEL DE CARRO
    (df['TICKER'] == 'MOVI3') | (df['TICKER'] == 'RENT3'),

    # ALUGUEL DE MAQUINAS E EQUIPAMENTOS
    (df['TICKER'] == 'ARML3') | (df['TICKER'] == 'MILS3') | (df['TICKER'] == 'VAMO3'),

    # ARMAS E MUNICAO
    (df['TICKER'] == 'TASA3') | (df['TICKER'] == 'TASA4'),

    # AUTOMOVEIS E MOTOCICLETAS
    (df['TICKER'] == 'AMOB3') | (df['TICKER'] == 'LEVE3') | (df['TICKER'] == 'MYPK3') | 
    (df['TICKER'] == 'PLAS3'),	

    # AVIACAO
    (df['TICKER'] == 'AZUL4') | (df['TICKER'] == 'GOLL4'),

    # ACUCAR E ALCOOL
    (df['TICKER'] == 'BSEV3') | (df['TICKER'] == 'JALL3')	| (df['TICKER'] == 'SMTO3'),	

    # BANCOS
    (df['TICKER'] == 'ABCB4') | (df['TICKER'] == 'BAZA3') | (df['TICKER'] == 'BBAS3') |
    (df['TICKER'] == 'BBDC3') | (df['TICKER'] == 'BBDC4') | (df['TICKER'] == 'BEES3') |
    (df['TICKER'] == 'BEES4') | (df['TICKER'] == 'BGIP3') | (df['TICKER'] == 'BGIP4') |
    (df['TICKER'] == 'BMEB3') | (df['TICKER'] == 'BMEB4')	| (df['TICKER'] == 'BMGB4') |
    (df['TICKER'] == 'BMIN3') | (df['TICKER'] == 'BMIN4') | (df['TICKER'] == 'BNBR3') |
    (df['TICKER'] == 'BPAC11')| (df['TICKER'] == 'BPAC3') | (df['TICKER'] == 'BPAC5') |
    (df['TICKER'] == 'BPAN4') | (df['TICKER'] == 'BPAR3') | (df['TICKER'] == 'BRBI11')|
    (df['TICKER'] == 'BRIV3') | (df['TICKER'] == 'BRIV4') | (df['TICKER'] == 'BRSR3') | 
    (df['TICKER'] == 'BRSR5') | (df['TICKER'] == 'BRSR6') | (df['TICKER'] == 'BSLI3') | 
    (df['TICKER'] == 'BSLI4') | (df['TICKER'] == 'ITSA3') | (df['TICKER'] == 'ITSA4') | 
    (df['TICKER'] == 'ITUB3') | (df['TICKER'] == 'ITUB4') | (df['TICKER'] == 'MODL11')| 
    (df['TICKER'] == 'MODL3') | (df['TICKER'] == 'MODL4') | (df['TICKER'] == 'PINE3') | 
    (df['TICKER'] == 'PINE4') | (df['TICKER'] == 'RPAD3') | (df['TICKER'] == 'RPAD5') | 
    (df['TICKER'] == 'RPAD6') | (df['TICKER'] == 'SANB11')| (df['TICKER'] == 'SANB3') | 
    (df['TICKER'] == 'SANB4'),

    # BEBIDAS
    (df['TICKER'] == 'ABEV3'),

    # CARNES E DERIVADOS
    (df['TICKER'] == 'BAUH4') | (df['TICKER'] == 'BEEF3') | (df['TICKER'] == 'BRFS3') |
    (df['TICKER'] == 'JBSS3') | (df['TICKER'] == 'MNPR3') | (df['TICKER'] == 'MBRF3') |
    (df['TICKER'] == 'MRFG3'),

    # COMPUTADORES E EQUIPAMENTOS
    (df['TICKER'] == 'INTB3') | (df['TICKER'] == 'MLAS3') | (df['TICKER'] == 'POSI3'),
    
    # CONSTRUÇAO CIVIL
    (df['TICKER'] == 'AVLL3') | (df['TICKER'] == 'CALI3') | (df['TICKER'] == 'CRDE3') |
    (df['TICKER'] == 'CURY3') | (df['TICKER'] == 'CYRE3') | (df['TICKER'] == 'DIRR3') | 
    (df['TICKER'] == 'EVEN3') | (df['TICKER'] == 'EZTC3') | (df['TICKER'] == 'FIEI3') |
    (df['TICKER'] == 'GFSA3') |	(df['TICKER'] == 'HBOR3') | (df['TICKER'] == 'JFEN3') | 
    (df['TICKER'] == 'JHSF3')	| (df['TICKER'] == 'LAVV3') | (df['TICKER'] == 'MDNE3') | 
    (df['TICKER'] == 'MELK3') | (df['TICKER'] == 'MRVE3') | (df['TICKER'] == 'MTRE3') | 
    (df['TICKER'] == 'PDGR3') | (df['TICKER'] == 'PLPL3') | (df['TICKER'] == 'RDNI3') | 
    (df['TICKER'] == 'RSID3') | (df['TICKER'] == 'TCSA3') | (df['TICKER'] == 'TEND3') | 
    (df['TICKER'] == 'TRIS3') | (df['TICKER'] == 'VIVR3'),

    # CONSTRUÇAO E ENGENHARIA
    (df['TICKER'] == 'AZEV3') | (df['TICKER'] == 'AZEV4') | (df['TICKER'] == 'ETER3') |
    (df['TICKER'] == 'HAGA3') | (df['TICKER'] == 'HAGA4') | (df['TICKER'] == 'PTBL3') |
    (df['TICKER'] == 'SOND3') | (df['TICKER'] == 'SOND5') | (df['TICKER'] == 'SOND6') |
    (df['TICKER'] == 'TCNO3') | (df['TICKER'] == 'TCNO4'),

    # ENERGIA ELETRICA
    (df['TICKER'] == 'AESB3') | (df['TICKER'] == 'AFLT3') | (df['TICKER'] == 'ALUP11')|
    (df['TICKER'] == 'ALUP3') | (df['TICKER'] == 'ALUP4') | (df['TICKER'] == 'AURE3') |
    (df['TICKER'] == 'AXIA3') | (df['TICKER'] == 'AXIA5') | (df['TICKER'] == 'AXIA6') |
    (df['TICKER'] == 'CBEE3') | (df['TICKER'] == 'CEBR3') | (df['TICKER'] == 'CEBR5') |
    (df['TICKER'] == 'CEBR6') | (df['TICKER'] == 'CEEB3') | (df['TICKER'] == 'CEEB5') |
    (df['TICKER'] == 'CEED3') | (df['TICKER'] == 'CEED4') | (df['TICKER'] == 'CEPE5') |
    (df['TICKER'] == 'CEPE6') | (df['TICKER'] == 'CLSC3') | (df['TICKER'] == 'CLSC4') | 
    (df['TICKER'] == 'CMIG3') | (df['TICKER'] == 'CMIG4') | (df['TICKER'] == 'COCE3') | 
    (df['TICKER'] == 'COCE5') | (df['TICKER'] == 'COCE6') | (df['TICKER'] == 'CPFE3') |
    (df['TICKER'] == 'CPLE11')| (df['TICKER'] == 'CPLE3') | (df['TICKER'] == 'CPLE5') | 
    (df['TICKER'] == 'CPLE6') | (df['TICKER'] == 'CSRN3') | (df['TICKER'] == 'CSRN5') | 
    (df['TICKER'] == 'CSRN6') | (df['TICKER'] == 'EGIE3') | (df['TICKER'] == 'EKTR3') | 
    (df['TICKER'] == 'EKTR4') | (df['TICKER'] == 'ELET3') | (df['TICKER'] == 'ELET5') | 
    (df['TICKER'] == 'ELET6') | (df['TICKER'] == 'EMAE3') | (df['TICKER'] == 'EMAE4') | 
    (df['TICKER'] == 'ENBR3') | (df['TICKER'] == 'ENEV3') | (df['TICKER'] == 'ENGI11')| 
    (df['TICKER'] == 'ENGI3') | (df['TICKER'] == 'ENGI4') | (df['TICKER'] == 'ENMT3') | 
    (df['TICKER'] == 'ENMT4') | (df['TICKER'] == 'EQPA3') | (df['TICKER'] == 'EQPA5') | 
    (df['TICKER'] == 'EQPA6') | (df['TICKER'] == 'EQPA7') | (df['TICKER'] == 'EQTL3') | 
    (df['TICKER'] == 'GEPA3') | (df['TICKER'] == 'GEPA4') | (df['TICKER'] == 'GPAR3') | 
    (df['TICKER'] == 'ISAE3') | (df['TICKER'] == 'ISAE4') | (df['TICKER'] == 'LIGT3') |
    (df['TICKER'] == 'LIPR3') | (df['TICKER'] == 'MEGA3') | (df['TICKER'] == 'NEOE3') | 
    (df['TICKER'] == 'REDE3') | (df['TICKER'] == 'RNEW11')| (df['TICKER'] == 'RNEW3') | 
    (df['TICKER'] == 'RNEW4') | (df['TICKER'] == 'SRNA3') | (df['TICKER'] == 'TAEE11')| 
    (df['TICKER'] == 'TAEE3') | (df['TICKER'] == 'TAEE4') | (df['TICKER'] == 'TRPL3') | 
    (df['TICKER'] == 'TRPL4'),

    # EQUIPAMENTOS E SERVIÇOS
    (df['TICKER'] == 'LUPA3') | (df['TICKER'] == 'OSXB3'),

    # EXPLORAÇAO DE IMOVEIS
    (df['TICKER'] == 'ALOS3') | (df['TICKER'] == 'ALSO3') | (df['TICKER'] == 'BRML3') | 
    (df['TICKER'] == 'BRPR3') | (df['TICKER'] == 'GSHP3') | (df['TICKER'] == 'HBRE3') | 
    (df['TICKER'] == 'HBTS5') | (df['TICKER'] == 'IGBR3') | (df['TICKER'] == 'IGTI11')| 
    (df['TICKER'] == 'IGTI3') | (df['TICKER'] == 'IGTI4') | (df['TICKER'] == 'LOGG3')	| 
    (df['TICKER'] == 'MULT3') | (df['TICKER'] == 'NEXP3') | (df['TICKER'] == 'SCAR3') | 
    (df['TICKER'] == 'SYNE3'),

    #EXPLORAÇAO DE RODOVIAS
    (df['TICKER'] == 'CCRO3') | (df['TICKER'] == 'ECOR3') | (df['TICKER'] == 'MOTV3') | 
    (df['TICKER'] == 'TPIS3'),

    #FIOS E TECIDOS
    (df['TICKER'] == 'CEDO3') | (df['TICKER'] == 'CEDO4') | (df['TICKER'] == 'CTKA3') |
    (df['TICKER'] == 'CTKA4') | (df['TICKER'] == 'CTNM3') | (df['TICKER'] == 'CTNM4') |
    (df['TICKER'] == 'CTSA3') | (df['TICKER'] == 'CTSA4') | (df['TICKER'] == 'DOHL3') |
    (df['TICKER'] == 'DOHL4') | (df['TICKER'] == 'ECPR3')	| (df['TICKER'] == 'ECPR4') |
    (df['TICKER'] == 'PTNT3') | (df['TICKER'] == 'PTNT4') | (df['TICKER'] == 'SGPS3') |
    (df['TICKER'] == 'TEKA3') | (df['TICKER'] == 'TEKA4') | (df['TICKER'] == 'TXRX3') |
    (df['TICKER'] == 'TXRX4'),

    #GAS
    (df['TICKER'] == 'CEGR3') | (df['TICKER'] == 'CGAS3') | (df['TICKER'] == 'CGAS5'),

    #GESTAO DE RECURSOS E INVESTIMENTOS
    (df['TICKER'] == 'G2DI33') | (df['TICKER'] == 'GPIV33') | (df['TICKER'] == 'PPLA11'),

    #HOLDINGS DIVERSIFICADAS
    (df['TICKER'] == 'BLUT3') | (df['TICKER'] == 'BLUT4') | (df['TICKER'] == 'EPAR3') |	
    (df['TICKER'] == 'MOAR3') | (df['TICKER'] == 'PEAB3') | (df['TICKER'] == 'PEAB4') |
    (df['TICKER'] == 'SIMH3') | (df['TICKER'] == 'MAPT3'),

    #HOTEIS E RESTAURANTES
    (df['TICKER'] == 'HOOT4') | (df['TICKER'] == 'MEAL3') | (df['TICKER'] == 'ZAMP3'),	

    #INTERMEDIAÇAO IMOBILIARIA
    (df['TICKER'] == 'BBRK3') | (df['TICKER'] == 'LPSB3'),

    #MADEIRAS E PAPEL
    (df['TICKER'] == 'DXCO3') | (df['TICKER'] == 'EUCA3')	| (df['TICKER'] == 'EUCA4') |
    (df['TICKER'] == 'KLBN11')| (df['TICKER'] == 'KLBN3') | (df['TICKER'] == 'KLBN4') |
    (df['TICKER'] == 'MSPA3') | (df['TICKER'] == 'MSPA4') | (df['TICKER'] == 'RANI3') |
    (df['TICKER'] == 'SUZB3'),

    #MAQUINAS E EQUIPAMENTOS
    (df['TICKER'] == 'AERI3') | (df['TICKER'] == 'BDLL3') | (df['TICKER'] == 'BDLL4') | 
    (df['TICKER'] == 'EALT3') |	(df['TICKER'] == 'EALT4') | (df['TICKER'] == 'FRIO3') | 
    (df['TICKER'] == 'INEP3') | (df['TICKER'] == 'INEP4') | (df['TICKER'] == 'KEPL3') | 
    (df['TICKER'] == 'MTSA3') | (df['TICKER'] == 'MTSA4') | (df['TICKER'] == 'NORD3') | 
    (df['TICKER'] == 'ROMI3') | (df['TICKER'] == 'SHUL3') | (df['TICKER'] == 'SHUL4') | 
    (df['TICKER'] == 'WEGE3'),

    #MATERIAL AERONAUTICO
    (df['TICKER'] == 'EMBR3') | (df['TICKER'] == 'EMBJ3'), 

    #MATERIAL RODOVIARIO
    (df['TICKER'] == 'FRAS3') | (df['TICKER'] == 'MWET3')	| (df['TICKER'] == 'MWET4') |	
    (df['TICKER'] == 'POMO3') | (df['TICKER'] == 'POMO4') | (df['TICKER'] == 'RAPT3') |
    (df['TICKER'] == 'RAPT4') | (df['TICKER'] == 'RCSL3') | (df['TICKER'] == 'RCSL4') |
    (df['TICKER'] == 'RSUL4') | (df['TICKER'] =='TUPY3'),

    #MEDICAMENTOS E OUTROS PRODUTOS
    (df['TICKER'] == 'BIOM3') | (df['TICKER'] == 'BLAU3') | (df['TICKER'] == 'DMVF3') |
    (df['TICKER'] == 'HYPE3') | (df['TICKER'] == 'OFSA3') | (df['TICKER'] == 'PFRM3') |
    (df['TICKER'] == 'PGMN3') | (df['TICKER'] == 'PNVL3')	| (df['TICKER'] == 'RADL3') |
    (df['TICKER'] == 'VVEO3'),

    #MINERACAO
    (df['TICKER'] == 'AURA33')| (df['TICKER'] == 'BRAP3') | (df['TICKER'] == 'BRAP4') |
    (df['TICKER'] == 'CBAV3') | (df['TICKER'] == 'CMIN3') | (df['TICKER'] == 'MMXM3') |
    (df['TICKER'] == 'VALE3'),

    #OUTROS
    (df['TICKER'] == 'ATOM3') | (df['TICKER'] == 'BALM3') | (df['TICKER'] == 'BALM4') |
    (df['TICKER'] == 'FIGE3') | (df['TICKER'] == 'HETA3') | (df['TICKER'] == 'HETA4') | 
    (df['TICKER'] == 'MAPT4') | (df['TICKER'] == 'MTIG4'),

    #PETROLEO, GAS E BIOCOMBUSTIVEL
    (df['TICKER'] == 'AZTE3') | (df['TICKER'] == 'BRAV3') | (df['TICKER'] == 'CSAN3') | 
    (df['TICKER'] == 'DMMO3') | (df['TICKER'] == 'DMMO11') | (df['TICKER'] == 'ENAT3') | 
    (df['TICKER'] == 'PETR3') | (df['TICKER'] == 'PETR4') | (df['TICKER'] == 'PRIO3') | 
    (df['TICKER'] == 'RAIZ4') | (df['TICKER'] == 'RECV3') | (df['TICKER'] == 'RPMG3') | 
    (df['TICKER'] == 'RRRP3') | (df['TICKER'] == 'UGPA3') | (df['TICKER'] == 'VBBR3'),

    #PRODUTOS DE LIMPEZA
    (df['TICKER'] == 'BOBR3') | (df['TICKER'] == 'BOBR4'),

    #PRODUTOS DE USO PESSOAL
    (df['TICKER'] == 'ESPA3') | (df['TICKER'] == 'NTCO3') | (df['TICKER'] == 'NATU3'),	
    
    #PROGRAMAS E SERVICOS
    (df['TICKER'] == 'BMOB3') | (df['TICKER'] == 'CASH3') | (df['TICKER'] == 'CLSA3') |	
    (df['TICKER'] == 'DOTZ3') | (df['TICKER'] == 'ENJU3') | (df['TICKER'] == 'IFCM3') |
    (df['TICKER'] == 'LINX3') | (df['TICKER'] == 'LVTC3') | (df['TICKER'] == 'LWSA3') |
    (df['TICKER'] == 'NGRD3') | (df['TICKER'] == 'NINJ3') | (df['TICKER'] == 'OBTC3') |
    (df['TICKER'] == 'REAG3') | (df['TICKER'] == 'SQIA3') | (df['TICKER'] == 'TOTS3') | 
    (df['TICKER'] == 'TRAD3'),

    #QUIMICOS
    (df['TICKER'] == 'BRKM3') | (df['TICKER'] == 'BRKM5') | (df['TICKER'] == 'BRKM6') |
    (df['TICKER'] == 'CRPG3') | (df['TICKER'] == 'CRPG5') | (df['TICKER'] == 'CRPG6') |
    (df['TICKER'] == 'DEXP3') | (df['TICKER'] == 'DEXP4') | (df['TICKER'] == 'FHER3') |
    (df['TICKER'] == 'NUTR3') | (df['TICKER'] == 'UNIP3') | (df['TICKER'] == 'UNIP5') |
    (df['TICKER'] == 'UNIP6') | (df['TICKER'] == 'VITT3'),

    #SAUDE 
    (df['TICKER'] == 'AALR3') | (df['TICKER'] == 'DASA3') | (df['TICKER'] == 'FLRY3') |
    (df['TICKER'] == 'HAPV3') | (df['TICKER'] == 'KRSA3') | (df['TICKER'] == 'MATD3') | 
    (df['TICKER'] == 'ODPV3') | (df['TICKER'] == 'ONCO3') | (df['TICKER'] == 'PARD3') | 
    (df['TICKER'] == 'QUAL3') | (df['TICKER'] == 'RDOR3'),

    #SEGUROS
    (df['TICKER'] == 'APER3') | (df['TICKER'] == 'BBSE3') | (df['TICKER'] == 'BRGE11')|
    (df['TICKER'] == 'BRGE12')| (df['TICKER'] == 'BRGE3') | (df['TICKER'] == 'BRGE5') |
    (df['TICKER'] == 'BRGE6') | (df['TICKER'] == 'BRGE7') | (df['TICKER'] == 'BRGE8') |
    (df['TICKER'] == 'CSAB3') | (df['TICKER'] == 'CSAB4') | (df['TICKER'] == 'CXSE3') | 
    (df['TICKER'] == 'IRBR3') | (df['TICKER'] == 'PSSA3') | (df['TICKER'] == 'SULA11')| 
    (df['TICKER'] == 'SULA3') | (df['TICKER'] == 'SULA4') | (df['TICKER'] == 'WIZC3'),

    #SERVIÇOS DE APOIO E ARMAZENAGEM
    (df['TICKER'] == 'PORT3') | (df['TICKER'] == 'STBP3'),

    #SERVIÇOS DIVERSOS
    (df['TICKER'] == 'ALPK3') | (df['TICKER'] == 'ATMP3') | (df['TICKER'] == 'CARD3') |
    (df['TICKER'] == 'CSUD3') | (df['TICKER'] == 'CTAX3') | (df['TICKER'] == 'DTCY3') |
    (df['TICKER'] == 'ELMD3') | (df['TICKER'] == 'GGPS3') | (df['TICKER'] == 'PRNR3') | 
    (df['TICKER'] == 'SEQL3') | (df['TICKER'] == 'SMLS3') | (df['TICKER'] == 'SNSY3') | 
    (df['TICKER'] == 'SNSY5') | (df['TICKER'] == 'SNSY6') | (df['TICKER'] == 'UCAS3') | 
    (df['TICKER'] == 'VLID3') | (df['TICKER'] == 'WLMM3') | (df['TICKER'] == 'WLMM4'),

    #SERVIÇOS EDUCACIONAIS
    (df['TICKER'] == 'ANIM3') | (df['TICKER'] == 'BAHI3') | (df['TICKER'] == 'COGN3') |
    (df['TICKER'] == 'CSED3') | (df['TICKER'] == 'SEER3') | (df['TICKER'] == 'VTRU3') |
    (df['TICKER'] == 'YDUQ3'),
    
    #SERVIÇOS FINANCEIROS
    (df['TICKER'] == 'B3SA3') | (df['TICKER'] == 'BOAS3') | (df['TICKER'] == 'CIEL3') |
    (df['TICKER'] == 'GETT11')| (df['TICKER'] == 'GETT3') | (df['TICKER'] == 'GETT4') |	
    (df['TICKER'] == 'PDTC3'),

    #SOC. CREDITO E FINANCIAMENTO
    (df['TICKER'] == 'CRIV3') | (df['TICKER'] == 'CRIV4') | (df['TICKER'] == 'FNCN3') |
    (df['TICKER'] == 'MERC3') | (df['TICKER'] == 'MERC4'),

    #SIDERURGIA E METALURGIA
    (df['TICKER'] == 'CSNA3') | (df['TICKER'] == 'FESA3') | (df['TICKER'] == 'FESA4') |
    (df['TICKER'] == 'GGBR3') | (df['TICKER'] == 'GGBR4') | (df['TICKER'] == 'GOAU3') |
    (df['TICKER'] == 'GOAU4') | (df['TICKER'] == 'MGEL3') | (df['TICKER'] == 'MGEL4') | 
    (df['TICKER'] == 'PATI3') | (df['TICKER'] == 'PATI4') | (df['TICKER'] == 'PMAM3') | 
    (df['TICKER'] == 'TKNO3') | (df['TICKER'] == 'TKNO4') | (df['TICKER'] == 'USIM3') | 
    (df['TICKER'] == 'USIM5') | (df['TICKER'] == 'USIM6'),
    
    #SUPERMERCADO
    (df['TICKER'] == 'ASAI3') | (df['TICKER'] == 'CRFB3') | (df['TICKER'] == 'GMAT3') |
    (df['TICKER'] == 'PCAR3'),

    #TELECOMUNICAÇOES
    (df['TICKER'] == 'BRIT3') | (df['TICKER'] == 'BRST3') | (df['TICKER'] == 'DESK3') | 
    (df['TICKER'] == 'FIQE3') | (df['TICKER'] == 'OIBR3') | (df['TICKER'] == 'OIBR4') | 
    (df['TICKER'] == 'TELB3') | (df['TICKER'] == 'TELB4') | (df['TICKER'] == 'TIMS3') |	
    (df['TICKER'] == 'VIVT3'),

    #TRANSPORTE FERROVIARIO
    (df['TICKER'] == 'MRSA3B') | (df['TICKER'] == 'MRSA5B') | (df['TICKER'] == 'MRSA6B') |
    (df['TICKER'] == 'RAIL3') | (df['TICKER'] == 'VSPT3'),	
    
    #TRANSPORTE HIDROVIARIO
    (df['TICKER'] == 'HBSA3') | (df['TICKER'] == 'LOGN3')	| (df['TICKER'] == 'LUXM4') |
    (df['TICKER'] == 'OPCT3'),

    #TRANSPORTE RODOVIARIO
    (df['TICKER'] == 'JSLG3') | (df['TICKER'] == 'TGMA3'),

    #VAREJO
    (df['TICKER'] == 'ALLD3') | (df['TICKER'] == 'ALPA3') | (df['TICKER'] == 'ALPA4') |
    (df['TICKER'] == 'AMAR3') | (df['TICKER'] == 'AMER3') | (df['TICKER'] == 'ARZZ3') | 
    (df['TICKER'] == 'AZZA3') | (df['TICKER'] == 'BHIA3') | (df['TICKER'] == 'CAMB3') | 
    (df['TICKER'] == 'CEAB3') | (df['TICKER'] == 'CGRA3') | (df['TICKER'] == 'CGRA4') | 
    (df['TICKER'] == 'GRND3') | (df['TICKER'] == 'GUAR3') | (df['TICKER'] == 'LJQQ3') | 
    (df['TICKER'] == 'VSTE3') | (df['TICKER'] == 'LREN3') | (df['TICKER'] == 'MBLY3') | 
    (df['TICKER'] == 'MGLU3') | (df['TICKER'] == 'MNDL3') | (df['TICKER'] == 'PETZ3') | 
    (df['TICKER'] == 'SBFG3') | (df['TICKER'] == 'SLED3') | (df['TICKER'] == 'SLED4') | 
    (df['TICKER'] == 'SOMA3') | (df['TICKER'] == 'TECN3') | (df['TICKER'] == 'TFCO4') | 
    (df['TICKER'] == 'TOKY3') | (df['TICKER'] == 'VIIA3') | (df['TICKER'] == 'VIVA3') | 
    (df['TICKER'] == 'VULC3') | (df['TICKER'] == 'WEST3') | (df['TICKER'] == 'WHRL3') | 
    (df['TICKER'] == 'WHRL4'),
    
    #VIAGEM E LAZER
    (df['TICKER'] == 'AHEB3') | (df['TICKER'] == 'AHEB5') | (df['TICKER'] == 'AHEB6') |
    (df['TICKER'] == 'BMKS3') | (df['TICKER'] == 'CVCB3') | (df['TICKER'] == 'ESTR3') |
    (df['TICKER'] == 'ESTR4') | (df['TICKER'] == 'SHOW3') | (df['TICKER'] == 'SMFT3') 

]

  choices = [
           'AGUA E SANEAMENTO', 
           'AGRICULTURA',
           'ALIMENTOS DIVERSOS',
           'ALUGUEL DE CARRO',
           'ALUGUEL DE MAQ. E EQUIP.',
           'ARMAS E MUNICAO',
           'AUTOMOVEIS E MOTOCICLETAS',
           'AVIACAO',
           'AÇUCAR E ALCOOL',
           'BANCOS',
           'BEBIDAS',
           'CARNES E DERIVADOS',
           'COMPUTADORES E EQUIPAMENTOS',
           'CONSTRUÇAO CIVIL',
           'CONSTRUÇAO E ENGENHARIA',
           'ENERGIA ELETRICA',
           'EQUIPAMENTOS E SERVIÇOS',
           'EXPLORAÇAO DE IMOVEIS',
           'EXPLORAÇAO DE RODOVIAS',
           'FIOS E TECIDOS',
           'GAS',
           'GESTAO DE RECURSOS E INVESTIMENTOS',
           'HOLDINGS DIVERSIFICADAS',
           'HOTEIS E RESTAURANTES',
           'INTERMEDIAÇAO IMOBILIARIA',
           'MADEIRAS E PAPEL',
           'MAQUINAS E EQUIPAMENTOS',
           'MATERIAL AERONAUTICO',
           'MATERIAL RODOVIARIO',
           'MEDICAMENTOS E OUTROS PRODUTOS',
           'MINERACAO',
           'OUTROS',
           'PETROLEO, GAS E BIOCOMBUSTIVEL',
           'PRODUTOS DE LIMPEZA',
           'PRODUTOS DE USO PESSOAL',
           'PROGRAMAS E SERVICOS',
           'QUIMICOS',
           'SAUDE',
           'SEGUROS',
           'SERVIÇOS DE APOIO E ARMAZENAGEM',
           'SERVIÇOS DIVERSOS',
           'SERVIÇOS EDUCACIONAIS',
           'SERVIÇOS FINANCEIROS',
           'SOC. CREDITO E FINANCIAMENTO',
           'SIDERURGIA E METALURGIA',
           'SUPERMERCADO',
           'TELECOMUNICAÇOES',
           'TRANSPORTE FERROVIARIO',
           'TRANSPORTE HIDROVIARIO',
           'TRANSPORTE RODOVIARIO',
           'VAREJO',
           'VIAGEM E LAZER'
  ]

  df['SEGMENTO'] = np.select(conditions, choices, default='NA')

  # Algumas colunas possuem números que tem o '.' como separador de milhar
  # Primeiro, eu tenho que remover primeiro o '.' por '' depois trocar as ',' por '.' 
  # Depois transformar a coluna em float
  # 'regex=False' significa que 
  df['PRECO'] = df.loc[:, 'PRECO'].str.replace('.','', regex=False).str.replace(',','.', regex=False).astype(float)
  df['DY'] = df.loc[:, 'DY'].str.replace(',','.', regex=False).astype(float)
  df['P/L'] = df.loc[:, 'P/L'].str.replace('.','', regex=False).str.replace(',','.', regex=False).astype(float)
  df['P/VP'] = df.loc[:, 'P/VP'].str.replace('.','', regex=False).str.replace(',','.', regex=False).astype(float)
  df['MARGEM BRUTA'] = df.loc[:, 'MARGEM BRUTA'].str.replace('.','', regex=False).str.replace(',','.', regex=False).astype(float)
  df['MARG. LIQUIDA'] = df.loc[:, 'MARG. LIQUIDA'].str.replace('.','', regex=False).str.replace(',','.', regex=False).astype(float)
  df['EV/EBIT'] = df.loc[:, 'EV/EBIT'].str.replace('.','', regex=False).str.replace(',','.', regex=False).astype(float)
  df['DIVIDA LIQUIDA / EBIT'] = df.loc[:, 'DIVIDA LIQUIDA / EBIT'].str.replace('.','', regex=False).str.replace(',','.', regex=False).astype(float)
  df['DIV. LIQ. / PATRI.'] = df.loc[:, 'DIV. LIQ. / PATRI.'].str.replace('.','', regex=False).str.replace(',', '.', regex=False).astype(float)
  df['PSR'] = df.loc[:, 'PSR'].str.replace('.','', regex=False).str.replace(',','.', regex=False).astype(float)
  df['ROE'] = df.loc[:, 'ROE'].str.replace('.','', regex=False).str.replace(',','.', regex=False).astype(float)
  df['ROIC'] = df.loc[:, 'ROIC'].str.replace('.','', regex=False).str.replace(',','.', regex=False).astype(float)
  df['LIQUIDEZ MEDIA DIARIA'] = df.loc[:, 'LIQUIDEZ MEDIA DIARIA'].str.replace('.','', regex=False).str.replace(',','.', regex=False).astype(float)
  df['LPA'] = df.loc[:, 'LPA'].str.replace('.','', regex=False).str.replace(',','.', regex=False).astype(float)
  df['VALOR DE MERCADO'] = df.loc[:, 'VALOR DE MERCADO'].str.replace('.','', regex=False).str.replace(',','.', regex=False).astype(float)

  # Criando uma coluna nova do indicador 'Earning Yields'
  df['EY'] = round((df['LPA'] / df['PRECO']) * 100, 2)

  # Formatando os números de todas as colunas. Principalmente p/ as colunas 'LIQUIDEZ MEDIA DIARIA' e 'VALOR DE MERCADO	' que estavam em notação científica
  pd.set_option('float_format', '{:,.2f}'.format)

  return df


def df_rank_ey(df: pd.DataFrame) -> pd.DataFrame:
  """
  Função que faz o rank do indicador 'Earning Yield' das empresas.

  Parameters
  ----------
  df: pd.DataFrame
        DataFrame do Statusinvest.
  
  Returns
  -------
  pd.DataFrame
        DataFrame com o rank do indicador 'Earning Yield'.
  """
  # Lista de empresas que estão em recuperação judicial.
  lista_rec_jud = ['RNEW11', 'RNEW3', 'RNEW4',   
                   'OIBR3', 'OIBR4', 
                   'SLED3', 'SLED4',
                   'BDLL3', 'BDLL4', 
                   'INEP3', 'INEP4',
                   'TCNO3', 'TCNO4',
                   'TEKA3', 'TEKA4',
                   'MWET3', 'MWET4',
                   'IGBR3', 
                   'ETER3', 
                   'FHER3',
                   'HOOT4',
                   'JFEN3',
                   'LUPA3',
                   'FRTA3',
                   'RPMG3',
                   'VIVR3',
  ]
  
  # Criando um rank de EY
  rank_ey = df.loc[:, ['TICKER', 'EY']].sort_values(by='EY', ascending=False)

  # Retirando as empresas que estão em recuperação judicial do rank. O '~' retorna o df sem as empresas em recuperação judicial
  # Top 20 do rank
  rank_ey = rank_ey[~np.isin(rank_ey['TICKER'], lista_rec_jud)][:20]

  return rank_ey


def pl_ibov(df: pd.DataFrame, df_ibovespa: pd.DataFrame) -> pd.DataFrame:
  """
  Função que calcula o PL médio do IBOV.

  Parameters
  ----------
  df_ibovespa: pd.DataFrame
    DataFrame do ibovespa.
  df: pd.DataFrame 
    DataFrame do Statusinvest.

  Returns
  -------
  pl_ibov: pd.DataFrame
    DataFrame do indicador 'P/L' médio do Ibovespa.
  """
  #Criando uma lista do IBOV
  lista_ibov = df_ibovespa['Código'].tolist()
  
  # Criando um novo df que só tem as ações que compõem o IBOV
  df_just_ibov = df[df['TICKER'].isin(lista_ibov)]

  # Selecionando a coluna 'P/L'
  df_just_ibov = df_just_ibov.loc[:, ['TICKER', 'P/L']]

  # Fazendo uma cópia do df é da primeira parte deste notebook ('Arquivos IBOV e SMALL11')
  df_ibov2 = df_ibovespa.copy() 

  # Renomeando a coluna 'Código' para 'TICKER'
  df_ibov2.rename({'Código':'TICKER'}, axis=1, inplace=True)

  # Selecionando as colunas que eu preciso para o merge
  df_ibov2 = df_ibov2.loc[:, ['TICKER', 'Part. (%)']]

  # Realizando o merge com os dois dfs
  df_ibov_pl = pd.merge(df_just_ibov, df_ibov2, on='TICKER')

  # Caso eu queira tirar as empresa com P/L negativo
  df_ibov_pl = df_ibov_pl[df_ibov_pl['P/L'] > 0]

  # Calculando o P/L do IBOV
  pl_ibov = np.average(df_ibov_pl['P/L'], weights=df_ibov_pl['Part. (%)'])
  pl_ibov = round(pl_ibov, 2)

  #print(f'O P/L médio do IBOV é de {pl_ibov}')

  return pl_ibov


def vol_anual(df_setor: pd.DataFrame, ticker: str, ano: str) -> pd.Series:
    """
    Função que calcula a volatilidade anualizada.

    Parameters
    ----------
    df_setor: pd.DataFrame
        DataFrame do setor.
    ticker: str 
        Ticker da empresa.
    ano: str
        Período escolhido.

    Returns
    -------
    annualized_volatility: pd.Series
        Volatilidade anualiazada.

    NOTE: para calcular a vol mensal trocar apenas o np.sqrt(12) e a vol semanal trocar apenas o np.sqrt(52).
    """
    lst_annualized_volatility = []

    for ticker in df_setor.columns:
        # Calculando o retorno logarítmico
        log_return = np.log(df_setor.loc[ano, ticker] / df_setor.loc[ano, ticker].shift(1))

        # Calculando a volatilidade anualizada
        annualized_volatility = np.std(log_return) * np.sqrt(252)

        # Adicionando na lista
        lst_annualized_volatility.append(annualized_volatility)
    
    # Criando o df
    df_annualized_volatility = pd.DataFrame(lst_annualized_volatility, index=df_setor.columns).rename(columns={0: 'vol_anual'})

    return df_annualized_volatility


def drawdown(df_returns: pd.DataFrame) -> pd.Series:
    """
    Função que calcula drawdown.

    Parameters
    ----------
    df_returns: pd.DataFrame
        DataFrame do retorno diário do ativo.

    Returns
    -------
    pd.Series
        Ponto mínimo do drawdown.
    """
    # Calculando o retorno acumulado
    cumulative_returns = (1+df_returns).cumprod()

    # Calculando o pico
    peak = cumulative_returns.expanding(min_periods=1).max()

    # Calculando o drawdown
    drawdown = (cumulative_returns / peak) - 1

    return drawdown.min()


def cluster_corr(corr_array: pd.DataFrame, inplace=False) -> pd.DataFrame:
    """
    Reorganiza a matriz de correlação ('corr_array') para que os grupos que possuem
    alta correlação estejam próximas entre si.

    Parameters
    ----------
    corr_array: pd.DataFrame
        DataFrame de uma matriz de correlação (NxN).
        
    Returns
    -------
    corr_array: pd.DataFrame
        DataFrame de uma matriz de correlação (NxN) reorganizada.

    NOTE: https://wil.yegelwel.com/cluster-correlation-matrix/
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    
    if not inplace:
        corr_array = corr_array.copy()
    
    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
        
    return corr_array[idx, :][:, idx]


def cagr(start_value: float, end_value: float, num_periods: int) -> float:
    """
    Função que calcula o CAGR.

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
    float
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


def maior_streak(ticker: str, start: str) -> pd.DataFrame:
    """
    Mostra a maior sequência tanto de alta quanto de baixa de um ativo.

    Parameters
    ----------
    ticker: str
        Ticker da empresa.
    start: str
        Data de início do DataFrame dos dados da empresa.

    Returns
    -------
    df_maior_sequencia_alta: pd.DataFrame
        DataFrame coma a maior sequência de alta.
    df_maior_sequencia_baixa: pd.DataFrame
        DataFrame coma a maior sequência de baixa.
    """
    df = yf.download(ticker, start, auto_adjust=True)
    df['returns'] = df['Close'].pct_change()

    # Criando uma coluna que indica o início de cada sequência de alta
    df['inicio_sequencia'] = (df['returns'] > 0) & (df['returns'].shift() <= 0)

    # Atribuindo um número de sequência para cada grupo consecutivo de alta
    df['sequencia_alta'] = df['inicio_sequencia'].cumsum().where(df['returns'] > 0)

    # Calculando o tamanho da maior sequência de alta
    maior_sequencia_alta = df.groupby('sequencia_alta').size().max()

    # Filtrando o DataFrame para mostrar apenas a parte da maior sequência de alta
    filt_alta = df['sequencia_alta'] == df['sequencia_alta'].value_counts().idxmax()
    df_maior_sequencia_alta = df.loc[filt_alta, ['Close', 'returns']]

    # Criando uma coluna que indica o início de cada sequência de baixa
    df['inicio_sequencia'] = (df['returns'] < 0) & (df['returns'].shift() >= 0)

    # Atribuindo um número de sequência para cada grupo consecutivo de baixa
    df['sequencia_baixa'] = df['inicio_sequencia'].cumsum().where(df['returns'] < 0)

    # Calculando o tamanho da maior sequência de baixa
    maior_sequencia_baixa = df.groupby('sequencia_baixa').size().max()

    # Filtrando o dataframe para mostrar apenas a parte da maior sequência de baixa
    filt_baixa = df['sequencia_baixa'] == df['sequencia_baixa'].value_counts().idxmax()
    df_maior_sequencia_baixa = df.loc[filt_baixa, ['Close', 'returns']]

    print('-'*50)
    print(f'Sobre o ativo ({ticker}): ')
    print(f'Sua maior sequência é de {maior_sequencia_alta} dias seguidos de alta.')
    print(f'Sua maior sequência é de {maior_sequencia_baixa} dias seguidos de baixa.')
    print('-'*50)

    return df_maior_sequencia_alta, df_maior_sequencia_baixa


def get_data(stocks: str, start: str, end: str) -> pd.Series:
    """
    Calcula a média dos retornos e a covariância.

    Parameters
    ----------
    stocks: str
        DataFrame das ações.
    start: str
        Data de início.
    end: str
        Data final.

    Returns
    -------
    pd.Series
        Média dos retornos e a covariância das ações.
    """
    stockData = yf.download(stocks, start, end, auto_adjust=True)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()

    return meanReturns, covMatrix


def mcVaR(returns: pd.Series, alpha=5) -> float:
    """
    Calcula o VaR.

    Parameters
    ----------
    returns: pd.Series
        Séries pandas dos retornos.

    Returns
    -------
    float
        Percentil da distribuição de retornos para um determinado nível de confiança alfa.
    """
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    else:
        raise TypeError("Expected a pandas data series")
    

def mcCVaR(returns, alpha=5) -> float:
    """
    Calcula o CVaR.

    Parameters
    ----------
    returns: pd.Series
        Séries pandas dos retornos.

    Returns
    -------
    float
        CVaR ou Expected Shortfall para um determinado nível de confiança alfa.
    """
    if isinstance(returns, pd.Series):
        belowVaR = returns <= mcVaR(returns, alpha=alpha)
        return returns[belowVaR].mean()
    else:
        raise TypeError("Expected a pandas data series")
    

def corr_matrix(df: pd.DataFrame, start: int, end: int) -> pd.Series:
    """
    Calcula a matriz de correlação.
    
    Parameters
    ----------
    df: pd.DataFrame
        DataFrame dos retornos logarítmicos ativos.
    start: int
        Nº inícial.
    end: int
        Nº final.

    Returns
    -------
    corr_seq: pd.Series
        Matriz de correlação.
    """
    seq = df.iloc[start:end, :]
    corr_seq = seq.corr().values
    return corr_seq


def structural_entropy(carteira_logret: pd.DataFrame, sequence_length: int, t: int) -> pd.DataFrame:
    """
    Calcula a entropia estrutural dos ativos.

    Parameters
    ----------
    carteira_logret: pd.DataFrame
        DataFrame dos retornos logarítmicos dos ativos.
    sequence_length: int
        Número do comprimento da janela.
    t: int
        Limites (thresholds). 

    Returns
    -------
    structural_entropy_df: pd.DataFrame
        DataFrame da entropia estrutural.
    """
    structural_entropy = {'Date': [], 'structural_entropy': []}

    for d in tqdm(range(sequence_length, carteira_logret.shape[0])):

        _corr = corr_matrix(carteira_logret, d-sequence_length, d)

        _corr = (np.abs(_corr) > t).astype(int)
        _, _labels = connected_components(_corr)

        _, _count = np.unique(_labels, return_counts=True)
        _countnorm = _count / _count.sum()
        _entropy = -(_countnorm * np.log2(_countnorm)).sum()

        structural_entropy['Date'].append(carteira_logret.index[d])
        structural_entropy['structural_entropy'].append(_entropy)

    structural_entropy_df = pd.DataFrame(structural_entropy)
    structural_entropy_df = structural_entropy_df.set_index('Date')
    
    return structural_entropy_df


def calculate_snr(signal: pd.Series) -> float:
    """
    Calcula o SNR.

    Parameters
    ----------
    signal: pd.Series
        Preços de fechamento ajustados dos ativos.

    Returns
    -------
    snr: float
        Relação sinal-ruído.
    """
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean((signal - np.mean(signal)) ** 2)
    snr = 10 * np.log10(signal_power/noise_power)

    return snr


def rolling_snr(signal: pd.Series, window: int) -> np.array:
    """
    Calcula o SNR móvel.

    Parameters
    ----------
    signal: pd.Series
        Preços de fechamento ajustados dos ativos.
    window: int
        Nº da janela móvel.

    Returns
    -------
    snr: np.array
        Relação sinal-ruído móvel.
    """
    snr_values = []
    for i in range(len(signal) - window + 1):
        window_signal = signal[i:i+window]
        snr = calculate_snr(window_signal)
        snr_values.append(snr)

    # Preenchendo com NaNs para alinhar os índices
    return np.concatenate((np.full(window-1, np.nan), np.array(snr_values)))


def plot_snr_adj_close(
        ticker: str,
        dates: pd.Series, 
        close: pd.Series, 
        snr_values: pd.Series, 
        window:int
    ):
    """
    Parameters
    ----------
    ticker: str
        Ticker do ativo.
    dates: pd.Series 
        Data da série temporal (index).
    close: pd.Series 
        Preço de fechamento do ativo.
    snr_values: pd.Series
        Valores do SNR.
    window: int
        Nº da janela móvel.

    Returns
    -------
    np.array
        Plot do preço de fechamento e SNR móvel do ativo.
    """
    fig = make_subplots(
        rows=2, 
        cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.1, 
        subplot_titles=('Close', 'SNR')
    )

    fig.add_trace(go.Scatter(
        x=dates,
        y=close, 
        mode='lines',
        name='Close',
        line=dict(color='blue')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dates,
        y=snr_values, 
        mode='lines',
        name='SNR',
        line=dict(color='red')
    ), row=2, col=1)

    fig.update_layout(
        title=ticker + ' - Close e Rolling SNR (' + str(window) + 'dias)',
        height=900,
        width=900
    )

    fig.add_hline(y=snr_values.mean(), row=2, col=1)

    fig.update_xaxes(title_text='Data', row=2, col=1)
    fig.update_yaxes(title_text='Close', row=1, col=1)
    fig.update_yaxes(title_text='SNR (dB)', row=2, col=1)

    return fig.show()


def plot_snr_vs_returns(snr_values: pd.Series, returns: pd.Series):
    """
    Parameters
    ----------
    snr_values: pd.Series
        Valores do SNR.
    returns: pd.Series
        Valores da variação percentual do ativo.

    Returns
    -------
    Plot do entre o SNR e o desvio-padrão.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=snr_values,
        y=returns,
        mode='markers',
        name='SNR vs Returns',
        marker=dict(color='green')
    ))

    fig.update_layout(
        title='Scatter Plot: SNR vs STD',
        height=600,
        width=900,
        xaxis_title='SNR (dB)',
        yaxis_title='STD'
    )

    return fig.show()


def create_trading_signals(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Parameters
    ----------
    df: pd.DataFrame
        DataFrame do ativo.
    window: int
        Nº da janela móvel.

    Returns
    -------
    pd.DataFrame
        DataFrame que contém os sinais de trading.

    NOTE: você pode mudar os parâmetros de threshold de compra (1.5) e venda (0.9).
    """
    df['SNR'] = rolling_snr(signal=df['Close'].values, window=window)
    df['Signal'] = 0 # 0: hold, 1: buy, -1: sell

    # Regras de trading: comprar quando o SNR < threshold, vender quando o SNR > threshold
    threshold_buy = df['SNR'].min() * 1.25
    threshold_sell = df['SNR'].max() * 0.9

    df.loc[df['SNR'].rolling(10).mean() < threshold_buy, 'Signal'] = 1
    df.loc[df['SNR'].rolling(10).mean() > threshold_sell, 'Signal'] = -1

    return df


def calculate_rsi(df: pd.Series, window: int) -> pd.Series:
    """
    Calcula o índice de força relativa (RSI).

    Parameters
    ----------
    df: pd.Series
        Série dos preços de fechamento do ativo.
    window: int
        Janela para o cálculo do RSI.

    Returns
    -------
    rsi: pd.Series
        RSI calculado.
    """
    delta = df.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def detect_rsi_divergence(df: pd.Series, rsi: pd.Series, lookback: int) -> pd.DataFrame:
    """
    Detecta divergências entre o preço e o RSI.

    Parameters
    ----------
    df: pd.Series
        Série dos preços de fechamento do ativo.
    rsi: pd.Series
        RSI calculado.
    lookback: int
        Número de períodos para verificar divergências.

    Returns
    -------
    df_divergence: pd.DataFrame
        DataFrame com divergências detectadas.
    """
    # Inicializa um dicionário para armazenar as divergências detectadas
    divergence_dict = {'Date': [], 'Type': [], 'Price': [], 'RSI': []}

    for i in range(lookback, len(df)):
            # fatiamento com iloc
            window_price = df.iloc[i-lookback:i]
            
            price_max = window_price.max()
            price_min = window_price.min()

            # idxmax() retorna o rótulo (data), então usamos rsi[data] ou rsi.loc[data]
            rsi_max = rsi.loc[window_price.idxmax()]
            rsi_min = rsi.loc[window_price.idxmin()]

            current_price = df.iloc[i]
            current_rsi = rsi.iloc[i]

            # Lógica de Divergência de Alta (Bullish)
            if current_price < price_min and current_rsi > rsi_min:
                divergence_dict['Date'].append(df.index[i])
                divergence_dict['Type'].append('Bullish')
                divergence_dict['Price'].append(current_price)
                divergence_dict['RSI'].append(current_rsi)

            # Lógica de Divergência de Baixa (Bearish)
            if current_price > price_max and current_rsi < rsi_max:
                divergence_dict['Date'].append(df.index[i])
                divergence_dict['Type'].append('Bearish')
                divergence_dict['Price'].append(current_price)
                divergence_dict['RSI'].append(current_rsi)

    # Converte o dicionário de divergência em um df
    df_divergence = pd.DataFrame(divergence_dict)

    return df_divergence


def plot_rsi_divergence(df: pd.Series, rsi: pd.Series, divergences: pd.DataFrame, symbol: str):
    """
    Plota o gráfico de preços e RSI com as divergências identificadas.

    Parameters
    ----------
    df:  pd.Series
        Série dos preços de fechamento do ativo.
    rsi: pd.Series
        RSI calculado.
    divergences: pd.DataFrame
        Divergências detectadas.
    symbol: str
        Ticker do ativo.
    """
    # Criar subplots: 2 linhas, 1 coluna (Preço em cima, RSI embaixo)
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        subplot_titles=(f'{symbol} - Preço e Divergências', 'RSI e Divergências'),
        row_heights=[0.7, 0.3]
    )

    # Gráfico de Preço (Linha)
    fig.add_trace(
        go.Scatter(x=df.index, y=df, name='Preço', line=dict(color='royalblue', width=1)),
        row=1, col=1
    )

    # Divergências no Preço (Scatter)
    bullish = divergences[divergences['Type'] == 'Bullish']
    bearish = divergences[divergences['Type'] == 'Bearish']

    fig.add_trace(
        go.Scatter(x=bullish['Date'], y=bullish['Price'], mode='markers',
                   name='Bullish Div (Preço)', marker=dict(color='green', size=8, symbol='triangle-up')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=bearish['Date'], y=bearish['Price'], mode='markers',
                   name='Bearish Div (Preço)', marker=dict(color='red', size=8, symbol='triangle-down')),
        row=1, col=1
    )

    # Gráfico de RSI (Linha)
    fig.add_trace(
        go.Scatter(x=rsi.index, y=rsi, name='RSI', line=dict(color='orange', width=1.5)),
        row=2, col=1
    )

    # Divergências no RSI (Scatter)
    fig.add_trace(
        go.Scatter(x=bullish['Date'], y=bullish['RSI'], mode='markers',
                   name='Bullish Div (RSI)', marker=dict(color='green', size=8, symbol='circle')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=bearish['Date'], y=bearish['RSI'], mode='markers',
                   name='Bearish Div (RSI)', marker=dict(color='red', size=8, symbol='circle')),
        row=2, col=1
    )

    # Linhas de Sobrecompra e Sobrevenda
    fig.add_hline(y=70, line_dash='dash', line_color='red', opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash='dash', line_color='green', opacity=0.5, row=2, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor='gray', opacity=0.1, line_width=0, row=2, col=1)

    # Ajustes de Layout
    fig.update_layout(
        height=800,
        title_text=f'Análise de Divergência RSI - {symbol}',
        template='plotly_dark', 
        hovermode='x unified',
        showlegend=True
    )

    fig.update_yaxes(title_text='Preço', row=1, col=1)
    fig.update_yaxes(title_text='RSI', range=[0, 100], row=2, col=1)

    fig.show()


# Funções análise fundamentalistas
def df_dre(path: str) -> pl.DataFrame:
    """
    Abre e filtra o arquivo compilado da DRE.

    Parameters
    ----------
    path: str
        Caminho do arquivo.

    Returns
    -------
    dre: pl.DataFrame
        DataFrame da DRE filtrada.
    """
    # Lendo o arquivo parquet
    dre = pl.read_parquet(path)

    # Selecionando as colunas
    dre = dre.select([
        'DT_REFER', 
        'DENOM_CIA', 
        'CD_CVM', 
        'ORDEM_EXERC', 
        'DT_INI_EXERC',
        'DT_FIM_EXERC',
        'CD_CONTA', 
        'DS_CONTA', 
        'VL_CONTA'
    ])

    # Transformando os dtypes das colunas
    dre = dre.with_columns([
        pl.col('DT_REFER').str.strptime(pl.Date, '%Y-%m-%d').alias('DT_REFER'),
        pl.col('DENOM_CIA').cast(pl.Utf8),
        pl.col('CD_CVM').cast(pl.Int32, strict=False),
        pl.col('ORDEM_EXERC').cast(pl.Utf8),
        pl.col('DT_INI_EXERC').str.strptime(pl.Date, '%Y-%m-%d').alias('DT_INI_EXERC'),
        pl.col('DT_FIM_EXERC').str.strptime(pl.Date, '%Y-%m-%d').alias('DT_FIM_EXERC'),
        pl.col('DS_CONTA').cast(pl.Utf8)
    ])

    # Transformando em datetime
    dre = dre.with_columns(
        pl.col('DT_REFER').cast(pl.Datetime).alias('DT_REFER'),
        pl.col('DT_INI_EXERC').cast(pl.Datetime).alias('DT_INI_EXERC'),
        pl.col('DT_FIM_EXERC').cast(pl.Datetime).alias('DT_FIM_EXERC'),
    )

    # Filtrando apenas 'ÚLTIMO' em ORDEM_EXERC
    dre = dre.filter(pl.col('ORDEM_EXERC') == 'ÚLTIMO')

    # Renomeando as colunas
    dre = dre.rename({col: col.lower() for col in dre.columns})

    return dre


def df_bp(path: str) -> pl.DataFrame:
    """
    Abre e filtra o arquivo compilado do BPA (ativo) e BPP (passivo).

    Parameters
    ----------
    path: str
        Caminho do arquivo.

    Returns
    -------
    bp: pl.DataFrame
        DataFrame do BPA e BPP filtrado.
    """
    # Lendo o arquivo parquet
    bp = pl.read_parquet(path)

    # Selecionando as colunas
    bp = bp.select([
        'DT_REFER', 
        'DENOM_CIA', 
        'CD_CVM', 
        'ORDEM_EXERC', 
        'CD_CONTA', 
        'DS_CONTA', 
        'VL_CONTA'
    ])

    # Transformando os dtypes das colunas
    bp = bp.with_columns([
        pl.col('DT_REFER').str.strptime(pl.Date, '%Y-%m-%d').alias('DT_REFER'),
        pl.col('DENOM_CIA').cast(pl.Utf8),
        pl.col('CD_CVM').cast(pl.Int32, strict=False),
        pl.col('ORDEM_EXERC').cast(pl.Utf8),
        pl.col('DS_CONTA').cast(pl.Utf8)
    ])

    # Transformando em datetime
    bp = bp.with_columns(
        pl.col('DT_REFER').cast(pl.Datetime).alias('DT_REFER')
    )

    # Filtrando apenas 'ÚLTIMO' em ORDEM_EXERC
    bp = bp.filter(pl.col('ORDEM_EXERC') == 'ÚLTIMO')

    # Renomeando as colunas
    bp = bp.rename({col: col.lower() for col in bp.columns})

    return bp


def df_dfc(path: str) -> pl.DataFrame:
    """
    Abre e filtra o arquivo compilado da DFC_MI (método indireto).

    Parameters
    ----------
    path: str
        Caminho do arquivo.

    Returns
    -------
    dfc: pl.DataFrame
        DataFrame da DFC filtrada.
    """
    # Lendo o arquivo parquet
    dfc = pl.read_parquet(path)

    # Selecionando as colunas
    dfc = dfc.select([
        'DT_REFER', 
        'DENOM_CIA', 
        'CD_CVM', 
        'ORDEM_EXERC', 
        'CD_CONTA', 
        'DS_CONTA', 
        'VL_CONTA'
    ])

    # Transformando os dtypes das colunas
    dfc = dfc.with_columns([
        pl.col('DT_REFER').str.strptime(pl.Date, '%Y-%m-%d').alias('DT_REFER'),
        pl.col('DENOM_CIA').cast(pl.Utf8),
        pl.col('CD_CVM').cast(pl.Int32, strict=False),
        pl.col('ORDEM_EXERC').cast(pl.Utf8),
        pl.col('DS_CONTA').cast(pl.Utf8)
    ])

    # Transformando em datetime
    dfc = dfc.with_columns(
        pl.col('DT_REFER').cast(pl.Datetime).alias('DT_REFER'),
    )

    # Filtrando apenas 'ÚLTIMO' em ORDEM_EXERC
    dfc = dfc.filter(pl.col('ORDEM_EXERC') == 'ÚLTIMO')

    # Renomeando as colunas
    dfc = dfc.rename({col: col.lower() for col in dfc.columns})

    return dfc


def df_dva(path: str) -> pl.DataFrame:
    """
    Abre e filtra o arquivo compilado da DVA.

    Parameters
    ----------
    path: str
        Caminho do arquivo.

    Returns
    -------
    dva: pl.DataFrame
        DataFrame da DVA filtrada.
    """
    # Lendo o arquivo parquet
    dva = pl.read_parquet(path)

    # Selecionando as colunas
    dva = dva.select([
        'DT_REFER', 
        'DENOM_CIA', 
        'CD_CVM', 
        'ORDEM_EXERC', 
        'DT_INI_EXERC',
        'DT_FIM_EXERC',
        'CD_CONTA', 
        'DS_CONTA', 
        'VL_CONTA'
    ])

    # Transformando os dtypes das colunas
    dva = dva.with_columns([
        pl.col('DT_REFER').str.strptime(pl.Date, '%Y-%m-%d').alias('DT_REFER'),
        pl.col('DENOM_CIA').cast(pl.Utf8),
        pl.col('CD_CVM').cast(pl.Int32, strict=False),
        pl.col('ORDEM_EXERC').cast(pl.Utf8),
        pl.col('DT_INI_EXERC').str.strptime(pl.Date, '%Y-%m-%d').alias('DT_INI_EXERC'),
        pl.col('DT_FIM_EXERC').str.strptime(pl.Date, '%Y-%m-%d').alias('DT_FIM_EXERC'),
        pl.col('DS_CONTA').cast(pl.Utf8)
    ])

    # Transformando em datetime
    dva = dva.with_columns(
        pl.col('DT_REFER').cast(pl.Datetime).alias('DT_REFER'),
        pl.col('DT_INI_EXERC').cast(pl.Datetime).alias('DT_INI_EXERC'),
        pl.col('DT_FIM_EXERC').cast(pl.Datetime).alias('DT_FIM_EXERC'),
    )

    # Filtrando apenas 'ÚLTIMO' em ORDEM_EXERC
    dva = dva.filter(pl.col('ORDEM_EXERC') == 'ÚLTIMO')

    # Renomeando as colunas
    dva = dva.rename({col: col.lower() for col in dva.columns})

    return dva


def dict_to_polars_df(dictionary: dict) -> pl.DataFrame:
    """
    Transforma um dicionário em um DataFrame polars.

    Parameters
    ----------
    dictionary: dict
        Dicionário python.

    Returns
    -------
    df: pl.DataFrame
        DataFrame polars.
    """

    # Transformando o dicionário em um df
    df = pl.DataFrame(dictionary)

    # Transformando a coluna em datetime
    df = df.with_columns(
        pl.col('dt_refer').str.strptime(pl.Datetime, '%Y-%m-%d')
    )

    # Ordenando pela data
    df = df.sort('dt_refer')

    return df


def df_composicao_capital(path: str) -> pl.DataFrame:
    """
    Abre e filtra o arquivo compilado da composição de capital.

    Parameters
    ----------
    path: str
        Caminho do arquivo.

    Returns
    -------
    composicao_capital: pl.DataFrame
        DataFrame da composição de capital filtrada.
    """
    # Lendo o arquivo parquet
    composicao_capital = pl.read_parquet(path)

    # Selecionando as colunas
    composicao_capital = composicao_capital.select(['DT_REFER', 'DENOM_CIA', 'QT_ACAO_ORDIN_CAP_INTEGR', 'QT_ACAO_PREF_CAP_INTEGR', 'QT_ACAO_TOTAL_CAP_INTEGR'])

    # Transformando os dtypes das colunas
    composicao_capital = composicao_capital.with_columns([
        pl.col('DT_REFER').str.strptime(pl.Date, '%Y-%m-%d').alias('DT_REFER'),
        pl.col('DENOM_CIA').cast(pl.Utf8),
    ])

    # Transformando em datetime
    composicao_capital = composicao_capital.with_columns(pl.col('DT_REFER').cast(pl.Datetime).alias('DT_REFER'))

    # Renomeando as colunas
    composicao_capital = composicao_capital.rename({col: col.lower() for col in composicao_capital.columns})

    composicao_capital = composicao_capital.rename({
        'qt_acao_ordin_cap_integr': 'num_on',
        'qt_acao_pref_cap_integr': 'num_pn',	
        'qt_acao_total_cap_integr': 'num_total'
    })

    return composicao_capital


def ativo_fechamento_anual(
    ticker_on_unit: str,
    list_ticker_on_pn: list,
    primeiro_ano: int, 
    ultimo_ano: int, 
) -> pl.DataFrame:
    """
    Seleciona os preços de fechamento anual da empresa.

    Parameters
    ----------
    ticker_on_unit: str
        Ticker da empresa ON ou UNIT.
    list_ticker_on_pn: list
        Lista que contém os tickers das empresas ON e PN.
    primeiro_ano: int
        Primeiro ano dos dados utilizados da empresa.
    ultimo_ano: int
        Último ano dos dados utilizados da empresa.

    Returns
    -------
    df_ativo_fechamento_ano: pl.DataFrame
        DataFrame dos preços de fechamento anual da empresa.
    """        
    # Preço de fechamento de uma ação ON ou UNIT
    if ticker_on_unit:
        # Preço de fechamento do ativo
        ativo_fechamento = yf.download(ticker_on_unit, start='2017-01-01', auto_adjust=True, multi_level_index=False, progress=False)['Close']

        # Selecionando o último fechamento de cada ano
        ultimo_fechamento_ano = ativo_fechamento.resample('YE').last().dropna()

        # Filtrando pelo intervalo de anos desejados 
        ultimo_fechamento_hist = ultimo_fechamento_ano.loc[str(primeiro_ano):str(ultimo_ano)]

        # Renomeando as colunas e o index
        if ticker_on_unit[4] == '3':
            ultimo_fechamento_hist = ultimo_fechamento_hist.rename('close_on')
        if ticker_on_unit[4] == '1':
            ultimo_fechamento_hist = ultimo_fechamento_hist.rename('close_unit')

        ultimo_fechamento_hist.index = ultimo_fechamento_hist.index.rename('dt_refer')
    
    # Preço de fechamento das ações ON e PN
    if list_ticker_on_pn:
        # Preço de fechamento do ativo
        ativo_fechamento = yf.download(list_ticker_on_pn, start='2017-01-01', auto_adjust=True, multi_level_index=False, progress=False)['Close']

        # Selecionando o último fechamento de cada ano
        ultimo_fechamento_ano = ativo_fechamento.resample('YE').last().dropna()

        # Filtrando pelo intervalo de anos desejados 
        ultimo_fechamento_hist = ultimo_fechamento_ano.loc[str(primeiro_ano):str(ultimo_ano)]

        # Renomeando as colunas e o index
        ultimo_fechamento_hist.columns = ['close_on', 'close_pn']
        ultimo_fechamento_hist.index = ultimo_fechamento_hist.index.rename('dt_refer')

    # Transformando em df polars
    df_ativo_fechamento_ano = pd.DataFrame(ultimo_fechamento_hist).reset_index()
    df_ativo_fechamento_ano = pl.from_pandas(df_ativo_fechamento_ano)

    # Transformando o datetime de 'datetime[ns]' para 'datetime[μs]'
    df_ativo_fechamento_ano = df_ativo_fechamento_ano.with_columns(
        pl.col('dt_refer').cast(pl.Datetime('us'))
    )

    return df_ativo_fechamento_ano


def ativo_fechamento_anual_csv(
    ticker_on_unit: str,
    path_csv_on_unit: str, 
    path_csv_pn: str, 
    primeiro_ano: int, 
    ultimo_ano: int, 
) -> pl.DataFrame:
    """
    Seleciona os preços de fechamento anual da empresa (arquivo csv do Investing.com).

    Parameters
    ----------
    ticker_on_unit: str
        Ticker da empresa ON ou UNIT.
    path_csv_on_unit: str
        Caminho do arquivo csv da empresa ON ou UNIT.
    path_csv_pn: str
        Caminho do arquivo csv da empresa PN.
    primeiro_ano: int
        Primeiro ano dos dados utilizados da empresa.
    ultimo_ano: int
        Último ano dos dados utilizados da empresa.

    Returns
    -------
    df_ativo_fechamento_ano: pl.DataFrame
        DataFrame dos preços de fechamento anual da empresa.
    """ 
    if ticker_on_unit:
        # Preço de fechamento do ativo
        ativo_fechamento = pd.read_csv(path_csv_on_unit)
        ativo_fechamento = ativo_fechamento.rename(columns={'Data':'dt_refer', 'Último':'Close'})
        ativo_fechamento = ativo_fechamento[['dt_refer', 'Close']]
        ativo_fechamento['dt_refer'] = pd.to_datetime(ativo_fechamento['dt_refer'], format='%d.%m.%Y')
        ativo_fechamento['Close'] = ativo_fechamento['Close'].str.replace(',', '.').astype(float)
        ativo_fechamento = ativo_fechamento.set_index('dt_refer')
        ativo_fechamento = ativo_fechamento.iloc[::-1]

        # Selecionando o último fechamento de cada ano
        ultimo_fechamento_ano = ativo_fechamento.resample('YE').last().dropna()

        # Filtrando pelo intervalo de anos desejados 
        ultimo_fechamento_hist = ultimo_fechamento_ano.loc[str(primeiro_ano):str(ultimo_ano)]

        # Renomeando a coluna 
        if ticker_on_unit[4] == '3':
            ultimo_fechamento_hist = ultimo_fechamento_hist.rename(columns={'Close':'close_on'})
        if ticker_on_unit[4] == '1':
            ultimo_fechamento_hist = ultimo_fechamento_hist.rename(columns={'Close':'close_unit'})

    else:
        # Preço de fechamento do ativo ON
        ativo_fechamento_on = pd.read_csv(path_csv_on_unit)
        ativo_fechamento_on = ativo_fechamento_on.rename(columns={'Data':'dt_refer', 'Último':'close_on'})
        ativo_fechamento_on = ativo_fechamento_on[['dt_refer', 'close_on']]
        ativo_fechamento_on['dt_refer'] = pd.to_datetime(ativo_fechamento_on['dt_refer'], format='%d.%m.%Y')
        ativo_fechamento_on['close_on'] = ativo_fechamento_on['close_on'].str.replace(',', '.').astype(float)
        ativo_fechamento_on = ativo_fechamento_on.set_index('dt_refer')
        ativo_fechamento_on = ativo_fechamento_on.iloc[::-1]

        # Preço de fechamento do ativo PN
        ativo_fechamento_pn = pd.read_csv(path_csv_pn)
        ativo_fechamento_pn = ativo_fechamento_pn.rename(columns={'Data':'dt_refer', 'Último':'close_pn'})
        ativo_fechamento_pn = ativo_fechamento_pn[['dt_refer', 'close_pn']]
        ativo_fechamento_pn['dt_refer'] = pd.to_datetime(ativo_fechamento_pn['dt_refer'], format='%d.%m.%Y')
        ativo_fechamento_pn['close_pn'] = ativo_fechamento_pn['close_pn'].str.replace(',', '.').astype(float)
        ativo_fechamento_pn = ativo_fechamento_pn.set_index('dt_refer')
        ativo_fechamento_pn = ativo_fechamento_pn.iloc[::-1]

        # Juntando os dfs
        ativo_fechamento = pd.concat([ativo_fechamento_on, ativo_fechamento_pn], axis=1)

        # Selecionando o último fechamento de cada ano
        ultimo_fechamento_ano = ativo_fechamento.resample('YE').last().dropna()

        # Filtrando pelo intervalo de anos desejados 
        ultimo_fechamento_hist = ultimo_fechamento_ano.loc[str(primeiro_ano):str(ultimo_ano)]

    # Transformando em df polars
    df_ativo_fechamento_ano = pd.DataFrame(ultimo_fechamento_hist).reset_index()
    df_ativo_fechamento_ano = pl.from_pandas(df_ativo_fechamento_ano)

    # Transformando o datetime de 'datetime[ns]' para 'datetime[μs]'
    df_ativo_fechamento_ano = df_ativo_fechamento_ano.with_columns(
        pl.col('dt_refer').cast(pl.Datetime('us'))
    )
    
    return df_ativo_fechamento_ano


def ativo_fechamento_trimestral(    
    ticker_on_unit: str, 
    list_ticker_on_pn: list,
    primeiro_ano: int, 
    ultimo_ano: int, 
    pl_series_dt_refer: pl.Series
) -> pl.DataFrame:
    """
    Seleciona os preços de fechamento trimestral da empresa.

    Parameters
    ----------
    ticker_on_unit: str
        Ticker da empresa ON ou UNIT.
    list_ticker_on_pn: list
        Lista que contém os tickers das empresas ON e PN.
    primeiro_ano: int
        Primeiro ano dos dados utilizados da empresa.
    ultimo_ano: int
        Último ano dos dados utilizados da empresa.
    pl_series_dt_refer: pl.Series
        Série polars de uma coluna 'dt_refer' para ser adicionado no 'df_ativo_fechamento_mes'.

    Returns
    -------
    df_ativo_fechamento_mes: pl.DataFrame
        DataFrame dos preços de fechamento trimestral da empresa.
    """
    # Preço de fechamento de uma ação ON ou UNIT
    if ticker_on_unit:
        # Preço de fechamento do ativo
        ativo_fechamento = yf.download(ticker_on_unit, start='2020-01-01', auto_adjust=True, multi_level_index=False, progress=False)['Close']

        # Selecionando o último fechamento de cada mês 
        ultimo_fechamento_mes = ativo_fechamento.resample('ME').last().dropna()

        # Lista do meses trimestrais
        lst_mes_alvo = [3, 6, 9]

        # Filtrando os fechamentos pelos meses trimestrais
        fechamento_trimestral_filt = ultimo_fechamento_mes[ultimo_fechamento_mes.index.month.isin(lst_mes_alvo)]

        # Filtrando pelo intervalo de anos desejados - retirando o ano mais atual
        primeiro_ano_str = str(primeiro_ano)
        ultimo_ano_str = str(ultimo_ano - 1)
        ultimo_fechamento_hist = fechamento_trimestral_filt.loc[primeiro_ano_str:ultimo_ano_str]

        # Ano atual
        ano_atual_int = datetime.now().year

        # Filtrando pelo ano atual, selecionando apenas as datas do ano atual
        ultimo_fechamento_filt_atual = ultimo_fechamento_mes.loc[str(ano_atual_int)]

        # Último mês do trimestre mais recente da série 'pl_series_dt_refer'
        ultimo_mes_refer = pl_series_dt_refer[-1].month 

        # Lista do mês atual conforme o último trimestre lançado
        lst_mes_atual = []
        if ultimo_mes_refer >= 3:
            lst_mes_atual.append(3) # ['03']
        if ultimo_mes_refer >= 6:
            lst_mes_atual.append(6) # ['03', '06']
        if ultimo_mes_refer >= 9:
            lst_mes_atual.append(9) # ['03', '06', '09']

        # Filtrando pelos meses da 'lst_mes_atual'
        ultimo_fechamento_atual = ultimo_fechamento_filt_atual[ultimo_fechamento_filt_atual.index.month.isin(lst_mes_atual)]

        # Juntando as duas series pandas
        ativo_fechamento_mes = pd.concat([ultimo_fechamento_hist , ultimo_fechamento_atual])

        # Renomeando a série e o index
        if ticker_on_unit[4] == '3':
            ativo_fechamento_mes = ativo_fechamento_mes.rename('close_on')
        elif ticker_on_unit[4] == '1':
            ativo_fechamento_mes = ativo_fechamento_mes.rename('close_unit')

        ativo_fechamento_mes.index = ativo_fechamento_mes.index.rename('dt_refer')

    # Preço de fechamento das ações ON e PN
    if list_ticker_on_pn:
        # Preço de fechamento do ativo
        ativo_fechamento = yf.download(list_ticker_on_pn, start='2020-01-01', auto_adjust=True, multi_level_index=False, progress=False)['Close']

        # Selecionando o último fechamento de cada mês 
        ultimo_fechamento_mes = ativo_fechamento.resample('ME').last().dropna()

        # Lista do meses trimestrais
        lst_mes_alvo = [3, 6, 9]

        # Filtrando os fechamentos dos meses trimestrais
        fechamento_trimestral_filt = ultimo_fechamento_mes[ultimo_fechamento_mes.index.month.isin(lst_mes_alvo)]

        # Filtrando pelo intervalo de anos desejados 
        primeiro_ano_str = str(primeiro_ano)
        ultimo_ano_str = str(ultimo_ano - 1)
        ultimo_fechamento_hist = fechamento_trimestral_filt.loc[primeiro_ano_str:ultimo_ano_str]

        # Ano atual
        ano_atual_int = datetime.now().year

        # Filtrando pelo ano atual, selecionando apenas as datas do ano atual
        ultimo_fechamento_filt_atual = ultimo_fechamento_mes.loc[str(ano_atual_int)]

        # Último mês do trimestre mais recente (pl_series_dt_refer)
        ultimo_mes_refer = pl_series_dt_refer[-1].month 

        # Lista do mês atual conforme o último trimestre lançado
        lst_mes_atual = []
        if ultimo_mes_refer >= 3:
            lst_mes_atual.append(3) # ['03']
        if ultimo_mes_refer >= 6:
            lst_mes_atual.append(6) # ['03', '06']
        if ultimo_mes_refer >= 9:
            lst_mes_atual.append(9) # ['03', '06', '09']

        # Filtrando pelos meses da 'lst_mes_atual'
        ultimo_fechamento_atual = ultimo_fechamento_filt_atual[ultimo_fechamento_filt_atual.index.month.isin(lst_mes_atual)]

        # Juntando as duas series pandas
        ativo_fechamento_mes = pd.concat([ultimo_fechamento_hist , ultimo_fechamento_atual])

        # Renomeando as colunas e o index
        ativo_fechamento_mes.columns = ['close_on', 'close_pn']
        ativo_fechamento_mes.index = ativo_fechamento_mes.index.rename('dt_refer')

    # Transformando em df polars
    df_ativo_fechamento_mes = pd.DataFrame(ativo_fechamento_mes).reset_index()
    df_ativo_fechamento_mes = pl.from_pandas(df_ativo_fechamento_mes)

    # Transformando o datetime de 'datetime[ns]' para 'datetime[μs]'
    df_ativo_fechamento_mes = df_ativo_fechamento_mes.with_columns(
        pl.col('dt_refer').cast(pl.Datetime('us'))
    )

    return df_ativo_fechamento_mes


def ativo_fechamento_trimestral_csv(    
    ticker_on_unit: str,
    list_ticker_on_pn: list,
    path_csv_on_unit: str, 
    path_csv_pn: str, 
    primeiro_ano: int, 
    ultimo_ano: int, 
    pl_series_dt_refer: pl.Series
) -> pl.DataFrame:
    """
    Seleciona os preços de fechamento trimestral da empresa.

    Parameters
    ----------
    ticker_on_unit: str
        Ticker da empresa ON ou UNIT.
    list_ticker_on_pn: list
        Lista que contém os tickers das empresas ON e PN.
    path_csv_on_unit: str
        Caminho do arquivo csv da empresa ON ou UNIT.
    path_csv_pn: str
        Caminho do arquivo csv da empresa PN.
    primeiro_ano: int
        Primeiro ano dos dados utilizados da empresa.
    ultimo_ano: int
        Último ano dos dados utilizados da empresa.
    pl_series_dt_refer: pl.Series
        Série polars de uma coluna 'dt_refer' para ser adicionado no 'df_ativo_fechamento_mes'.

    Returns
    -------
    df_ativo_fechamento_mes: pl.DataFrame
        DataFrame dos preços de fechamento trimestral da empresa.
    """
    # Preço de fechamento de uma ação ON ou UNIT
    if ticker_on_unit:
        ativo_fechamento = pd.read_csv(path_csv_on_unit)
        ativo_fechamento = ativo_fechamento.rename(columns={'Data':'dt_refer', 'Último':'Close'})
        ativo_fechamento = ativo_fechamento[['dt_refer', 'Close']]
        ativo_fechamento['dt_refer'] = pd.to_datetime(ativo_fechamento['dt_refer'], format='%d.%m.%Y')
        ativo_fechamento['Close'] = ativo_fechamento['Close'].str.replace(',', '.').astype(float)
        ativo_fechamento = ativo_fechamento.set_index('dt_refer')
        ativo_fechamento = ativo_fechamento.iloc[::-1]

        # Selecionando o último fechamento de cada mês
        ultimo_fechamento_mes = ativo_fechamento.resample('ME').last().dropna()

        # Lista do meses trimestrais
        lst_mes_alvo = [3, 6, 9]

        # Filtrando os fechamentos pelos meses trimestrais
        fechamento_trimestral_filt = ultimo_fechamento_mes[ultimo_fechamento_mes.index.month.isin(lst_mes_alvo)]

        # Filtrando pelo intervalo de anos desejados - retirando o ano mais atual
        primeiro_ano_str = str(primeiro_ano)
        ultimo_ano_str = str(ultimo_ano - 1)
        ultimo_fechamento_mes = fechamento_trimestral_filt.loc[primeiro_ano_str:ultimo_ano_str]

        # Ano atual
        ano_atual_int = datetime.now().year
        # Ano do lançamento mais recente
        ano_lancamento_mais_recente = pl_series_dt_refer[-1].year

        # Se o ano atual for igual ao ano do lançamento mais recente
        if ano_lancamento_mais_recente == ano_atual_int:

            # Filtrando pelo ano atual, selecionando apenas as datas do ano atual
            ultimo_fechamento_filt_atual = ultimo_fechamento_mes.loc[str(ano_atual_int)]

            # Último mês do trimestre mais recente da série 'pl_series_dt_refer'
            ultimo_mes_refer = pl_series_dt_refer[-1].month 

            # Lista do mês atual conforme o último trimestre lançado
            lst_mes_atual = []
            if ultimo_mes_refer >= 3:
                lst_mes_atual.append(3) # ['03']
            if ultimo_mes_refer >= 6:
                lst_mes_atual.append(6) # ['03', '06']
            if ultimo_mes_refer >= 9:
                lst_mes_atual.append(9) # ['03', '06', '09']

            # Filtrando pelos meses da 'lst_mes_atual'
            ultimo_fechamento_atual = ultimo_fechamento_filt_atual[ultimo_fechamento_filt_atual.index.month.isin(lst_mes_atual)]

            # Juntando as duas series pandas
            ativo_fechamento_mes = pd.concat([ultimo_fechamento_mes , ultimo_fechamento_atual])

            # Renomeando a série e o index
            if ticker_on_unit[4] == '3':
                ativo_fechamento_mes = ativo_fechamento_mes.rename(columns={'Close':'close_on'})
            elif ticker_on_unit[4] == '1':
                ativo_fechamento_mes = ativo_fechamento_mes.rename(columns={'Close':'close_unit'})

            ativo_fechamento_mes.index = ativo_fechamento_mes.index.rename('dt_refer')

        else: 
            # Renomeando a série e o index
            if ticker_on_unit[4] == '3':
                ativo_fechamento_mes = ultimo_fechamento_mes.rename(columns={'Close':'close_on'})
            elif ticker_on_unit[4] == '1':
                ativo_fechamento_mes = ultimo_fechamento_mes.rename(columns={'Close':'close_unit'})

            ativo_fechamento_mes.index = ativo_fechamento_mes.index.rename('dt_refer')

    # Preço de fechamento das ações ON e PN
    if list_ticker_on_pn:
        # Ação ON
        ativo_fechamento_on = pd.read_csv(path_csv_on_unit)
        ativo_fechamento_on = ativo_fechamento_on.rename(columns={'Data':'dt_refer', 'Último':'Close'})
        ativo_fechamento_on = ativo_fechamento_on[['dt_refer', 'Close']]
        ativo_fechamento_on['dt_refer'] = pd.to_datetime(ativo_fechamento_on['dt_refer'], format='%d.%m.%Y')
        ativo_fechamento_on['Close'] = ativo_fechamento_on['Close'].str.replace(',', '.').astype(float)
        ativo_fechamento_on = ativo_fechamento_on.set_index('dt_refer')
        ativo_fechamento_on = ativo_fechamento_on.iloc[::-1]
        # Ação PN
        ativo_fechamento_pn = pd.read_csv(path_csv_pn)
        ativo_fechamento_pn = ativo_fechamento_pn.rename(columns={'Data':'dt_refer', 'Último':'Close'})
        ativo_fechamento_pn = ativo_fechamento_pn[['dt_refer', 'Close']]
        ativo_fechamento_pn['dt_refer'] = pd.to_datetime(ativo_fechamento_pn['dt_refer'], format='%d.%m.%Y')
        ativo_fechamento_pn['Close'] = ativo_fechamento_pn['Close'].str.replace(',', '.').astype(float)
        ativo_fechamento_pn = ativo_fechamento_pn.set_index('dt_refer')
        ativo_fechamento_pn = ativo_fechamento_pn.iloc[::-1]

        # Selecionando o último fechamento de cada mês
        ultimo_fechamento_mes_on = ativo_fechamento_on.resample('ME').last().dropna()
        ultimo_fechamento_mes_pn = ativo_fechamento_pn.resample('ME').last().dropna()

        # Lista do meses trimestrais
        lst_mes_alvo = [3, 6, 9]

        # Filtrando os fechamentos pelos meses trimestrais
        fechamento_trimestral_filt_on = ultimo_fechamento_mes_on[ultimo_fechamento_mes_on.index.month.isin(lst_mes_alvo)]
        fechamento_trimestral_filt_pn = ultimo_fechamento_mes_pn[ultimo_fechamento_mes_pn.index.month.isin(lst_mes_alvo)]

        # Filtrando pelo intervalo de anos desejados - retirando o ano mais atual
        primeiro_ano_str = str(primeiro_ano)
        ultimo_ano_str = str(ultimo_ano - 1)
        ultimo_fechamento_mes_on = fechamento_trimestral_filt_on.loc[primeiro_ano_str:ultimo_ano_str]
        ultimo_fechamento_mes_pn = fechamento_trimestral_filt_pn.loc[primeiro_ano_str:ultimo_ano_str]

        # Ano atual
        ano_atual_int = datetime.now().year
        # Ano do lançamento mais recente
        ano_lancamento_mais_recente = pl_series_dt_refer[-1].year

        # Se o ano atual for igual ao ano do lançamento mais recente
        if ano_lancamento_mais_recente == ano_atual_int:

            # Filtrando pelo ano atual, selecionando apenas as datas do ano atual
            ultimo_fechamento_filt_atual_on = ultimo_fechamento_mes_on.loc[str(ano_atual_int)]
            ultimo_fechamento_filt_atual_pn = ultimo_fechamento_mes_pn.loc[str(ano_atual_int)]

            # Último mês do trimestre mais recente da série 'pl_series_dt_refer'
            ultimo_mes_refer = pl_series_dt_refer[-1].month 

            # Lista do mês atual conforme o último trimestre lançado
            lst_mes_atual = []
            if ultimo_mes_refer >= 3:
                lst_mes_atual.append(3) # ['03']
            if ultimo_mes_refer >= 6:
                lst_mes_atual.append(6) # ['03', '06']
            if ultimo_mes_refer >= 9:
                lst_mes_atual.append(9) # ['03', '06', '09']

            # Filtrando pelos meses da 'lst_mes_atual'
            ultimo_fechamento_atual_on = ultimo_fechamento_filt_atual_on[ultimo_fechamento_filt_atual_on.index.month.isin(lst_mes_atual)]
            ultimo_fechamento_atual_pn = ultimo_fechamento_filt_atual_pn[ultimo_fechamento_filt_atual_pn.index.month.isin(lst_mes_atual)]

            # Juntando as duas series pandas
            ativo_fechamento_mes_on = pd.concat([ultimo_fechamento_mes_on , ultimo_fechamento_atual_on])
            ativo_fechamento_mes_pn = pd.concat([ultimo_fechamento_mes_pn , ultimo_fechamento_atual_pn]) 

            # Renomeando as colunas
            ativo_fechamento_mes_on = ultimo_fechamento_mes_on.rename(columns={'Close':'close_on'})
            ativo_fechamento_mes_pn = ultimo_fechamento_mes_pn.rename(columns={'Close':'close_pn'})

        else:
            # Renomeando as colunas
            ativo_fechamento_mes_on = ultimo_fechamento_mes_on.rename(columns={'Close':'close_on'})
            ativo_fechamento_mes_pn = ultimo_fechamento_mes_pn.rename(columns={'Close':'close_pn'})

        # Concatenando os dfs
        ativo_fechamento_mes = pd.concat([ativo_fechamento_mes_on, ativo_fechamento_mes_pn], axis=1)

    # Transformando em df polars
    df_ativo_fechamento_mes = pd.DataFrame(ativo_fechamento_mes).reset_index()
    df_ativo_fechamento_mes = pl.from_pandas(df_ativo_fechamento_mes)

    # Transformando o datetime de 'datetime[ns]' para 'datetime[μs]'
    df_ativo_fechamento_mes = df_ativo_fechamento_mes.with_columns(
        pl.col('dt_refer').cast(pl.Datetime('us'))
    )

    return df_ativo_fechamento_mes


def indicador_vm_atual(df_capital_social: pl.DataFrame, ticker_on: str, ticker_pn: str) -> float:
    """
    Calcula o valor de mercado da empresa atual. 

    Parameters
    ----------
    df_capital_social: pl.DataFrame
        DataFrame que contém o nº de ações ordinárias e preferenciais.
    ticker_on: str
        Ticker da empresa ON.
    ticker_pn: str
        Ticker da empresa PN.

    Returns
    -------
    valor_mercado: float
        Valor de mercado da empresa.
    """
    # Empresas do novo mercado
    if not ticker_pn:
        # Selecionando apenas as letras do ticker da ação
        filt_ticker_on = df_capital_social['ticker'] == ticker_on[0:4]
        
        # Números das ações ordinárias
        num_on = df_capital_social.filter(filt_ticker_on).select('num_on').item()

        # Preço da ação ordinária
        preço_on = yf.download(ticker_on, auto_adjust=True, multi_level_index=False, progress=False)['Close'].iloc[-1]

        # Calculando o valor de mercado
        valor_mercado = round((num_on * preço_on), 2)
    
    # Empresas que não são do novo mercado
    else:
        # Selecionando apenas as letras do ticker da ação
        filt_ticker_on = df_capital_social['ticker'] == ticker_on[0:4]

        # Números das ações ordinárias
        num_on = df_capital_social.filter(filt_ticker_on).select('num_on').item()

        # Preço da ação ordinária
        preço_on = yf.download(ticker_on, auto_adjust=True, multi_level_index=False, progress=False)['Close'].iloc[-1]

        # Selecionando apenas as letras do ticker da ação
        filt_ticker_pn = df_capital_social['ticker'] == ticker_pn[0:4]

        # Números das ações preferenciais
        num_pn = df_capital_social.filter(filt_ticker_pn).select('num_pn').item()

        # Preço da ação ordinária
        preço_pn = yf.download(ticker_pn, auto_adjust=True, multi_level_index=False, progress=False)['Close'].iloc[-1]

        # Calculando o valor de mercado
        valor_mercado = round(((num_on * preço_on) + (num_pn * preço_pn)), 2)
    
    return valor_mercado


def indicador_vm(df_capital_social: pl.DataFrame, df_fechamento: pl.DataFrame, novo_mercado: bool) -> pl.DataFrame:
    """
    Calcula o valor de mercado da empresa. 

    Parameters
    ----------
    df_capital_social: pl.DataFrame
        DataFrame que contém o nº de ações ordinárias e preferenciais.
    df_fechamento: pl.DataFrame
        DataFrame que contém os preços de fechamento da empresa.
    novo_mercado: bool
        A empresa é do novo mercado (True), se não é do novo mercado (False).

    Returns
    -------
    df_valor_mercado: pl.DataFrame
        DataFrame do valor de mercado da empresa.
    """
    # Empresas do novo mercado ou units
    if novo_mercado:

        # Juntando os dfs
        df_valor_mercado = df_capital_social.join(df_fechamento, on='dt_refer', how='inner')
        
        # Calculando o valor de mercado
        df_valor_mercado = df_valor_mercado.with_columns(
            (pl.col('num_on') * pl.col('close_on')).alias('valor_mercado')
        )

        # Selecionando as principais colunas
        df_valor_mercado = df_valor_mercado.select('dt_refer', 'valor_mercado')

    # Empresas que não são do novo mercado
    else:
        # Juntando os dfs
        df_valor_mercado = df_capital_social.join(df_fechamento, on='dt_refer', how='inner')

        # Calculando o valor de mercado
        df_valor_mercado = df_valor_mercado.with_columns(
            ((pl.col('num_on') * pl.col('close_on')) + (pl.col('num_pn') * pl.col('close_pn'))).alias('valor_mercado')
        )

        # Selecionando as principais colunas
        df_valor_mercado = df_valor_mercado.select('dt_refer', 'valor_mercado')

    return df_valor_mercado


def indicador_lpa(df_capital_social: pl.DataFrame, df_dre: pl.DataFrame, cod_cvm: int, cod_ll: str, novo_mercado: bool) -> pl.DataFrame:
    """
    Calcula o lucro por ação (lpa) da empresa. 

    Parameters
    ----------
    df_capital_social: pl.DataFrame
        DataFrame que contém o nº de ações ordinárias e preferenciais.
    df_dre: pl.DataFrame
        DataFrame da demonstração do resultado do exercício.
    cod_cvm: int
        Código da empresa na B3.
    cod_ll: str
        Código do lucro líquido na dre.
    novo_mercado: bool
        A empresa é do novo mercado (True), se não é do novo mercado (False).

    Returns
    -------
    df_lpa: pl.DataFrame
        DataFrame do lucro por ação da empresa.
    """
    # Selecionando a dre
    dre_empresa = df_dre.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o lucro líquido
    lucro_liquido = dre_empresa.filter(pl.col('cd_conta') == cod_ll).select('dt_refer', 'vl_conta')

    # Juntando os dfs
    df_lpa = lucro_liquido.join(df_capital_social, on='dt_refer', how='inner')

    # Empresas do novo mercado
    if novo_mercado:
        # Calculando o lpa
        df_lpa = df_lpa.with_columns(
            ((pl.col('vl_conta')*1000) / pl.col('num_on')).alias('lpa')
        ).select('dt_refer', 'lpa')

    # Empresas que não são do novo mercado
    else:
        # Calculando o lpa
        df_lpa = df_lpa.with_columns(
            ((pl.col('vl_conta')*1000) / pl.col('num_total')).alias('lpa')
        ).select('dt_refer', 'lpa')

    return df_lpa


def indicador_lpa_trimestral(df_capital_social: pl.DataFrame, df_dre: pl.DataFrame, cod_cvm: int, cod_ll: str, novo_mercado: bool) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Calcula o lucro por ação (LPA) trimestral e acumulado da empresa. 

    Parameters
    ----------
    df_capital_social: pl.DataFrame
        DataFrame que contém o nº de ações ordinárias e preferenciais.
    df_dre: pl.DataFrame
        DataFrame da demonstração do resultado do exercício.
    cod_cvm: int
        Código da empresa na B3.
    cod_ll: str
        Código do lucro líquido na dre.
    novo_mercado: bool
        A empresa é do novo mercado (True), se não é do novo mercado (False).

    Returns
    -------
    df_lpa_trimestre: pl.DataFrame
        DataFrame do LPA trimestral da empresa.
    df_lpa_trimestre_acum: pl.DataFrame
        DataFrame do LPA trimestral acumulado da empresa.
    """
    # Selecionando a dre
    dre_empresa = df_dre.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o lucro líquido
    lucro_liquido = dre_empresa.filter(pl.col('cd_conta') == cod_ll).select('dt_refer', 'dt_ini_exerc', 'vl_conta')

    # Existem linhas duplicadas na coluna 'dt_refer', removendo as linhas duplicadas, mantendo pela data mais recente da 'dt_ini_exerc'
    lucro_liquido = (lucro_liquido
        .sort(['dt_refer', 'dt_ini_exerc'], descending=[False, True])
        .unique(subset='dt_refer', keep='first')
        .sort('dt_refer')
    ).select('dt_refer', 'vl_conta')

    # Juntando os dfs
    df = lucro_liquido.join(df_capital_social, on='dt_refer', how='inner')

    # Empresas do novo mercado
    if novo_mercado:
        # Calculando o lpa
        df = df.with_columns(
            ((pl.col('vl_conta')*1000) / pl.col('num_on')).alias('lpa')
        )
    
    # Empresas que não são do novo mercado
    else:
        # Calculando o lpa
        df = df.with_columns(
            ((pl.col('vl_conta')*1000) / pl.col('num_total')).alias('lpa')
        )

    # Calculando lpa acumulado
    df = df.with_columns(
        pl.col('lpa')
        .rolling_sum(window_size=4, min_periods=4) 
        .round(2)                                   
        .alias('lpa_acum')
    )

    # Selecionando as principais colunas
    df_lpa_trimestre = df.select('dt_refer', 'lpa')
    df_lpa_trimestre_acum = df.select('dt_refer', 'lpa_acum').rename({'lpa_acum': 'lpa'})

    return df_lpa_trimestre, df_lpa_trimestre_acum


def indicador_pl_lp(df_fechamento: pl.DataFrame, tipo_acao: str, df_lpa: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Calcula os indicadores P/L (Preço/Lucro) e L/P (Lucro/Preço - 'Earning Yield').

    Parameters
    ----------
    df_fechamento: pl.DataFrame
        DataFrame que contém os preços de fechamento da empresa.
    tipo_acao: str
        'on' (ação ordinária), 'pn' (ação preferencial) e 'unit' (units).
    df_lpa: pl.DataFrame
        DataFrame do LPA (Lucro Por Ação) da empresa.

    Returns
    -------
    df_pl: pl.DataFrame
        DataFrame com a coluna 'p/l' e 'dt_refer'.
    df_lp: pl.DataFrame
        DataFrame com a coluna 'l/p' e 'dt_refer'.
    """
    # Mapeando o tipo de ação para o nome da coluna de fechamento
    coluna_fechamento_map = {
        'on': 'close_on',
        'pn': 'close_pn',
        'unit': 'close_unit'
    }

    try:
        coluna_fechamento = coluna_fechamento_map[tipo_acao]
    except KeyError:
        raise ValueError(f"Tipo de ação '{tipo_acao}' inválido. Use 'on', 'pn' ou 'unit'.")

    # Juntando os DataFrames
    df_combinado = df_fechamento.join(df_lpa, on='dt_refer', how='inner')

    # Calculando os indicadores
    df_calculado = df_combinado.with_columns([
        # Cálculo do P/L com tratamento para LPA zero - para não dar 'inf', e sim, zero
        pl.when(pl.col('lpa') == 0)
          .then(0)
          .otherwise(pl.col(coluna_fechamento) / pl.col('lpa'))
          .alias('p/l'),
        
        # Cálculo do L/P (Earning Yield %)
        ((pl.col('lpa') / pl.col(coluna_fechamento)) * 100).alias('l/p')
    ])

    # Criando os dfs dos indicadores
    df_pl = df_calculado.select('dt_refer', 'p/l')
    df_lp = df_calculado.select('dt_refer', 'l/p')

    return df_pl, df_lp


def indicador_ebitda(df_dre: pl.DataFrame, df_dfc: pl.DataFrame, cod_cvm: int, cod_ebit: str, cod_depreciacao: str) -> pl.DataFrame:
    """
    Calcula o ebitda da empresa.

    Parameters
    ----------
    df_dre: pl.DataFrame
        DataFrame da demonstração do resultado do exercício.
    df_dfc: pl.DataFrame
        DataFrame do demonstrativo de fluxo de caixa.
    cod_cvm: int
        Código da empresa na B3.
    cod_ebit: str
        Código do ebit na dre.
    cod_depreciacao: str
        Código da depreciação na dfc.

    Returns
    -------
    df_ebitda: pl.DataFrame
        DataFrame do ebitda da empresa.
    """
    # Selecionando a dre
    dre_empresa = df_dre.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o ebit
    ebit = dre_empresa.filter(pl.col('cd_conta') == cod_ebit).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    ebit = ebit.rename({'vl_conta': 'ebit'})

    # Selecionando a dfc
    dfc_empresa = df_dfc.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando a depreciação
    depreciacao = dfc_empresa.filter(pl.col('cd_conta') == cod_depreciacao).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    depreciacao = depreciacao.rename({'vl_conta': 'depreciacao'})

    # Juntando os dfs
    df_ebitda = ebit.join(depreciacao, on='dt_refer', how='inner')

    # Calculando o ebitda
    df_ebitda = df_ebitda.with_columns(
        (pl.col('ebit') + pl.col('depreciacao')).alias('ebitda')
    ).select('dt_refer', 'ebitda')

    return df_ebitda


def indicador_ebitda_trimestral(df_dre: pl.DataFrame, df_dfc: pl.DataFrame, cod_cvm: int, cod_ebit: str, cod_depreciacao: str) -> pl.DataFrame:
    """
    Calcula o ebitda da empresa.

    Parameters
    ----------
    df_dre: pl.DataFrame
        DataFrame da demonstração do resultado do exercício.
    df_dfc: pl.DataFrame
        DataFrame do demonstrativo de fluxo de caixa.
    cod_cvm: int
        Código da empresa na B3.
    cod_ebit: str
        Código do ebit na dre.
    cod_depreciacao: str
        Código da depreciação na dfc.

    Returns
    -------
    df_ebitda_trimestral: pl.DataFrame
        DataFrame do ebitda trimestral da empresa.
    """
    # Selecionando a dre
    dre_empresa = df_dre.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o ebit
    ebit = dre_empresa.filter(pl.col('cd_conta') == cod_ebit).select('dt_refer', 'dt_ini_exerc', 'vl_conta')

    # Existem linhas duplicadas na coluna 'dt_refer', removendo as linhas duplicadas, mantendo pela data mais recente da 'dt_ini_exerc'
    ebit = (ebit
        .sort(['dt_refer', 'dt_ini_exerc'], descending=[False, True])
        .unique(subset='dt_refer', keep='first')
        .sort('dt_refer')
    ).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    ebit = ebit.rename({'vl_conta': 'ebit'})

    # Selecionando a dfc
    dfc_empresa = df_dfc.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando a depreciação
    depreciacao = dfc_empresa.filter(pl.col('cd_conta') == cod_depreciacao).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    depreciacao = depreciacao.rename({'vl_conta':'depreciacao'})

    # Calculando a depreciaçao não acumulada para os meses 06 e 09
    expr_deprec_nao_acum = (
        pl.when(
            (pl.col('dt_refer').dt.month() == 6) | (pl.col('dt_refer').dt.month() == 9)
        )
        .then(
            # Verifica se depreciacao é 0 para manter 0. Caso contrário, faz a subtração.
            pl.when(pl.col('depreciacao') == 0)
            .then(0)
            .otherwise(pl.col('depreciacao') - pl.col('depreciacao').shift(1))
        )
        .otherwise(pl.col('depreciacao')) 
        .alias('depreciacao_nao_acum')
    )

    depreciacao_nao_acum = depreciacao.with_columns([expr_deprec_nao_acum]).select('dt_refer', 'depreciacao_nao_acum')

    # Juntando os dfs
    df_ebitda_trimestral = ebit.join(depreciacao_nao_acum, on='dt_refer', how='inner')

    # Calculando o ebitda trimestral
    df_ebitda_trimestral = df_ebitda_trimestral.with_columns(
        (pl.col('ebit') + pl.col('depreciacao_nao_acum')).alias('ebitda')
    )

    # Calculando o ebitda trimestral acumulado
    df_ebitda_trimestral_acumulada = df_ebitda_trimestral.with_columns(
        pl.col('ebitda')
        .rolling_sum(window_size=4, min_periods=4) 
        .round(2)                                   
        .alias('ebitda_acum')
    )

    # Selecionando as principais colunas
    df_ebitda_trimestral = df_ebitda_trimestral.select('dt_refer', 'ebitda')
    df_ebitda_trimestral_acumulada = df_ebitda_trimestral_acumulada.select('dt_refer', 'ebitda_acum').rename({'ebitda_acum': 'ebitda'})

    return df_ebitda_trimestral, df_ebitda_trimestral_acumulada


def indicador_divida_bruta(df_bpp: pl.DataFrame, cod_cvm: int, cod_emprest_circ: str, cod_emprest_n_circ: str) -> pl.DataFrame:
    """
    Calcula a dívida bruta da empresa.

    Parameters
    ----------
    df_bpp: pl.DataFrame
        DataFrame do balanço patrimonial passivo.
    cod_cvm: int
        Código da empresa na B3.
    cod_emprest_circ: str
        Código do empréstimo circulante no bpp.
    cod_emprest_n_circ: str
        Código do empréstimo não circulante no bpp.

    Returns
    -------
    df_endividamento_total: pl.DataFrame
        DataFrame da dívida bruta da empresa.
    """
    # Selecionando a bpp
    bpp_empresa = df_bpp.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o empréstimo circulante
    emprestimo_circulante = bpp_empresa.filter(pl.col('cd_conta') == cod_emprest_circ).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    emprestimo_circulante = emprestimo_circulante.rename({'vl_conta': 'emprestimo_circulante'})

    # Selecionando o empréstimo não circulante
    emprestimo_nao_circulante = bpp_empresa.filter(pl.col('cd_conta') == cod_emprest_n_circ).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    emprestimo_nao_circulante = emprestimo_nao_circulante.rename({'vl_conta': 'emprestimo_nao_circulante'})

    # Juntando os dfs
    df_endividamento_total = emprestimo_circulante.join(emprestimo_nao_circulante, on='dt_refer', how='inner')

    # Calculando o caixa total
    df_endividamento_total = df_endividamento_total.with_columns(
        (pl.col('emprestimo_circulante') + pl.col('emprestimo_nao_circulante')).alias('divida_bruta')
    ).select('dt_refer', 'divida_bruta')

    return df_endividamento_total


def indicador_caixa(df_bpa: pl.DataFrame, cod_cvm: int, cod_caixa: str, cod_aplic_financ: str) -> pl.DataFrame:
    """
    Calcula o caixa total da empresa.

    Parameters
    ----------
    df_bpa: pl.DataFrame
        DataFrame do balanço patrimonial ativo.
    cod_cvm: int
        Código da empresa na B3.
    cod_caixa: str
        Código da caixa e equivalentes no bpa.
    cod_caixa: str
        Código da aplicações financeiras no bpa.

    Returns
    -------
    df_caixa_total: pl.DataFrame
        DataFrame do caixa total da empresa
    """
    # Selecionando a bpa
    bpa_empresa = df_bpa.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o caixa e equivalentes de caixa 	
    caixa = bpa_empresa.filter(pl.col('cd_conta') == cod_caixa).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    caixa = caixa.rename({'vl_conta': 'caixa'})

    # Selecionando as aplicações financeiras
    aplicacoes_financeiras = bpa_empresa.filter(pl.col('cd_conta') == cod_aplic_financ).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    aplicacoes_financeiras = aplicacoes_financeiras.rename({'vl_conta': 'aplicacoes_financeiras'})

    # Juntando os dfs
    df_caixa_total = caixa.join(aplicacoes_financeiras, on='dt_refer', how='inner')

    # Calculando o caixa total
    df_caixa_total = df_caixa_total.with_columns(
        (pl.col('caixa') + pl.col('aplicacoes_financeiras')).alias('caixa_total')
    ).select('dt_refer', 'caixa_total')

    return df_caixa_total


def indicador_divida_liquida(df_caixa_total: pl.DataFrame, df_divida_bruta: pl.DataFrame) -> pl.DataFrame:
    """
    Calcula a dívida (caixa) líquida da empresa.

    Parameters
    ----------
    df_divida_bruta: pl.DataFrame
        DataFrame da dívida bruta da empresa.
    df_caixa_total: pl.DataFrame
        DataFrame do caixa total da empresa.

    Returns
    -------
    df_endividamento_líquido: pl.DataFrame
        DataFrame da dívida (caixa) líquida da empresa.
    """
    # Juntando os dfs
    df_endividamento_liquido = df_caixa_total.join(df_divida_bruta, on='dt_refer', how='inner')

    # Calculando a dívida líquida
    df_endividamento_liquido = df_endividamento_liquido.with_columns(
        (pl.col('divida_bruta') - pl.col('caixa_total')).alias('divida_liquida')
    ).select('dt_refer', 'divida_liquida')

    return df_endividamento_liquido


def indicador_ev_ebitda(df_vm: pl.DataFrame, df_divida_liquida: pl.DataFrame, df_ebitda: pl.DataFrame) -> pl.DataFrame:
    """
    Calcula o EV/EBITDA da empresa.

    Parameters
    ----------
    df_vm: pl.DataFrame
        DataFrame do valor de mercado da empresa.
    df_divida_liquida: pl.DataFrame
        DataFrame da dívida (caixa) líquida da empresa.
    df_ebitda: pl.DataFrame
        DataFrame do ebitda da empresa.

    Returns
    -------
    df_ev_ebitda: pl.DataFrame
        DataFrame do EV/EBITDA da empresa.
    """
    # Lista dos dfs para juntar em apenas um df
    lst_df = [
        df_vm,
        df_divida_liquida,
        df_ebitda
    ]

    # Juntando os dfs
    df_ev_ebitda = reduce(
        lambda left, right: left.join(right, on='dt_refer', how='inner'), lst_df
    )

    # Calculando o EV/EBITDA
    df_ev_ebitda = df_ev_ebitda.with_columns(
        ((pl.col('valor_mercado') + (pl.col('divida_liquida')*1000)) / (pl.col('ebitda')*1000)).alias('ev_ebitda')
    ).select('dt_refer', 'ev_ebitda')

    return df_ev_ebitda


def indicador_vpa(df_bpp: pl.DataFrame, df_capital_social: pl.DataFrame, cod_cvm: int, cod_pl: str, novo_mercado: bool) -> pl.DataFrame:
    """
    Calcula o valor patrimonial por ação (vpa) da empresa. 

    Parameters
    ----------
    df_bpp: pl.DataFrame
        DataFrame do balanço patrimonial passivo.
    df_capital_social: pl.DataFrame
        DataFrame que contém o nº de ações ordinárias e preferenciais.
    cod_cvm: int
        Código da empresa na B3.
    cod_pl: str
        Código do patrimônio líquido no bpp.
    novo_mercado: bool
        A empresa é do novo mercado (True), se não é do novo mercado (False).

    Returns
    -------
    df_vpa: pl.DataFrame
        DataFrame do valor patrimonial por ação da empresa.
    """
    # Selecionando o bpp
    bpp_empresa = df_bpp.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o patrimônio líquido
    patrimonio_liquido = bpp_empresa.filter(pl.col('cd_conta') == cod_pl).select('dt_refer', 'vl_conta')

    # Juntando os dfs
    df_vpa = patrimonio_liquido.join(df_capital_social, on='dt_refer', how='inner')

    # Empresas do novo mercado
    if novo_mercado:
        # Calculando o vpa
        df_vpa = df_vpa.with_columns(
            ((pl.col('vl_conta')*1000) / pl.col('num_on')).alias('vpa')
        ).select('dt_refer', 'vpa')

    # Empresas que não são do novo mercado
    else:
        # Calculando o vpa
        df_vpa = df_vpa.with_columns(
            ((pl.col('vl_conta')*1000) / pl.col('num_total')).alias('vpa')
        ).select('dt_refer', 'vpa')

    return df_vpa


def indicador_p_vpa(df_fechamento: pl.DataFrame, tipo_acao: str, df_vpa: pl.DataFrame) -> pl.DataFrame:
    """
    Calcula o indicador P/VP da empresa.

    Parameters
    ----------
    df_fechamento: pl.DataFrame
        DataFrame que contém os preços de fechamento da empresa.
    tipo_acao: str
        'on' (ação ordinária), 'pn' (ação preferencial) e 'unit' (units).
    df_vpa: pl.DataFrame
        DataFrame do VPA da empresa.

    Returns
    -------
    df_p_vp: pl.DataFrame
        DataFrame do P/VP da empresa.
    """
    # Mapeando o tipo de ação para o nome da coluna de fechamento
    coluna_fechamento_map = {
        'on': 'close_on',
        'pn': 'close_pn',
        'unit': 'close_unit'
    }

    try:
        coluna_fechamento = coluna_fechamento_map[tipo_acao]
    except KeyError:
        raise ValueError(f"Tipo de ação '{tipo_acao}' inválido. Use 'on', 'pn' ou 'unit'.")

    # Juntando os DataFrames
    df_p_vp = df_fechamento.join(df_vpa, on='dt_refer', how='inner')

    # Calculando o P/VP
    df_p_vp = df_p_vp.with_columns(
        (pl.col(coluna_fechamento) / pl.col('vpa')).alias('p/vp'),
    ).select('dt_refer', 'p/vp')

    return df_p_vp


def indicador_ml(df_dre: pl.DataFrame, cod_cvm: int, cod_ll: str, cod_rl: str) -> pl.DataFrame:
    """
    Calcula a margem líquida da empresa.

    Parameters
    ----------
    df_dre: pl.DataFrame
        DataFrame da demonstração do resultado do exercício.
    cod_cvm: int
        Código da empresa na B3.
    cod_ll: str
        Código do lucro líquido na dre.
    cod_rl: str
        Código da receita líquida na dre.

    Returns
    -------
    df_margem_liquida: pl.DataFrame
        DataFrame da margem líquida da empresa.
    """
    # Selecionando a dre
    dre_empresa = df_dre.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o lucro líquido
    lucro_liquido = dre_empresa.filter(pl.col('cd_conta') == cod_ll).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    lucro_liquido = lucro_liquido.rename({'vl_conta': 'lucro_liquido'})

    # Selecionando a receita líquida
    receita_liquida = dre_empresa.filter(pl.col('cd_conta') == cod_rl).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    receita_liquida = receita_liquida.rename({'vl_conta': 'receita_liquida'})

    # Juntando os dfs
    df_margem_liquida = lucro_liquido.join(receita_liquida, on='dt_refer', how='inner')

    # Calculando a margem líquida
    df_margem_liquida = df_margem_liquida.with_columns(
        ((pl.col('lucro_liquido') / pl.col('receita_liquida')) * 100).alias('margem_liquida')
    ).select('dt_refer', 'margem_liquida')

    return df_margem_liquida


def indicador_ml_trimestral(df_dre: pl.DataFrame, cod_cvm: int, cod_ll: str, cod_rl: str) -> pl.DataFrame:
    """
    Calcula a margem líquida trimestral da empresa.

    Parameters
    ----------
    df_dre: pl.DataFrame
        DataFrame da demonstração do resultado do exercício.
    cod_cvm: int
        Código da empresa na B3.
    cod_ll: str
        Código do lucro líquido na dre.
    cod_rl: str
        Código da receita líquida na dre.

    Returns
    -------
    df_margem_liquida: pl.DataFrame
        DataFrame da margem líquida trimestral da empresa.
    """
    # Selecionando a dre
    dre_empresa = df_dre.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o lucro líquido
    lucro_liquido = dre_empresa.filter(pl.col('cd_conta') == cod_ll).select('dt_refer', 'dt_ini_exerc', 'vl_conta')

    # Existem linhas duplicadas na coluna 'dt_refer', removendo as linhas duplicadas, mantendo pela data mais recente da 'dt_ini_exerc'
    lucro_liquido = (lucro_liquido
        .sort(['dt_refer', 'dt_ini_exerc'], descending=[False, True])
        .unique(subset='dt_refer', keep='first')
        .sort('dt_refer')
    ).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    lucro_liquido = lucro_liquido.rename({'vl_conta': 'lucro_liquido'})

    # Selecionando a receita líquida
    receita_liquida = dre_empresa.filter(pl.col('cd_conta') == cod_rl).select('dt_refer', 'dt_ini_exerc', 'vl_conta')

    # Existem linhas duplicadas na coluna 'dt_refer', removendo as linhas duplicadas, mantendo pela data mais recente da 'dt_ini_exerc'
    receita_liquida = (receita_liquida
        .sort(['dt_refer', 'dt_ini_exerc'], descending=[False, True])
        .unique(subset='dt_refer', keep='first')
        .sort('dt_refer')
    ).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    receita_liquida = receita_liquida.rename({'vl_conta': 'receita_liquida'})

    # Juntando os dfs
    df_margem_liquida = lucro_liquido.join(receita_liquida, on='dt_refer', how='inner')

    # Calculando a margem líquida
    df_margem_liquida = df_margem_liquida.with_columns(
        ((pl.col('lucro_liquido') / pl.col('receita_liquida')) * 100).alias('margem_liquida')
    ).select('dt_refer', 'margem_liquida')

    return df_margem_liquida


def indicador_roe(df_dre: pl.DataFrame, df_bpp: pl.DataFrame, cod_cvm: int, cod_ll: str, cod_pl: str) -> pl.DataFrame:
    """
    Calcula o ROE da empresa.

    Parameters
    ----------
    df_dre: pl.DataFrame
        DataFrame da demonstração do resultado do exercício.
    df_bpp: pl.DataFrame
        DataFrame do balanço patrimonial passivo.
    cod_cvm: int
        Código da empresa na B3.
    cod_ll: str
        Código do lucro líquido na dre.
    cod_pl: str
        Código do patrimônio líquido no bpp.

    Returns
    -------
    df_roe: pl.DataFrame
        DataFrame do ROE da empresa.
    """
    # Selecionando a dre
    dre_empresa = df_dre.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o lucro líquido
    lucro_liquido = dre_empresa.filter(pl.col('cd_conta') == cod_ll).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    lucro_liquido = lucro_liquido.rename({'vl_conta': 'lucro_liquido'})

    # Selecionando o bpp
    bpp_empresa = df_bpp.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o patrimônio líquido
    patrimonio_liquido = bpp_empresa.filter(pl.col('cd_conta') == cod_pl).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    patrimonio_liquido = patrimonio_liquido.rename({'vl_conta': 'patrimonio_liquido'})

    # Juntando os dfs
    df_roe = lucro_liquido.join(patrimonio_liquido, on='dt_refer', how='inner')

    # Calculando o ROE
    df_roe = df_roe.with_columns(
        ((pl.col('lucro_liquido') / pl.col('patrimonio_liquido')) * 100).alias('roe')
    ).select('dt_refer', 'roe')

    return df_roe


def indicador_roe_trimestral(df_dre: pl.DataFrame, df_bpp: pl.DataFrame, cod_cvm: int, cod_ll: str, cod_pl: str) -> pl.DataFrame:
    """
    Calcula o ROE trimestral da empresa.

    Parameters
    ----------
    df_dre: pl.DataFrame
        DataFrame da demonstração do resultado do exercício.
    df_bpp: pl.DataFrame
        DataFrame do balanço patrimonial passivo.
    cod_cvm: int
        Código da empresa na B3.
    cod_ll: str
        Código do lucro líquido na dre.
    cod_pl: str
        Código do patrimônio líquido no bpp.

    Returns
    -------
    df_roe: pl.DataFrame
        DataFrame do ROE trimestral da empresa.
    """
    # Selecionando a dre
    dre_empresa = df_dre.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o lucro líquido
    lucro_liquido = dre_empresa.filter(pl.col('cd_conta') == cod_ll).select('dt_refer', 'dt_ini_exerc', 'vl_conta')

    # Existem linhas duplicadas na coluna 'dt_refer', removendo as linhas duplicadas, mantendo pela data mais recente da 'dt_ini_exerc'
    lucro_liquido = (lucro_liquido
        .sort(['dt_refer', 'dt_ini_exerc'], descending=[False, True])
        .unique(subset='dt_refer', keep='first')
        .sort('dt_refer')
    ).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    lucro_liquido = lucro_liquido.rename({'vl_conta': 'lucro_liquido'})

    # Calculando o lucro líquido trimestral acumulado
    lucro_liquido_acumulado = lucro_liquido.with_columns(
        pl.col('lucro_liquido')
        .rolling_sum(window_size=4, min_periods=4) 
        .round(2)                                   
        .alias('lucro_liquido_acum')
    )

    # Selecionando o bpp
    bpp_empresa = df_bpp.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o patrimônio líquido
    patrimonio_liquido = bpp_empresa.filter(pl.col('cd_conta') == cod_pl).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    patrimonio_liquido = patrimonio_liquido.rename({'vl_conta': 'patrimonio_liquido'})

    # Juntando os dfs
    df_roe = lucro_liquido_acumulado.join(patrimonio_liquido, on='dt_refer', how='inner')[3:]

    # Calculando o ROE
    df_roe = df_roe.with_columns(
        ((pl.col('lucro_liquido_acum') / pl.col('patrimonio_liquido')) * 100).alias('roe')
    ).select('dt_refer', 'roe')

    return df_roe


def indicador_roic(df_dre: pl.DataFrame, df_bpp: pl.DataFrame, cod_cvm: int, cod_ebit: str, cod_ir: str, cod_pl: str, df_divida_bruta: pl.DataFrame) -> pl.DataFrame:
    """
    Calcula o ROIC da empresa.

    Parameters
    ----------
    df_dre: pl.DataFrame
        DataFrame da demonstração do resultado do exercício.
    df_bpp: pl.DataFrame
        DataFrame do balanço patrimonial passivo.
    cod_cvm: int
        Código da empresa na B3.
    cod_ebit: str
        Código do ebit na dre.
    cod_ir: str
        Código do imposto de renda na dre.approx_n_unique
    cod_pl: str
        Código do patrimônio líquido no bpp.
    df_divida_bruta: pl.DataFrame
        DataFrame da dívida bruta da empresa.

    Returns
    -------
    df_roic: pl.DataFrame
        DataFrame do ROIC da empresa.
    """
    # Selecionando a dre
    dre_empresa = df_dre.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o ebit
    ebit = dre_empresa.filter(pl.col('cd_conta') == cod_ebit).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    ebit = ebit.rename({'vl_conta': 'ebit'})

    # Selecionando o imposto pago
    ir = dre_empresa.filter(pl.col('cd_conta') == cod_ir).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    ir = ir.rename({'vl_conta': 'ir'})

    # Multiplicando por -1
    ir = ir.with_columns(
        (pl.col('ir') * -1)
    )

    # Juntando os dfs
    df_nopat = ebit.join(ir, on='dt_refer', how='inner')

    # Calculando o NOPAT
    df_nopat = df_nopat.with_columns(
        (pl.col('ebit') - pl.col('ir')).alias('nopat')
    ).select('dt_refer', 'nopat')

    # Selecionando o bpp
    bpp_empresa = df_bpp.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o patrimônio líquido
    patrimonio_liquido = bpp_empresa.filter(pl.col('cd_conta') == cod_pl).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    patrimonio_liquido = patrimonio_liquido.rename({'vl_conta': 'patrimonio_liquido'})

    # Juntando os dfs
    df_capital_investido = patrimonio_liquido.join(df_divida_bruta, on='dt_refer', how='inner')

    # Calculando o capital investido
    df_capital_investido = df_capital_investido.with_columns(
        (pl.col('patrimonio_liquido') + pl.col('divida_bruta')).alias('capital_investido')
    ).select('dt_refer', 'capital_investido')

    # Juntando os dfs
    df_roic = df_nopat.join(df_capital_investido, on='dt_refer', how='inner')

    # Calculando o ROIC
    df_roic = df_roic.with_columns(
        ((pl.col('nopat') / pl.col('capital_investido')) * 100).alias('roic')
    ).select('dt_refer', 'roic')

    return df_roic


def indicador_roic_trimestral(df_dre: pl.DataFrame, df_bpp: pl.DataFrame, cod_cvm: int, cod_ebit: str, cod_ir: str, cod_pl: str, df_divida_bruta: pl.DataFrame) -> pl.DataFrame:
    """
    Calcula o ROIC trimestral da empresa.

    Parameters
    ----------
    df_dre: pl.DataFrame
        DataFrame da demonstração do resultado do exercício.
    df_bpp: pl.DataFrame
        DataFrame do balanço patrimonial passivo.
    cod_cvm: int
        Código da empresa na B3.
    cod_ebit: str
        Código do ebit na dre.
    cod_ir: str
        Código do imposto de renda na dre.approx_n_unique
    cod_pl: str
        Código do patrimônio líquido no bpp.
    df_divida_bruta: pl.DataFrame
        DataFrame da dívida bruta da empresa.

    Returns
    -------
    df_roic: pl.DataFrame
        DataFrame do ROIC trimestral da empresa.
    """
    # Selecionando a dre
    dre_empresa = df_dre.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o ebit
    ebit = dre_empresa.filter(pl.col('cd_conta') == cod_ebit).select('dt_refer', 'dt_ini_exerc', 'vl_conta')

    # Existem linhas duplicadas na coluna 'dt_refer', removendo as linhas duplicadas, mantendo pela data mais recente da 'dt_ini_exerc'
    ebit = (ebit
        .sort(['dt_refer', 'dt_ini_exerc'], descending=[False, True])
        .unique(subset='dt_refer', keep='first')
        .sort('dt_refer')
    ).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    ebit = ebit.rename({'vl_conta': 'ebit'})

    # Calculando o ebit acumulado
    ebit_acumulado = ebit.with_columns(
        pl.col('ebit')
        .rolling_sum(window_size=4, min_periods=4) 
        .round(2)                                   
        .alias('ebit_acum')
    )

    # Selecionando o imposto pago
    ir = dre_empresa.filter(pl.col('cd_conta') == cod_ir).select('dt_refer', 'dt_ini_exerc', 'vl_conta')

    # Existem linhas duplicadas na coluna 'dt_refer', removendo as linhas duplicadas, mantendo pela data mais recente da 'dt_ini_exerc'
    ir = (ir
        .sort(['dt_refer', 'dt_ini_exerc'], descending=[False, True])
        .unique(subset='dt_refer', keep='first')
        .sort('dt_refer')
    ).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    ir = ir.rename({'vl_conta': 'ir'})

    # Multiplicando por -1
    ir = ir.with_columns(
        (pl.col('ir') * -1)
    )

    # Calculando o ir acumulado
    ir_acumulado = ir.with_columns(
        pl.col('ir')
        .rolling_sum(window_size=4, min_periods=4) 
        .round(2)                                   
        .alias('ir_acum')
    )

    # Juntando os dfs
    df_nopat = ebit_acumulado.join(ir_acumulado, on='dt_refer', how='inner')

    # Calculando o NOPAT
    df_nopat = df_nopat.with_columns(
        (pl.col('ebit_acum') - pl.col('ir_acum')).alias('nopat')
    ).select('dt_refer', 'nopat')

    # Selecionando o bpp
    bpp_empresa = df_bpp.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o patrimônio líquido
    patrimonio_liquido = bpp_empresa.filter(pl.col('cd_conta') == cod_pl).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    patrimonio_liquido = patrimonio_liquido.rename({'vl_conta': 'patrimonio_liquido'})

    # Juntando os dfs
    df_capital_investido = patrimonio_liquido.join(df_divida_bruta, on='dt_refer', how='inner')

    # Calculando o capital investido
    df_capital_investido = df_capital_investido.with_columns(
        (pl.col('patrimonio_liquido') + pl.col('divida_bruta')).alias('capital_investido')
    ).select('dt_refer', 'capital_investido')

    # Juntando os dfs
    df_roic = df_nopat.join(df_capital_investido, on='dt_refer', how='inner')

    # Calculando o ROIC
    df_roic = df_roic.with_columns(
        ((pl.col('nopat') / pl.col('capital_investido')) * 100).alias('roic')
    ).select('dt_refer', 'roic')

    return df_roic


def indicador_dl_ebitda(df_divida_liquida: pl.DataFrame, df_ebitda: pl.DataFrame) -> pl.DataFrame:
    """
    Calcula o DL/EBITDA da empresa.

    Parameters
    ----------
    df_divida_liquida: pl.DataFrame
        DataFrame da dívida (caixa) líquida da empresa.
    df_ebitda: pl.DataFrame
        DataFrame do ebitda da empresa.

    Returns
    -------
    df_dl_ebitda: pl.DataFrame
        DataFrame do DL/EBITDA da empresa.
    """
    # Juntando os dfs
    df_dl_ebitda = df_divida_liquida.join(df_ebitda, on='dt_refer', how='inner')

    # Calculando o DL/EBITDA
    df_dl_ebitda = df_dl_ebitda.with_columns(
        (pl.col('divida_liquida') / pl.col('ebitda')).alias('dl_ebitda')
    ).select('dt_refer', 'dl_ebitda')

    return df_dl_ebitda


def indicador_dl_pl(df_bpp: pl.DataFrame, cod_cvm: int, cod_pl: str, df_divida_liquida: pl.DataFrame) -> pl.DataFrame:
    """
    Calcula o DL/PL da empresa.

    Parameters
    ----------
    df_bpp: pl.DataFrame
        DataFrame do balanço patrimonial passivo.
    cod_cvm: int
        Código da empresa na B3.
    cod_pl: str
        Código do patrimônio líquido no bpp.
    df_divida_liquida: pl.DataFrame
        DataFrame da dívida (caixa) líquida da empresa.

    Returns
    -------
    df_dl_pl: pl.DataFrame
        DataFrame do DL/PL da empresa.
    """
    # Selecionando o bpp
    bpp_empresa = df_bpp.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o patrimônio líquido
    patrimonio_liquido = bpp_empresa.filter(pl.col('cd_conta') == cod_pl).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    patrimonio_liquido = patrimonio_liquido.rename({'vl_conta': 'patrimonio_liquido'})

    # Juntando os dfs
    df_dl_pl = df_divida_liquida.join(patrimonio_liquido, on='dt_refer', how='inner')

    # Calculando o DL/PL
    df_dl_pl = df_dl_pl.with_columns(
        (pl.col('divida_liquida') / pl.col('patrimonio_liquido')).alias('dl_pl')
    ).select('dt_refer', 'dl_pl')

    return df_dl_pl


def indicador_proventos(df_dva: pl.DataFrame, cod_cvm: int, cod_dividendos: str, cod_jcp: str) -> pl.DataFrame:
    """
    Proventos distribuídos pela empresa.

    Parameters
    ----------
    df_dva: pl.DataFrame
        DataFrame da demonstração do valor adicionado.
    cod_cvm: int
        Código da empresa na B3.
    cod_dividendos: str
        Código dos dividendos no dva.
    cod_jcp: str
        Código do jpc no dva.
    
    Returns
    -------
    df_dividendos: pl.DataFrame
        DataFrame dos proventos distribuídos pela empresa.
    """
    # Selecionando a dva
    dva_empresa = df_dva.filter(pl.col('cd_cvm') == cod_cvm)

    # Inicializando lista de dataframes
    dfs = []

    # Dividendos
    if cod_dividendos is not None:
        dividendos = (
            dva_empresa
            .filter(pl.col('cd_conta') == cod_dividendos)
            .select('dt_refer', pl.col('vl_conta').alias('dividendos'))
        )
        dfs.append(dividendos)

    # JCP
    if cod_jcp is not None:
        jcp = (
            dva_empresa
            .filter(pl.col('cd_conta') == cod_jcp)
            .select('dt_refer', pl.col('vl_conta').alias('jcp'))
        )
        dfs.append(jcp)

    # Se não houver nenhum código, retorna DataFrame vazio
    if not dfs:
        return pl.DataFrame(schema={'dt_refer': pl.Date})

    # Juntando os dfs
    df_proventos = dfs[0]
    for df in dfs[1:]:
        df_proventos = df_proventos.join(df, on='dt_refer', how='inner')

    # Colunas numéricas de proventos
    cols_proventos = [
        c for c in df_proventos.columns if c != 'dt_refer'
    ]

    if len(cols_proventos) == 1:
        # Apenas dividendos OU apenas JCP
        df_proventos = (
            df_proventos
            .rename({cols_proventos[0]: 'proventos'})
        )
    else:
        # Dividendos + JCP
        df_proventos = (
            df_proventos
            .with_columns(
                pl.sum_horizontal(
                    pl.col(cols_proventos).fill_null(0)
                ).alias('proventos')
            )
            .select('dt_refer', 'proventos')
        )

    return df_proventos


def indicador_proventos_trimestral(df_dva: pl.DataFrame, cod_cvm: int, cod_dividendos: str, cod_jcp: str) -> pl.DataFrame:
    """
    Proventos trimestrais distribuídos pela empresa.

    Parameters
    ----------
    df_dva: pl.DataFrame
        DataFrame da demonstração do valor adicionado.
    cod_cvm: int
        Código da empresa na B3.
    cod_dividendos: str
        Código dos dividendos no dva.
    cod_jcp: str
        Código do jpc no dva.
    
    Returns
    -------
    df_dividendos: pl.DataFrame
        DataFrame dos proventos trimestrais distribuídos pela empresa.
    """
    # Selecionando a dva
    dva_empresa = df_dva.filter(pl.col('cd_cvm') == cod_cvm)

    # Inicializando lista de dataframes
    dfs = []

    # Dividendos
    if cod_dividendos is not None:
        dividendos = (
            dva_empresa
            .filter(pl.col('cd_conta') == cod_dividendos)
            .select(['dt_refer', pl.col('vl_conta').alias('dividendos')])
        )

        # Calculando o dividendo não acumulado para os meses 06 e 09
        dividendos_trimestral_nao_acum = dividendos.with_columns([(        
            pl.when(
                (pl.col('dt_refer').dt.month() == 6) | (pl.col('dt_refer').dt.month() == 9)
            )
            .then(
                # Verifica se dividendos é 0 para manter 0. Caso contrário, faz a subtração.
                pl.when(pl.col('dividendos') == 0)
                .then(0)
                .otherwise(pl.col('dividendos') - pl.col('dividendos').shift(1))
            )
            .otherwise(pl.col('dividendos')) 
            .alias('dividendos_nao_acum')
        )])

        dfs.append(dividendos_trimestral_nao_acum)

    # JCP
    if cod_jcp is not None:
        jcp = (
            dva_empresa
            .filter(pl.col('cd_conta') == cod_jcp)
            .select(['dt_refer', pl.col('vl_conta').alias('jcp')])
        )

        # Calculando o jcp não acumulado para os meses 06 e 09
        jcp_trimestral_nao_acum = jcp.with_columns([(
            pl.when(
                (pl.col('dt_refer').dt.month() == 6) | (pl.col('dt_refer').dt.month() == 9)
            )
            .then(
                # Verifica se jcp é 0 para manter 0. Caso contrário, faz a subtração.
                pl.when(pl.col('jcp') == 0)
                .then(0)
                .otherwise(pl.col('jcp') - pl.col('jcp').shift(1))
            )
            .otherwise(pl.col('jcp')) 
            .alias('jcp_nao_acum')
        )])

        dfs.append(jcp_trimestral_nao_acum)

    # Se não houver nenhum código, retorna DataFrame vazio
    if not dfs:
        return pl.DataFrame(schema={'dt_refer': pl.Date, 'proventos': pl.Float64})

    # Juntando os dfs
    df_proventos = dfs[0]
    for df in dfs[1:]:
        df_proventos = df_proventos.join(df, on='dt_refer', how='inner')

    # Soma todos os proventos existentes (dividendos, jcp ou ambos)
    cols_proventos = [
        pl.col(c).fill_null(0)
        for c in df_proventos.columns
        if c.endswith("_nao_acum")
    ]

    df_proventos = (
        df_proventos
        .with_columns(
            sum(cols_proventos).alias("proventos")
        )
        .select("dt_refer", "proventos")
        .sort("dt_refer")
        [3:]
    )

    return df_proventos


def indicador_proventos_dfc(df_dfc: pl.DataFrame, cod_cvm: int, lst_cod_proventos: list) -> pl.DataFrame:
    """
    Proventos distribuídos pela empresa.

    Parameters
    ----------
    df_dfc: pl.DataFrame
        DataFrame do demonstrativo de fluxo de caixa.
    cod_cvm: int
        Código da empresa na B3.
    lst_cod_proventos: list
        Lista dos códigos que formam os proventos (dividendos e JCP) pagos no dfc.

    Returns
    -------
    df_proventos: pl.DataFrame
        DataFrame dos proventos distribuídos pela empresa.
    """
    # Selecionando a dfc
    dfc_empresa = df_dfc.filter(pl.col('cd_cvm') == cod_cvm)

    # Lista do proventos
    lst_proventos = []

    # Iterando sobre os códigos cvm
    for i, cod in enumerate(lst_cod_proventos, start=1):
        proventos = dfc_empresa.filter(pl.col('cd_conta') == cod).select(['dt_refer', pl.col('vl_conta').alias(f'vl_conta_{i}')])
        lst_proventos.append(proventos)
        
    # Fazendo o join de todos os dfs da lista
    df_proventos = lst_proventos[0]
    for df in lst_proventos[1:]:
        df_proventos = df_proventos.join(df, on='dt_refer', how='inner')

    # Calculando o total de proventos (soma das colunas — usar start=pl.lit(0) para evitar 0 + Expr do Python)
    expr_soma_proventos = sum(
        (pl.col(f'vl_conta_{i}') for i in range(1, len(lst_cod_proventos) + 1)),
        pl.lit(0)
    ) * -1

    df_proventos = df_proventos.with_columns(
        (expr_soma_proventos).alias('proventos')
    ).select('dt_refer', 'proventos')

    return df_proventos


def indicador_proventos_dfc_trimestral(df_dfc: pl.DataFrame, cod_cvm: int, lst_cod_proventos: list) -> pl.DataFrame:
    """
    Proventos trimestrais distribuídos pela empresa.

    Parameters
    ----------
    df_dfc: pl.DataFrame
        DataFrame do demonstrativo de fluxo de caixa.
    cod_cvm: int
        Código da empresa na B3.
    lst_cod_proventos: list
        Lista dos códigos que formam os proventos (dividendos e JCP) pagos no dfc.

    Returns
    -------
    df_proventos: pl.DataFrame
        DataFrame dos proventos distribuídos pela empresa.
    """
    # Selecionando a dfc
    dfc_empresa = df_dfc.filter(pl.col('cd_cvm') == cod_cvm)

    # Lista dos proventos
    lst_proventos = []

    # Iterando sobre os códigos cvm
    for i, cod in enumerate(lst_cod_proventos, start=1):
        proventos = (
            dfc_empresa
            .filter(pl.col('cd_conta') == cod)
            .select([
                'dt_refer', 
                pl.col('vl_conta').alias(f'vl_conta_{i}')
            ])
            .sort('dt_refer')  # garante ordem antes do shift
        )
        
        # Calculando o provento não acumulado para os meses 06 e 09 (e mantendo caso contrário)
        expr_proventos_nao_acum = (
            pl.when(
                (pl.col('dt_refer').dt.month() == 6) | (pl.col('dt_refer').dt.month() == 9)
            )
            .then(
                # Verifica se vl_conta_{i} é 0 para manter 0. Caso contrário, faz a subtração.
                pl.when(pl.col(f'vl_conta_{i}') == 0)
                .then(0)
                .otherwise(pl.col(f'vl_conta_{i}') - pl.col(f'vl_conta_{i}').shift(1))
            )
            .otherwise(pl.col(f'vl_conta_{i}')) 
            .alias(f'vl_conta_{i}_nao_acum')
        )

        proventos_nao_acum = proventos.with_columns([expr_proventos_nao_acum]).select('dt_refer', f'vl_conta_{i}_nao_acum').fill_null(0)

        lst_proventos.append(proventos_nao_acum)

    # Se não houver nada, retorna DataFrame vazio
    if not lst_proventos:
        return pl.DataFrame({'dt_refer': [], 'proventos': []})

    # Fazendo o join de todos os dfs da lista
    df_proventos = lst_proventos[0]
    for df in lst_proventos[1:]:
        df_proventos = df_proventos.join(df, on='dt_refer', how='full')

    # Preenchendo os nulos com zero
    df_proventos = df_proventos.fill_null(0)

    # Calculando o total de provento (soma das colunas — usar start=pl.lit(0) para evitar 0 + Expr do Python)
    expr_soma_proventos = sum(
        (pl.col(f'vl_conta_{i}_nao_acum') for i in range(1, len(lst_cod_proventos) + 1)),
        pl.lit(0)
    )

    df_proventos = df_proventos.with_columns(
        (expr_soma_proventos).alias('proventos')
    ).select('dt_refer', 'proventos')

    return df_proventos


def indicador_payout(df_dre: pl.DataFrame, cod_cvm: int, df_proventos: pl.DataFrame, cod_ll: str) -> pl.DataFrame:
    """
    Calcula o payout da empresa.

    Parameters
    ----------
    df_dre: pl.DataFrame
        DataFrame da demonstração do resultado do exercício.
    cod_cvm: int
        Código da empresa na B3.
    cod_ll: str
        Código do lucro líquido na dre.
    df_proventos: pl.DataFrame
        DataFrame dos proventos distribuídos pela empresa.

    Returns
    -------
    df_payout: pl.DataFrame
        DataFrame do payout da empresa.
    """
    # Selecionando a dre
    dre_empresa = df_dre.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o lucro líquido
    lucro_liquido = dre_empresa.filter(pl.col('cd_conta') == cod_ll).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    lucro_liquido = lucro_liquido.rename({'vl_conta': 'lucro_liquido'})

    # Juntando os dfs
    df_payout = df_proventos.join(lucro_liquido, on='dt_refer', how='inner')

    # Calculando o payout
    df_payout = df_payout.with_columns(
        ((pl.col('proventos') / pl.col('lucro_liquido')) * 100).alias('payout')
    ).select('dt_refer', 'payout')

    return df_payout


def indicador_payout_trimestral(df_dre: pl.DataFrame, cod_cvm: int, df_proventos: pl.DataFrame, cod_ll: str) -> pl.DataFrame:
    """
    Calcula o payout trimestral da empresa.

    Parameters
    ----------
    df_dre: pl.DataFrame
        DataFrame da demonstração do resultado do exercício.
    cod_cvm: int
        Código da empresa na B3.
    cod_ll: str
        Código do lucro líquido na dre.
    df_proventos: pl.DataFrame
        DataFrame dos proventos distribuídos pela empresa.

    Returns
    -------
    df_payout: pl.DataFrame
        DataFrame do payout trimestral da empresa.
    """
    # Selecionando a dre
    dre_empresa = df_dre.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o lucro líquido
    lucro_liquido = dre_empresa.filter(pl.col('cd_conta') == cod_ll).select('dt_refer', 'dt_ini_exerc', 'vl_conta')

    # Existem linhas duplicadas na coluna 'dt_refer', removendo as linhas duplicadas, mantendo pela data mais recente da 'dt_ini_exerc'
    lucro_liquido = (lucro_liquido
        .sort(['dt_refer', 'dt_ini_exerc'], descending=[False, True])
        .unique(subset='dt_refer', keep='first')
        .sort('dt_refer')
    ).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    lucro_liquido = lucro_liquido.rename({'vl_conta': 'lucro_liquido'})

    # Juntando os dfs
    df_payout = df_proventos.join(lucro_liquido, on='dt_refer', how='inner')

    # Calculando o payout
    df_payout = df_payout.with_columns(
        ((pl.col('proventos') / pl.col('lucro_liquido')) * 100).alias('payout')
    ).select('dt_refer', 'payout')

    return df_payout


def indicador_buyback(df_dfc: pl.DataFrame, cod_cvm: int, cod_buyback: str) -> pl.DataFrame:
    """
    Buyback feito pela empresa.

    Parameters
    ----------
    df_dfc: pl.DataFrame
        DataFrame do demonstrativo de fluxo de caixa.
    cod_cvm: int
        Código da empresa na B3.
    cod_buyback: str
        Código do buyback na dfc.

    Returns
    -------
    df_buyback: pl.DataFrame
        DataFrame do buyback feito pela empresa.
    """
    # Selecionando a dfc
    dfc_empresa = df_dfc.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o buyback
    buyback = dfc_empresa.filter(pl.col('cd_conta') == cod_buyback).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    df_buyback = buyback.rename({'vl_conta': 'buyback'})

    # Multiplicando por -1
    df_buyback = df_buyback.with_columns(
        (pl.col('buyback') * -1)
    )

    return df_buyback


def indicador_buyback_trimestral(df_dfc: pl.DataFrame, cod_cvm: int, cod_buyback: str) -> pl.DataFrame:
    """
    Buyback trimestral não acumulado feito pela empresa.

    Parameters
    ----------
    df_dfc: pl.DataFrame
        DataFrame do demonstrativo de fluxo de caixa.
    cod_cvm: int
        Código da empresa na B3.
    cod_buyback: str
        Código do buyback na dfc.

    Returns
    -------
    df_buyback_nao_acum: pl.DataFrame
        DataFrame do buyback trimestral não acumulado feito pela empresa.
    """
    # Selecionando a dfc
    dfc_empresa = df_dfc.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o buyback
    buyback = dfc_empresa.filter(pl.col('cd_conta') == cod_buyback).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    buyback = buyback.rename({'vl_conta': 'buyback'})

    # Multiplicando por -1
    buyback = buyback.with_columns(
        (pl.col('buyback') * -1)
    )

    # Calculando o buyback não acumulado para os meses 06 e 09
    df_buyback_nao_acum = buyback.with_columns([(
        pl.when(
            (pl.col('dt_refer').dt.month() == 6) | (pl.col('dt_refer').dt.month() == 9)
        )
        .then(
            # Verifica se buyback é 0 para manter 0. Caso contrário, faz a subtração.
            pl.when(pl.col('buyback') == 0)
            .then(0)
            .otherwise(pl.col('buyback') - pl.col('buyback').shift(1))
        )
        .otherwise(pl.col('buyback')) 
        .alias('buyback_nao_acum')
    )]).select('dt_refer', 'buyback_nao_acum').fill_null(0)

    return df_buyback_nao_acum


def indicador_fco(df_dfc: pl.DataFrame, cod_cvm: int, cod_fco: str) -> pl.DataFrame:
    """
    FCO da empresa.

    Parameters
    ----------
    df_dfc: pl.DataFrame
        DataFrame do demonstrativo de fluxo de caixa.
    cod_cvm: int
        Código da empresa na B3.
    cod_fco: str
        Código do FCO na dfc.

    Returns
    -------
    df_fco: pl.DataFrame
        DataFrame do FCO da empresa.
    """
    # Selecionando a dfc
    dfc_empresa = df_dfc.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o fco
    fco = dfc_empresa.filter(pl.col('cd_conta') == cod_fco).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    df_fco = fco.rename({'vl_conta': 'fco'})

    return df_fco


def indicador_fci(df_dfc: pl.DataFrame, cod_cvm: int, cod_fci: str) -> pl.DataFrame:
    """
    FCI da empresa.

    Parameters
    ----------
    df_dfc: pl.DataFrame
        DataFrame do demonstrativo de fluxo de caixa.
    cod_cvm: int
        Código da empresa na B3.
    cod_fci: str
        Código do FCI na dfc.

    Returns
    -------
    df_fci: pl.DataFrame
        DataFrame do FCI da empresa.
    """
    # Selecionando a dfc
    dfc_empresa = df_dfc.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o fci
    fci = dfc_empresa.filter(pl.col('cd_conta') == cod_fci).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    df_fci = fci.rename({'vl_conta': 'fci'})

    return df_fci


def indicador_fcf(df_dfc: pl.DataFrame, cod_cvm: int, cod_fcf: str) -> pl.DataFrame:
    """
    FCF da empresa.

    Parameters
    ----------
    df_dfc: pl.DataFrame
        DataFrame do demonstrativo de fluxo de caixa.
    cod_cvm: int
        Código da empresa na B3.
    cod_fcf: str
        Código do FCF na dfc.

    Returns
    -------
    df_fcf: pl.DataFrame
        DataFrame do FCF da empresa.
    """
    # Selecionando a dfc
    dfc_empresa = df_dfc.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o fcf
    fcf = dfc_empresa.filter(pl.col('cd_conta') == cod_fcf).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    df_fcf = fcf.rename({'vl_conta': 'fcf'})

    return df_fcf


def indicador_fco_trimestral(df_dfc: pl.DataFrame, cod_cvm: int, cod_fco: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    FCO trimestral da empresa.

    Parameters
    ----------
    df_dfc: pl.DataFrame
        DataFrame do demonstrativo de fluxo de caixa.
    cod_cvm: int
        Código da empresa na B3.
    cod_fco: str
        Código do FCO na dfc.

    Returns
    -------
    df_fco: pl.DataFrame
        DataFrame do FCO trimestral não acumulado da empresa.
    df_fco_acum: pl.DataFrame
        DataFrame do FCO trimestral acumulado.
    """
    # Selecionando a dfc
    dfc_empresa = df_dfc.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o fco
    fco = dfc_empresa.filter(pl.col('cd_conta') == cod_fco).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    fco = fco.rename({'vl_conta': 'fco'})

    # Calculando o fco não acumulado para os meses 06 e 09
    expr_fco_nao_acum = (
        pl.when(
            (pl.col('dt_refer').dt.month() == 6) | (pl.col('dt_refer').dt.month() == 9)
        )
        .then(
            # Verifica se fco é 0 para manter 0. Caso contrário, faz a subtração.
            pl.when(pl.col('fco') == 0)
            .then(0)
            .otherwise(pl.col('fco') - pl.col('fco').shift(1))
        )
        .otherwise(pl.col('fco')) 
        .alias('fco_nao_acum')
    )

    df_fco_nao_acum = fco.with_columns([expr_fco_nao_acum]).select('dt_refer', 'fco_nao_acum').fill_null(0)

    # Calculando o fco acumulado
    expr_fcf_acum = (
        pl.col('fco_nao_acum')
        .rolling_sum(window_size=4, min_periods=4) 
        .round(2)                                   
        .alias('fco')
    )

    df_fco_acum = df_fco_nao_acum.with_columns([expr_fcf_acum]).select('dt_refer', 'fco').fill_null(0)

    return df_fco_nao_acum, df_fco_acum


def indicador_fci_trimestral(df_dfc: pl.DataFrame, cod_cvm: int, cod_fci: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    FCI trimestral da empresa.

    Parameters
    ----------
    df_dfc: pl.DataFrame
        DataFrame do demonstrativo de fluxo de caixa.
    cod_cvm: int
        Código da empresa na B3.
    cod_fci: str
        Código do FCI na dfc.

    Returns
    -------
    df_fci: pl.DataFrame
        DataFrame do FCI trimestral não acumulado da empresa.
    df_fci_acum: pl.DataFrame
        DataFrame do FCI trimestral acumulado.
    """
    # Selecionando a dfc
    dfc_empresa = df_dfc.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o fci
    fci = dfc_empresa.filter(pl.col('cd_conta') == cod_fci).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    fci = fci.rename({'vl_conta': 'fci'})

    # Calculando o fci não acumulado para os meses 06 e 09
    expr_fci_nao_acum = (
        pl.when(
            (pl.col('dt_refer').dt.month() == 6) | (pl.col('dt_refer').dt.month() == 9)
        )
        .then(
            # Verifica se fci é 0 para manter 0. Caso contrário, faz a subtração.
            pl.when(pl.col('fci') == 0)
            .then(0)
            .otherwise(pl.col('fci') - pl.col('fci').shift(1))
        )
        .otherwise(pl.col('fci')) 
        .alias('fci_nao_acum')
    )

    df_fci_nao_acum = fci.with_columns([expr_fci_nao_acum]).select('dt_refer', 'fci_nao_acum').fill_null(0)

    # Calculando o fci acumulado
    expr_fcf_acum = (
        pl.col('fci_nao_acum')
        .rolling_sum(window_size=4, min_periods=4) 
        .round(2)                                   
        .alias('fci')
    )

    df_fci_acum = df_fci_nao_acum.with_columns([expr_fcf_acum]).select('dt_refer', 'fci').fill_null(0)

    return df_fci_nao_acum, df_fci_acum


def indicador_fcf_trimestral(df_dfc: pl.DataFrame, cod_cvm: int, cod_fcf: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    FCF trimestral da empresa.

    Parameters
    ----------
    df_dfc: pl.DataFrame
        DataFrame do demonstrativo de fluxo de caixa.
    cod_cvm: int
        Código da empresa na B3.
    cod_fcf: str
        Código do FCF na dfc.

    Returns
    -------
    df_fcf: pl.DataFrame
        DataFrame do FCF trimestral não acumulado da empresa.
    df_fcf_acum: pl.DataFrame
        DataFrame do FCF trimestral acumulado.
    """
    # Selecionando a dfc
    dfc_empresa = df_dfc.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o fcf
    fcf = dfc_empresa.filter(pl.col('cd_conta') == cod_fcf).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    fcf = fcf.rename({'vl_conta': 'fcf'})

    # Calculando o fcf não acumulado para os meses 06 e 09
    expr_fcf_nao_acum = (
        pl.when(
            (pl.col('dt_refer').dt.month() == 6) | (pl.col('dt_refer').dt.month() == 9)
        )
        .then(
            # Verifica se fcf é 0 para manter 0. Caso contrário, faz a subtração.
            pl.when(pl.col('fcf') == 0)
            .then(0)
            .otherwise(pl.col('fcf') - pl.col('fcf').shift(1))
        )
        .otherwise(pl.col('fcf')) 
        .alias('fcf_nao_acum')
    )

    df_fcf_nao_acum = fcf.with_columns([expr_fcf_nao_acum]).select('dt_refer', 'fcf_nao_acum').fill_null(0)

    # Calculando o fcf acumulado
    expr_fcf_acum = (
        pl.col('fcf_nao_acum')
        .rolling_sum(window_size=4, min_periods=4) 
        .round(2)                                   
        .alias('fcf')
    )

    df_fcf_acum = df_fcf_nao_acum.with_columns([expr_fcf_acum]).select('dt_refer', 'fcf').fill_null(0)

    return df_fcf_nao_acum, df_fcf_acum 


def indicador_capex(df_dfc: pl.DataFrame, cod_cvm: int, lst_cod_capex: list) -> pl.DataFrame:
    """
    Calcula o capex da empresa.

    Parameters
    ----------
    df_dfc: pl.DataFrame
        DataFrame do demonstrativo de fluxo de caixa.
    cod_cvm: int
        Código da empresa na B3.
    lst_cod_capex: list
        Lista dos códigos que formam o capex na dfc.

    Returns
    -------
    df_capex: pl.DataFrame
        DataFrame do capex da empresa.
    """
    # Selecionando a dfc
    dfc_empresa = df_dfc.filter(pl.col('cd_cvm') == cod_cvm)

    # Lista do capex
    lst_capex = []

    # Iterando sobre os códigos cvm
    for i, cod in enumerate(lst_cod_capex, start=1):
        capex = dfc_empresa.filter(pl.col('cd_conta') == cod).select(['dt_refer', pl.col('vl_conta').alias(f'vl_conta_{i}')])
        lst_capex.append(capex)

        # Fazendo o join de todos os dfs da lista
        df_capex = lst_capex[0]
        for df in lst_capex[1:]:
            df_capex = df_capex.join(df, on='dt_refer', how='inner')

    # Calcula o total (com sinal invertido)
    df_capex = df_capex.with_columns(
        (sum(pl.col(f'vl_conta_{i}') for i in range(1, len(lst_cod_capex) + 1)) * -1).alias('capex')
    ).select('dt_refer', 'capex')

    return df_capex


def indicador_capex_trimestral(df_dfc: pl.DataFrame, cod_cvm: int, lst_cod_capex: list) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Calcula o capex trimestral da empresa.

    Parameters
    ----------
    df_dfc: pl.DataFrame
        DataFrame do demonstrativo de fluxo de caixa.
    cod_cvm: int
        Código da empresa na B3.
    lst_cod_capex: list
        Lista dos códigos que formam o capex na dfc.

    Returns
    -------
    df_capex_nao_acum: pl.DataFrame
        DataFrame do capex trimestral da empresa.
    df_capex_acum: pl.DataFrame
        DataFrame do capex trimestral acumulado da empresa.
    """
    # Selecionando a dfc
    dfc_empresa = df_dfc.filter(pl.col('cd_cvm') == cod_cvm)

    # Lista do capex
    lst_capex_nao_acum = []
    lst_capex_acum = []

    # Iterando sobre os códigos do capex
    for i, cod in enumerate(lst_cod_capex, start=1):
        capex = (
            dfc_empresa
            .filter(pl.col('cd_conta') == cod)
            .select([
                'dt_refer',
                pl.col('vl_conta').alias(f'vl_conta_{i}')
            ])
            .sort('dt_refer')  # garante ordem antes do shift
        )

        # Calculando o capex não acumulado para os meses 06 e 09 (e mantendo caso contrário)
        expr_nao_acum = (
            pl.when(
                (pl.col('dt_refer').dt.month() == 6) | (pl.col('dt_refer').dt.month() == 9)
            )
            .then(
                # Verifica se vl_conta_{i} é 0 para manter 0. Caso contrário, faz a subtração.
                pl.when(pl.col(f'vl_conta_{i}') == 0)
                .then(0)
                .otherwise(pl.col(f'vl_conta_{i}') - pl.col(f'vl_conta_{i}').shift(1))
            )
            .otherwise(pl.col(f'vl_conta_{i}')) 
            .alias(f'vl_conta_{i}_nao_acum')
        )

        capex_nao_acum = capex.with_columns([expr_nao_acum]).select('dt_refer', f'vl_conta_{i}_nao_acum').fill_null(0)

        lst_capex_nao_acum.append(capex_nao_acum)

        # Calculando o capex trimestral acumulado
        expr_acum = (
            pl.col(f'vl_conta_{i}_nao_acum')
            .rolling_sum(window_size=4, min_periods=4) 
            .round(2)                                   
            .alias(f'vl_conta_{i}_acum')
        )

        capex_acum = capex_nao_acum.with_columns([expr_acum]).select('dt_refer', f'vl_conta_{i}_acum').fill_null(0)

        lst_capex_acum.append(capex_acum)

    # Se não houver nada, retorna DataFrame vazio
    if not lst_capex_nao_acum:
        return pl.DataFrame({'dt_refer': [], 'capex': []})
    
    if not lst_capex_acum:
        return pl.DataFrame({'dt_refer': [], 'capex': []})
    
    # Fazendo o join de todos os dfs da lista (inner join por dt_refer)
    df_capex_nao_acum = lst_capex_nao_acum[0]
    for df in lst_capex_nao_acum[1:]:
        df_capex_nao_acum = df_capex_nao_acum.join(df, on='dt_refer', how='inner')

    df_capex_acum = lst_capex_acum[0]
    for df in lst_capex_acum[1:]:
        df_capex_acum = df_capex_acum.join(df, on='dt_refer', how='inner')

    # # Calculando o total do capex (soma das colunas — usar start=pl.lit(0) para evitar 0 + Expr do Python)
    expr_soma_nao_acum = sum(
        (pl.col(f'vl_conta_{i}_nao_acum') for i in range(1, len(lst_cod_capex) + 1)),
        pl.lit(0)
    )
    
    expr_soma_acum = sum(
        (pl.col(f'vl_conta_{i}_acum') for i in range(1, len(lst_cod_capex) + 1)),
        pl.lit(0)
    )

    df_capex_nao_acum = df_capex_nao_acum.with_columns(
        (expr_soma_nao_acum * -1).alias('capex')
    ).select('dt_refer', 'capex')

    df_capex_acum = df_capex_acum.with_columns(
        (expr_soma_acum * -1).alias('capex')
    ).select('dt_refer', 'capex')

    return df_capex_nao_acum, df_capex_acum


def indicador_free_cash_flow(df_fco: pl.DataFrame, df_capex: pl.DataFrame) -> pl.DataFrame:
    """
    Calcula o free cash flow da empresa.

    Parameters
    ----------
    df_fco: pl.DataFrame
        DataFrame do FCO da empresa.
    df_capex: pl.DataFrame
        DataFrame do capex da empresa.

    Returns
    -------
    df_free_cash_flow: pl.DataFrame
        DataFrame do free cash flow da empresa.
    """
    # Juntando os dfs
    df_free_cash_flow = df_fco.join(df_capex, on='dt_refer', how='inner')

    # Calculando o free cash flow
    df_free_cash_flow = df_free_cash_flow.with_columns(
        (pl.col('fco') - pl.col('capex')).alias('free_cash_flow')
    ).select('dt_refer', 'free_cash_flow')

    return df_free_cash_flow


def indicador_net_capex(df_dfc: pl.DataFrame, cod_cvm: int, cod_depreciacao: str, df_capex: pl.DataFrame) -> pl.DataFrame:
    """
    Calcula o net capex da empresa.

    Parameters
    ----------
    df_dfc: pl.DataFrame
        DataFrame do demonstrativo de fluxo de caixa.
    cod_cvm: int
        Código da empresa na B3.
    cod_depreciacao: str
        Código da depreciação na dfc.
    df_capex: pl.DataFrame
        DataFrame do capex da empresa.

    Returns
    -------
    df_net_capex: pl.DataFrame
        DataFrame do net capex da empresa.
    """
    # Selecionando a dfc
    dfc_empresa = df_dfc.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando a depreciação
    depreciacao = dfc_empresa.filter(pl.col('cd_conta') == cod_depreciacao).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    depreciacao = depreciacao.rename({'vl_conta': 'depreciacao'})

    # Juntando os dfs
    df_net_capex = df_capex.join(depreciacao, on='dt_refer', how='inner')

    # Calculando o net capex
    df_net_capex = df_net_capex.with_columns(
        (pl.col('capex') - pl.col('depreciacao')).alias('net_capex')
    ).select('dt_refer', 'net_capex')

    return df_net_capex


def indicador_net_capex_trimestral(df_dfc: pl.DataFrame, cod_cvm: int, cod_depreciacao: str, df_capex: pl.DataFrame) -> pl.DataFrame:
    """
    Calcula o net capex trimestral da empresa.

    Parameters
    ----------
    df_dfc: pl.DataFrame
        DataFrame do demonstrativo de fluxo de caixa.
    cod_cvm: int
        Código da empresa na B3.
    cod_depreciacao: str
        Código da depreciação na dfc.
    df_capex: pl.DataFrame
        DataFrame do capex da empresa.

    Returns
    -------
    df_net_capex: pl.DataFrame
        DataFrame do net capex trimestral da empresa.
    """
    # Selecionando a dfc
    dfc_empresa = df_dfc.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando a depreciação
    depreciacao = dfc_empresa.filter(pl.col('cd_conta') == cod_depreciacao).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    depreciacao = depreciacao.rename({'vl_conta':'depreciacao'})

    # Calculando a depreciaçao não acumulada para os meses 06 e 09
    expr_deprec_nao_acum = (
        pl.when(
            (pl.col('dt_refer').dt.month() == 6) | (pl.col('dt_refer').dt.month() == 9)
        )
        .then(
            # Verifica se depreciacao é 0 para manter 0. Caso contrário, faz a subtração.
            pl.when(pl.col('depreciacao') == 0)
            .then(0)
            .otherwise(pl.col('depreciacao') - pl.col('depreciacao').shift(1))
        )
        .otherwise(pl.col('depreciacao')) 
        .alias('depreciacao_nao_acum')
    )

    depreciacao_nao_acum = depreciacao.with_columns([expr_deprec_nao_acum]).select('dt_refer', 'depreciacao_nao_acum')

    # Calculando a depreciação acumulada
    expr_depreciacao_acum = (
        pl.col('depreciacao_nao_acum')
        .rolling_sum(window_size=4, min_periods=4) 
        .round(2)                                   
        .alias('depreciacao_acum')
    )

    df_depreciacao_acum = depreciacao_nao_acum.with_columns([expr_depreciacao_acum]).select('dt_refer', 'depreciacao_acum').fill_null(0)

    # Juntando os dfs
    df_net_capex = df_capex.join(df_depreciacao_acum, on='dt_refer', how='inner')

    # Calculando o net capex
    df_net_capex = df_net_capex.with_columns(
        (pl.col('capex') - pl.col('depreciacao_acum')).alias('net_capex')
    ).select('dt_refer', 'net_capex')

    return df_net_capex


def indicador_rd(df_dre: pl.DataFrame, cod_cvm: int, cod_rd: str, anual: bool) -> pl.DataFrame:
    """
    Custo de pesquisa e desenvolvimento (P&D) da empresa.

    Parameters
    ----------
    df_dre: pl.DataFrame
        DataFrame da demonstração do resultado do exercício.
    cod_cvm: int
        Código da empresa na B3.
    cod_rd: str
        Código da P&D na dre.
    anual: se o dado for anual (True), so for trimestral (False).

    Returns
    -------
    df_rd_anual: pl.DataFrame
        DataFrame do R&D anual.
    df_rd_trimestral: pl.DataFrame
        DataFrame do R&D trimestral.
    df_rd_trimestral_acum: pl.DataFrame
        DataFrame do R&D trimestral acumulado.
    """
    # Anual
    if anual:
        # Selecionando a dre
        dre_empresa = df_dre.filter(pl.col('cd_cvm') == cod_cvm)

        # Selecionando o rd
        rd = dre_empresa.filter(pl.col('cd_conta') == cod_rd).select('dt_refer', 'vl_conta')

        # Renomeando a coluna
        df_rd_anual = rd.rename({'vl_conta': 'rd'})

        # Multiplicando por -1
        df_rd_anual = df_rd_anual.with_columns(
            (pl.col('rd') * -1)
        )

        return df_rd_anual
    
    # Trimestral
    else:
        # Selecionando a dre
        dre_empresa = df_dre.filter(pl.col('cd_cvm') == cod_cvm)

        # Selecionando o rd
        rd = dre_empresa.filter(pl.col('cd_conta') == cod_rd).select('dt_refer', 'dt_ini_exerc', 'vl_conta')

        # Existem linhas duplicadas na coluna 'dt_refer', removendo as linhas duplicadas, mantendo pela data mais recente da 'dt_ini_exerc'
        rd = (rd
            .sort(['dt_refer', 'dt_ini_exerc'], descending=[False, True])
            .unique(subset='dt_refer', keep='first')
            .sort('dt_refer')
        ).select('dt_refer', 'vl_conta')

        # Renomeando a coluna
        df_rd_trimestral = rd.rename({'vl_conta': 'rd'})

        # Multiplicando por -1
        df_rd_trimestral = df_rd_trimestral.with_columns(
            (pl.col('rd') * -1)
        )

        # Calculando o rd trimestral acumulado
        expr_rd_acum = (
            pl.col('rd')
            .rolling_sum(window_size=4, min_periods=4) 
            .round(2)                                   
        )

        df_rd_trimestral_acum = df_rd_trimestral.with_columns([expr_rd_acum]).select('dt_refer', 'rd').fill_null(0)


        return df_rd_trimestral, df_rd_trimestral_acum
    

def indicador_adjusted_net_capex(df_net_capex: pl.DataFrame, df_rd: pl.DataFrame) -> pl.DataFrame:
    """
    Calcula o adjusted net capex da empresa.

    Parameters
    ----------
    df_net_capex: pl.DataFrame
        DataFrame do net capex da empresa.
    df_rd: pl.DataFrame
        DataFrame de R&D da empresa.

    Returns
    -------
    df_net_capex: pl.DataFrame
        DataFrame do adjusted net capex da empresa.
    """
    # Juntando os dfs
    df_adjusted_net_capex = df_net_capex.join(df_rd, on='dt_refer', how='inner')

    # Calculando o adjusted net capex
    df_adjusted_net_capex = df_adjusted_net_capex.with_columns(
        (pl.col('net_capex') + pl.col('rd')).alias('adjusted_net_capex')
    ).select('dt_refer', 'adjusted_net_capex')

    return df_adjusted_net_capex


def indicador_working_capital(df_bpa: pl.DataFrame, df_bpp: pl.DataFrame, cod_cvm: int, cod_ativo_circ: str, cod_passivo_circ:str) -> pl.DataFrame:
    """
    Calcula o capital de giro da empresa.

    Parameters
    ----------
    df_bpa: pl.DataFrame
        DataFrame do balanço patrimonial ativo. 
    df_bpp: pl.DataFrame
        DataFrame do balanço patrimonial passivo.
    cod_cvm: int
        Código da empresa na B3.
    cod_ativo_circ: str
        Código do ativo circulante no bpa.
    cod_passivo_circ: str
        Código do passivo circulante no bpp.

    Returns
    -------
    df_working_capital: pl.DataFrame
        DataFrame do capital de giro da empresa.
    """
    # Selecionando a bpa
    bpa_empresa = df_bpa.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o passivo circulante
    ativo_circulante = bpa_empresa.filter(pl.col('cd_conta') == cod_ativo_circ).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    ativo_circulante = ativo_circulante.rename({'vl_conta': 'ativo_circulante'})

    # Selecionando a bpp
    bpp_empresa = df_bpp.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o passivo circulante
    passivo_circulante = bpp_empresa.filter(pl.col('cd_conta') == cod_passivo_circ).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    passivo_circulante = passivo_circulante.rename({'vl_conta': 'passivo_circulante'})

    # Juntando os dfs
    df_working_capital = ativo_circulante.join(passivo_circulante, on='dt_refer', how='inner')

    # Calculando o working capital
    df_working_capital = df_working_capital.with_columns(
        (pl.col('ativo_circulante') - pl.col('passivo_circulante')).alias('working_capital')
    ).select('dt_refer', 'working_capital')

    return df_working_capital


def indicador_change_in_non_cash_wc(df_dfc: pl.DataFrame, cod_cvm: int, cod_change_in_non_cash_wc: str) -> pl.DataFrame:
    """
    Variações nos ativos e passivos do fluxo de caixa da empresa.

    Parameters
    ----------
    df_dfc: pl.DataFrame
        DataFrame do demonstrativo de fluxo de caixa.
    cod_cvm: int
        Código da empresa na B3.
    cod_change_in_non_cash_wc: str
        Código das variações nos ativos e passivos na dfc.

    Returns
    -------
    df_change_in_non_cash_wc: pl.DataFrame
        DataFrame das	variações nos ativos e passivos do fluxo de caixa da empresa.
    """
    # Selecionando a dfc
    dfc_empresa = df_dfc.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o change in non-cash wc
    change_in_non_cash_wc = dfc_empresa.filter(pl.col('cd_conta') == cod_change_in_non_cash_wc).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    df_change_in_non_cash_wc = change_in_non_cash_wc.rename({'vl_conta': 'change_in_non_cash_wc'})

    # Multiplicando por -1
    df_change_in_non_cash_wc = df_change_in_non_cash_wc.with_columns(
        (pl.col('change_in_non_cash_wc') * -1)
    )

    return df_change_in_non_cash_wc


def indicador_change_in_non_cash_wc_trimestral(df_dfc: pl.DataFrame, cod_cvm: int, cod_change_in_non_cash_wc: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Variações nos ativos e passivos do fluxo de caixa da empresa.

    Parameters
    ----------
    df_dfc: pl.DataFrame
        DataFrame do demonstrativo de fluxo de caixa.
    cod_cvm: int
        Código da empresa na B3.
    cod_change_in_non_cash_wc: str
        Código das variações nos ativos e passivos na dfc.

    Returns
    -------
    df_change_in_non_cash_wc: pl.DataFrame
        DataFrame das variações nos ativos e passivos do fluxo de caixa da empresa.
    df_change_in_non_cash_wc_acum: pl.DataFrame
        DataFrame das variações nos ativos e passivos do fluxo de caixa acumulado da empresa.
    """
    # Selecionando a dfc
    dfc_empresa = df_dfc.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o change in non-cash wc
    change_in_non_cash_wc = dfc_empresa.filter(pl.col('cd_conta') == cod_change_in_non_cash_wc).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    df_change_in_non_cash_wc = change_in_non_cash_wc.rename({'vl_conta': 'change_in_non_cash_wc'})

    # Multiplicando por -1
    df_change_in_non_cash_wc = df_change_in_non_cash_wc.with_columns(
        (pl.col('change_in_non_cash_wc') * -1)
    )

    # Calculando o change in non-cash wc não acumulado para os meses 06 e 09 (e mantendo caso contrário)
    expr_nao_acum = (
        pl.when(
            (pl.col('dt_refer').dt.month() == 6) | (pl.col('dt_refer').dt.month() == 9)
        )
        .then(
            # Verifica se change_in_non_cash_wc é 0 para manter 0. Caso contrário, faz a subtração.
            pl.when(pl.col('change_in_non_cash_wc') == 0)
            .then(0)
            .otherwise(pl.col('change_in_non_cash_wc') - pl.col('change_in_non_cash_wc').shift(1))
        )
        .otherwise(pl.col('change_in_non_cash_wc')) 
        .alias('change_in_non_cash_wc_nao_acum')
    )

    df_change_in_non_cash_wc_nao_acum = df_change_in_non_cash_wc.with_columns([expr_nao_acum]).select('dt_refer', 'change_in_non_cash_wc_nao_acum').fill_null(0)

    # Calculando o change in non-cash wc trimestral acumulado
    expr_acum = (
        pl.col('change_in_non_cash_wc_nao_acum')
        .rolling_sum(window_size=4, min_periods=4) 
        .round(2)                                   
        .alias('change_in_non_cash_wc')
    )

    df_change_in_non_cash_wc_acum = df_change_in_non_cash_wc_nao_acum.with_columns([expr_acum]).select('dt_refer', 'change_in_non_cash_wc').fill_null(0)

    return df_change_in_non_cash_wc, df_change_in_non_cash_wc_acum


def indicador_reinvestment_rate(df_dre: pl.DataFrame, cod_cvm: int, cod_ebit: str, cod_ir: str, df_net_capex: pl.DataFrame, df_change_in_non_cash_wc: pl.DataFrame) -> pl.DataFrame:
    """
    Calcula a taxa de reinvestimento da empresa.

    Parameters
    ----------
    df_dre: pl.DataFrame
        DataFrame da demonstração do resultado do exercício.
    cod_cvm: int
        Código da empresa na B3.
    cod_ebit: str
        Código do ebit na dre.
    cod_ir: str
        Código do imposto de renda na dre.
    df_net_capex: pl.DataFrame
        DataFrame do net capex da empresa.
    df_change_in_non_cash_wc: pl.DataFrame
        DataFrame das	variações nos ativos e passivos da empresa.

    Returns
    -------
    df_reinvestment_rate: pl.DataFrame
        DataFrame da taxa de reinvestimento da empresa.
    """
    # Selecionando a dre
    dre_empresa = df_dre.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o ebit
    ebit = dre_empresa.filter(pl.col('cd_conta') == cod_ebit).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    ebit = ebit.rename({'vl_conta': 'ebit'})

    # Selecionando o imposto pago
    ir = dre_empresa.filter(pl.col('cd_conta') == cod_ir).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    ir = ir.rename({'vl_conta': 'ir'})

    # Multiplicando por -1
    ir = ir.with_columns(
        (pl.col('ir') * -1)
    )

    # Juntando os dfs
    df_reinvestment_rate_denominator = ebit.join(ir, on='dt_refer', how='inner')

    # Calculando o denominador 
    df_reinvestment_rate_denominator = df_reinvestment_rate_denominator.with_columns(
        (pl.col('ebit') - pl.col('ir')).alias('reinvestment_rate_denominator')
    ).select('dt_refer', 'reinvestment_rate_denominator')

    # Juntando os dfs
    df_reinvestment_rate_numerator = df_net_capex.join(df_change_in_non_cash_wc, on='dt_refer', how='inner')

    # Calculando o numerador
    df_reinvestment_rate_numerator = df_reinvestment_rate_numerator.with_columns(
        (pl.col('adjusted_net_capex') + pl.col('change_in_non_cash_wc')).alias('reinvestment_rate_numerator')
    ).select('dt_refer', 'reinvestment_rate_numerator')

    # Juntando os dfs
    df_reinvestment_rate = df_reinvestment_rate_numerator.join(df_reinvestment_rate_denominator, on='dt_refer', how='inner')

    # Calculando o reinvestment rate
    df_reinvestment_rate = df_reinvestment_rate.with_columns(
        (pl.col('reinvestment_rate_numerator') / pl.col('reinvestment_rate_denominator')).alias('reinvestment_rate')
    ).select('dt_refer', 'reinvestment_rate')

    return df_reinvestment_rate


def indicador_reinvestment_rate_trimestral(df_dre: pl.DataFrame, cod_cvm: int, cod_ebit: str, cod_ir: str, df_net_capex: pl.DataFrame, df_change_in_non_cash_wc: pl.DataFrame) -> pl.DataFrame:
    """
    Calcula a taxa de reinvestimento trimestral da empresa.

    Parameters
    ----------
    df_dre: pl.DataFrame
        DataFrame da demonstração do resultado do exercício.
    cod_cvm: int
        Código da empresa na B3.
    cod_ebit: str
        Código do ebit na dre.
    cod_ir: str
        Código do imposto de renda na dre.
    df_net_capex: pl.DataFrame
        DataFrame do net capex da empresa.
    df_change_in_non_cash_wc: pl.DataFrame
        DataFrame das	variações nos ativos e passivos da empresa.

    Returns
    -------
    df_reinvestment_rate: pl.DataFrame
        DataFrame da taxa de reinvestimento trimestral da empresa.
    """
    # Selecionando a dre
    dre_empresa = df_dre.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o ebit
    ebit = dre_empresa.filter(pl.col('cd_conta') == cod_ebit).select('dt_refer', 'dt_ini_exerc', 'vl_conta')

    # Existem linhas duplicadas na coluna 'dt_refer', removendo as linhas duplicadas, mantendo pela data mais recente da 'dt_ini_exerc'
    ebit = (ebit
        .sort(['dt_refer', 'dt_ini_exerc'], descending=[False, True])
        .unique(subset='dt_refer', keep='first')
        .sort('dt_refer')
    ).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    ebit = ebit.rename({'vl_conta': 'ebit'})

    # Calculando o ebit trimestral acumulado
    expr_ebit_acum = (
        pl.col('ebit')
        .rolling_sum(window_size=4, min_periods=4) 
        .round(2)                                   
    )

    df_ebit_acum = ebit.with_columns([expr_ebit_acum]).select('dt_refer', 'ebit').fill_null(0)

    # Selecionando o imposto pago
    ir = dre_empresa.filter(pl.col('cd_conta') == cod_ir).select('dt_refer', 'dt_ini_exerc', 'vl_conta')

    # Existem linhas duplicadas na coluna 'dt_refer', removendo as linhas duplicadas, mantendo pela data mais recente da 'dt_ini_exerc'
    ir = (ir
        .sort(['dt_refer', 'dt_ini_exerc'], descending=[False, True])
        .unique(subset='dt_refer', keep='first')
        .sort('dt_refer')
    ).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    ir = ir.rename({'vl_conta': 'ir'})

    # Calculando o ir trimestral acumulado
    expr_ir_acum = (
        pl.col('ir')
        .rolling_sum(window_size=4, min_periods=4) 
        .round(2)                                   
    )

    df_ir_acum = ir.with_columns([expr_ir_acum]).select('dt_refer', 'ir').fill_null(0)

    # Multiplicando por -1
    df_ir_acum = df_ir_acum.with_columns(
        (pl.col('ir') * -1)
    )

    # Juntando os dfs
    df_reinvestment_rate_denominator = df_ebit_acum.join(df_ir_acum, on='dt_refer', how='inner')

    # Calculando o denominador 
    df_reinvestment_rate_denominator = df_reinvestment_rate_denominator.with_columns(
        (pl.col('ebit') - pl.col('ir')).alias('reinvestment_rate_denominator')
    ).select('dt_refer', 'reinvestment_rate_denominator')

    # Juntando os dfs
    df_reinvestment_rate_numerator = df_net_capex.join(df_change_in_non_cash_wc, on='dt_refer', how='inner')

    # Calculando o numerador
    df_reinvestment_rate_numerator = df_reinvestment_rate_numerator.with_columns(
        (pl.col('adjusted_net_capex') + pl.col('change_in_non_cash_wc')).alias('reinvestment_rate_numerator')
    ).select('dt_refer', 'reinvestment_rate_numerator')

    # Juntando os dfs
    df_reinvestment_rate = df_reinvestment_rate_numerator.join(df_reinvestment_rate_denominator, on='dt_refer', how='inner')

    # Calculando o reinvestment rate
    df_reinvestment_rate = df_reinvestment_rate.with_columns(
        (pl.col('reinvestment_rate_numerator') / pl.col('reinvestment_rate_denominator')).alias('reinvestment_rate')
    ).select('dt_refer', 'reinvestment_rate')

    return df_reinvestment_rate


def indicador_effective_tax_rate(df_dre: pl.DataFrame, cod_cvm: int, cod_ir: str, cod_ebt: str, anual: bool) -> pl.DataFrame:
    """
    Calcula a taxa de imposto de renda efetiva.

    Parameters
    ----------
    df_dre: pl.DataFrame
        DataFrame da demonstração do resultado do exercício.
    cod_cvm: int
        Código da empresa na B3.
    cod_ir: str
        Código do imposto de renda na dre.
    cod_ebt: str
        Código do resultado antes dos tributos sobre o lucro (earnings before tax - ebt) na dre.
    anual: bool
        Se o dado for anual (True), so for trimestral (False).

    Returns
    -------
    df_effective_tax_rate: pl.DataFrame
        DataFrame da taxa de imposto de renda efetiva.
    """
    # Anual
    if anual:
        # Selecionando a dre
        dre_empresa = df_dre.filter(pl.col('cd_cvm') == cod_cvm)

        # Selecionando o imposto pago
        ir = dre_empresa.filter(pl.col('cd_conta') == cod_ir).select('dt_refer', 'vl_conta')

        # Renomeando a coluna
        ir = ir.rename({'vl_conta': 'ir'})

        # Multiplicando por -1
        ir = ir.with_columns(
            (pl.col('ir') * -1)
        )

        # Selecionando o resultado antes dos tributos sobre o lucro (earnings before tax - ebt)	
        ebt = dre_empresa.filter(pl.col('cd_conta') == cod_ebt).select('dt_refer', 'vl_conta')

        # Renomeando a coluna
        ebt = ebt.rename({'vl_conta': 'ebt'})

    # Trimestral
    else:
        # Selecionando a dre
        dre_empresa = df_dre.filter(pl.col('cd_cvm') == cod_cvm)

        # Selecionando o imposto pago
        ir = dre_empresa.filter(pl.col('cd_conta') == cod_ir).select('dt_refer', 'dt_ini_exerc', 'vl_conta')

        # Existem linhas duplicadas na coluna 'dt_refer', removendo as linhas duplicadas, mantendo pela data mais recente da 'dt_ini_exerc'
        ir = (ir
            .sort(['dt_refer', 'dt_ini_exerc'], descending=[False, True])
            .unique(subset='dt_refer', keep='first')
            .sort('dt_refer')
        ).select('dt_refer', 'vl_conta')

        # Renomeando a coluna
        ir = ir.rename({'vl_conta': 'ir'})

        # Multiplicando por -1
        ir = ir.with_columns(
            (pl.col('ir') * -1)
        )

        # Selecionando o resultado antes dos tributos sobre o lucro (earnings before tax - ebt)	
        ebt = dre_empresa.filter(pl.col('cd_conta') == cod_ebt).select('dt_refer', 'dt_ini_exerc', 'vl_conta')

        # Existem linhas duplicadas na coluna 'dt_refer', removendo as linhas duplicadas, mantendo pela data mais recente da 'dt_ini_exerc'
        ebt = (ebt
            .sort(['dt_refer', 'dt_ini_exerc'], descending=[False, True])
            .unique(subset='dt_refer', keep='first')
            .sort('dt_refer')
        ).select('dt_refer', 'vl_conta')

        # Renomeando a coluna
        ebt = ebt.rename({'vl_conta': 'ebt'})

    # Juntando os dfs
    df_effective_tax_rate = ir.join(ebt, on='dt_refer', how='inner')

    # Calculando a taxa de imposto efetiva
    df_effective_tax_rate = df_effective_tax_rate.with_columns(
        (pl.col('ir') / pl.col('ebt')).alias('effective_tax_rate')
    ).select('dt_refer', 'effective_tax_rate')

    # Substituindo as taxas negativas (significa que a empresa não pagou imposto, imposto corrente menor que o imposto diferido) por 0
    df_effective_tax_rate = df_effective_tax_rate.with_columns(
        pl.col('effective_tax_rate').clip(lower_bound=0).alias('effective_tax_rate')
    )

    return df_effective_tax_rate


def indicador_taxes_on_operating_income(df_dre: pl.DataFrame, cod_cvm: int, cod_ebit: str, df_effective_tax_rate: pl.DataFrame, anual: bool) -> pl.DataFrame:
    """
    Calcula o imposto sobre a receita operacional (ebit).

    Parameters
    ----------
    df_dre: pl.DataFrame
        DataFrame da demonstração do resultado do exercício.
    cod_cvm: int
        Código da empresa na B3.
    cod_ebit: str
        Código do ebit na dre.
    df_effective_tax_rate: pl.DataFrame
        DataFrame da taxa de imposto de renda efetiva.
    anual: bool
        Se o dado for anual (True), so for trimestral (False).

    Returns
    -------
    df_taxes_on_operating_income: pl.DataFrame
        DataFrame do imposto sobre a receita operacional (ebit).
    """
    # Anual
    if anual:
        # Selecionando a dre
        dre_empresa = df_dre.filter(pl.col('cd_cvm') == cod_cvm)

        # Selecionando o ebit
        ebit = dre_empresa.filter(pl.col('cd_conta') == cod_ebit).select('dt_refer', 'vl_conta')

        # Renomeando a coluna
        ebit = ebit.rename({'vl_conta': 'ebit'})
    
    # Trimestral
    else:
        # Selecionando a dre
        dre_empresa = df_dre.filter(pl.col('cd_cvm') == cod_cvm)

        # Selecionando o ebit
        ebit = dre_empresa.filter(pl.col('cd_conta') == cod_ebit).select('dt_refer', 'dt_ini_exerc', 'vl_conta')

        # Existem linhas duplicadas na coluna 'dt_refer', removendo as linhas duplicadas, mantendo pela data mais recente da 'dt_ini_exerc'
        ebit = (ebit
            .sort(['dt_refer', 'dt_ini_exerc'], descending=[False, True])
            .unique(subset='dt_refer', keep='first')
            .sort('dt_refer')
        ).select('dt_refer', 'vl_conta')

        # Renomeando a coluna
        ebit = ebit.rename({'vl_conta': 'ebit'})

    # Juntando os dfs
    df_taxes_on_operating_income = ebit.join(df_effective_tax_rate, on='dt_refer', how='inner')

    # Calculando o imposto sobre a receita operacional (ebit)
    df_taxes_on_operating_income = df_taxes_on_operating_income.with_columns(
        (pl.col('ebit') * pl.col('effective_tax_rate')).alias('taxes_on_operating_income')
    ).select('dt_refer', 'taxes_on_operating_income')

    return df_taxes_on_operating_income


def indicador_new_borrowing(df_dfc: pl.DataFrame, cod_cvm: int, lst_cod_new_borrowing: list) -> pl.DataFrame:
    """
    Calcula os empréstimos captados pela empresa.

    Parameters
    ----------
    df_dfc: pl.DataFrame
        DataFrame do demonstrativo de fluxo de caixa.
    cod_cvm: int
        Código da empresa na B3.
    lst_cod_new_borrowing: list
        Lista dos códigos que formam as captações de empréstimos no dfc.

    Returns
    -------
    df_new_borrowing: pl.DataFrame
        DataFrame dos empréstimos captados pela empresa.
    """
    # Selecionando a dfc
    dfc_empresa = df_dfc.filter(pl.col('cd_cvm') == cod_cvm)

    # Lista do new borrowing
    lst_new_borrowing = []

    # Iterando sobre os códigos cvm
    for i, cod in enumerate(lst_cod_new_borrowing, start=1):
        new_borrowing = dfc_empresa.filter(pl.col('cd_conta') == cod).select(['dt_refer', pl.col('vl_conta').alias(f'vl_conta_{i}')])
        lst_new_borrowing.append(new_borrowing)

    # Fazendo o join de todos os dfs da lista
    df_new_borrowing = lst_new_borrowing[0]
    for df in lst_new_borrowing[1:]:
        df_new_borrowing = df_new_borrowing.join(df, on='dt_refer', how='full')

    # Preenchendo os nulos com zero
    df_new_borrowing = df_new_borrowing.fill_null(0)

     # Calculando o total de new borrowing (soma das colunas — usar start=pl.lit(0) para evitar 0 + Expr do Python)
    expr_soma_new_borrowing = sum(
        (pl.col(f'vl_conta_{i}') for i in range(1, len(lst_cod_new_borrowing) + 1)),
        pl.lit(0)
    )
    
    df_new_borrowing = df_new_borrowing.with_columns(
        (expr_soma_new_borrowing).alias('new_borrowing')
    ).select('dt_refer', 'new_borrowing')

    return df_new_borrowing


def indicador_new_borrowing_trimestral(df_dfc: pl.DataFrame, cod_cvm: int, lst_cod_new_borrowing: list) -> pl.DataFrame:
    """
    Calcula os empréstimos captados pela empresa.

    Parameters
    ----------
    df_dfc: pl.DataFrame
        DataFrame do demonstrativo de fluxo de caixa.
    cod_cvm: int
        Código da empresa na B3.
    lst_cod_new_borrowing: list
        Lista dos códigos que formam as captações de empréstimos no dfc.

    Returns
    -------
    df_new_borrowing: pl.DataFrame
        DataFrame dos empréstimos captados pela empresa.
    """
    # Selecionando a dfc
    dfc_empresa = df_dfc.filter(pl.col('cd_cvm') == cod_cvm)

    # Lista do new borrowing
    lst_new_borrowing = []

    # Iterando sobre os códigos cvm
    for i, cod in enumerate(lst_cod_new_borrowing, start=1):
        new_borrowing = (
            dfc_empresa
            .filter(pl.col('cd_conta') == cod)
            .select([
                'dt_refer', 
                pl.col('vl_conta').alias(f'vl_conta_{i}')
            ])
            .sort('dt_refer')  # garante ordem antes do shift
        )
        
        # Calculando o new borrowing não acumulado para os meses 06 e 09 (e mantendo caso contrário)
        expr_new_borrowing_nao_acum = (
            pl.when(
                (pl.col('dt_refer').dt.month() == 6) | (pl.col('dt_refer').dt.month() == 9)
            )
            .then(
                # Verifica se vl_conta_{i} é 0 para manter 0. Caso contrário, faz a subtração.
                pl.when(pl.col(f'vl_conta_{i}') == 0)
                .then(0)
                .otherwise(pl.col(f'vl_conta_{i}') - pl.col(f'vl_conta_{i}').shift(1))
            )
            .otherwise(pl.col(f'vl_conta_{i}')) 
            .alias(f'vl_conta_{i}_nao_acum')
        )

        new_borrowing_nao_acum = new_borrowing.with_columns([expr_new_borrowing_nao_acum]).select('dt_refer', f'vl_conta_{i}_nao_acum').fill_null(0)

        # Calculando o capex trimestral acumulado
        expr_new_borrowing_acum = (
            pl.col(f'vl_conta_{i}_nao_acum')
            .rolling_sum(window_size=4, min_periods=4) 
            .round(2)                                   
            .alias(f'vl_conta_{i}_acum')
        )

        new_borrowing_acum = new_borrowing_nao_acum.with_columns([expr_new_borrowing_acum]).select('dt_refer', f'vl_conta_{i}_acum').fill_null(0)

        lst_new_borrowing.append(new_borrowing_acum)

    # Se não houver nada, retorna DataFrame vazio
    if not lst_new_borrowing:
        return pl.DataFrame({'dt_refer': [], 'new_borrowing': []})

    # Fazendo o join de todos os dfs da lista
    df_new_borrowing = lst_new_borrowing[0]
    for df in lst_new_borrowing[1:]:
        df_new_borrowing = df_new_borrowing.join(df, on='dt_refer', how='full')

    # Preenchendo os nulos com zero
    df_new_borrowing = df_new_borrowing.fill_null(0)

    # Calculando o total de new borrowing (soma das colunas — usar start=pl.lit(0) para evitar 0 + Expr do Python)
    expr_soma_new_borrowing = sum(
        (pl.col(f'vl_conta_{i}_acum') for i in range(1, len(lst_cod_new_borrowing) + 1)),
        pl.lit(0)
    )

    df_new_borrowing = df_new_borrowing.with_columns(
        (expr_soma_new_borrowing).alias('new_borrowing')
    ).select('dt_refer', 'new_borrowing')

    return df_new_borrowing


def indicador_debt_paid(df_dfc: pl.DataFrame, cod_cvm: int, lst_cod_debt_paid: list) -> pl.DataFrame:
    """
    Calcula os empréstimos pagos pela empresa.

    Parameters
    ----------
    df_dfc: pl.DataFrame
        DataFrame do demonstrativo de fluxo de caixa.
    cod_cvm: int
        Código da empresa na B3.
    lst_cod_debt_paid: list
        Lista dos códigos que formam os empréstimos pagos no dfc.

    Returns
    -------
    df_debt_paid: pl.DataFrame
        DataFrame dos empréstimos pagos pela empresa.
    """
    # Selecionando a dfc
    dfc_empresa = df_dfc.filter(pl.col('cd_cvm') == cod_cvm)

    # Lista do debt paid
    lst_debt_paid = []

    # Iterando sobre os códigos cvm
    for i, cod in enumerate(lst_cod_debt_paid, start=1):
        debt_paid = dfc_empresa.filter(pl.col('cd_conta') == cod).select(['dt_refer', pl.col('vl_conta').alias(f'vl_conta_{i}')])
        lst_debt_paid.append(debt_paid)
        
    # Fazendo o join de todos os dfs da lista
    df_debt_paid = lst_debt_paid[0]
    for df in lst_debt_paid[1:]:
        df_debt_paid = df_debt_paid.join(df, on='dt_refer', how='inner')

    # Calculando o totoal de new borrowing (soma das colunas — usar start=pl.lit(0) para evitar 0 + Expr do Python)
    expr_soma_debt_paid = sum(
        (pl.col(f'vl_conta_{i}') for i in range(1, len(lst_cod_debt_paid) + 1)),
        pl.lit(0)
    )

    df_debt_paid = df_debt_paid.with_columns(
        (expr_soma_debt_paid).alias('debt_paid')
    ).select('dt_refer', 'debt_paid')

    return df_debt_paid


def indicador_debt_paid_trimestral(df_dfc: pl.DataFrame, cod_cvm: int, lst_cod_debt_paid: list) -> pl.DataFrame:
    """
    Calcula os empréstimos pagos pela empresa.

    Parameters
    ----------
    df_dfc: pl.DataFrame
        DataFrame do demonstrativo de fluxo de caixa.
    cod_cvm: int
        Código da empresa na B3.
    lst_cod_debt_paid: list
        Lista dos códigos que formam os empréstimos pagos no dfc.

    Returns
    -------
    df_debt_paid: pl.DataFrame
        DataFrame dos empréstimos pagos pela empresa.
    """
    # Selecionando a dfc
    dfc_empresa = df_dfc.filter(pl.col('cd_cvm') == cod_cvm)

    # Lista do debt paid
    lst_debt_paid = []

    # Iterando sobre os códigos cvm
    for i, cod in enumerate(lst_cod_debt_paid, start=1):
        debt_paid = (
            dfc_empresa
            .filter(pl.col('cd_conta') == cod)
            .select([
                'dt_refer', 
                pl.col('vl_conta').alias(f'vl_conta_{i}')
            ])
            .sort('dt_refer')  # garante ordem antes do shift
        )

        # Calculando o debt paid não acumulado para os meses 06 e 09 (e mantendo caso contrário)
        expr_debt_paid_nao_acum = (
            pl.when(
                (pl.col('dt_refer').dt.month() == 6) | (pl.col('dt_refer').dt.month() == 9)
            )
            .then(
                # Verifica se vl_conta_{i} é 0 para manter 0. Caso contrário, faz a subtração.
                pl.when(pl.col(f'vl_conta_{i}') == 0)
                .then(0)
                .otherwise(pl.col(f'vl_conta_{i}') - pl.col(f'vl_conta_{i}').shift(1))
            )
            .otherwise(pl.col(f'vl_conta_{i}')) 
            .alias(f'vl_conta_{i}_nao_acum')
        )

        debt_paid_nao_acum = debt_paid.with_columns([expr_debt_paid_nao_acum]).select('dt_refer', f'vl_conta_{i}_nao_acum').fill_null(0)

        # Calculando o debt paid trimestral acumulado
        expr_debt_paid_acum = (
            pl.col(f'vl_conta_{i}_nao_acum')
            .rolling_sum(window_size=4, min_periods=4) 
            .round(2)                                   
            .alias(f'vl_conta_{i}_acum')
        )

        debt_paid_acum = debt_paid_nao_acum.with_columns([expr_debt_paid_acum]).select('dt_refer', f'vl_conta_{i}_acum').fill_null(0)

        lst_debt_paid.append(debt_paid_acum)

    # Se não houver nada, retorna DataFrame vazio
    if not lst_debt_paid:
        return pl.DataFrame({'dt_refer': [], 'debt_paid': []})
    
    # Fazendo o join de todos os dfs da lista
    df_debt_paid = lst_debt_paid[0]
    for df in lst_debt_paid[1:]:
        df_debt_paid = df_debt_paid.join(df, on='dt_refer', how='inner')

    # Calculando o total de new borrowing (soma das colunas — usar start=pl.lit(0) para evitar 0 + Expr do Python)
    expr_soma_debt_paid = sum(
        (pl.col(f'vl_conta_{i}_acum') for i in range(1, len(lst_cod_debt_paid) + 1)),
        pl.lit(0)
    ) * -1

    df_debt_paid = df_debt_paid.with_columns(
        (expr_soma_debt_paid).alias('debt_paid')
    ).select('dt_refer', 'debt_paid')

    return df_debt_paid


def indicador_fcfe(
    df_dre: pl.DataFrame, 
    df_dfc: pl.DataFrame,
    cod_cvm: int, 
    cod_ll: str, 
    cod_depreciacao: str,
    df_capex: pl.DataFrame,
    df_change_in_non_cash_wc: pl.DataFrame,
    df_new_borrowing: pl.DataFrame,
    df_debt_paid: pl.DataFrame,
) -> pl.DataFrame:
    """
    Calcula o FCFE da empresa.

    Parameters
    ----------
    df_dre: pl.DataFrame
        DataFrame da demonstração do resultado do exercício.
    df_dfc: pl.DataFrame
        DataFrame do demonstrativo de fluxo de caixa.
    cod_cvm: int
        Código da empresa na B3.
    cod_ll: str
        Código do lucro líquido na dre.
    cod_depreciacao: str
        Código da depreciação na dfc.
    df_capex: pl.DataFrame
        DataFrame do capex da empresa.
    df_change_in_non_cash_wc: pl.DataFrame
        DataFrame das variações nos ativos e passivos do fluxo de caixa da empresa.
    df_new_borrowing: pl.DataFrame
        DataFrame dos empréstimos captados pela empresa.
    df_debt_paid: pl.DataFrame
        DataFrame dos empréstimos pagos pela empresa.

    Returns
    -------
    df_fcfe: pl.DataFrame
        DataFrame do FCFE da empresa.
    """
    # Selecionando a dre
    dre_empresa = df_dre.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o lucro líquido
    lucro_liquido = dre_empresa.filter(pl.col('cd_conta') == cod_ll).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    lucro_liquido = lucro_liquido.rename({'vl_conta': 'lucro_liquido'})

    # Selecionando a dfc
    dfc_empresa = df_dfc.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando a depreciação
    depreciacao = dfc_empresa.filter(pl.col('cd_conta') == cod_depreciacao).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    depreciacao = depreciacao.rename({'vl_conta': 'depreciacao'})

    # Lista dos dfs para juntar em apenas um df
    lst_df = [
        lucro_liquido, 
        depreciacao, 
        df_capex, 
        df_change_in_non_cash_wc, 
        df_new_borrowing, 
        df_debt_paid
    ]

    # Juntando os dfs
    df_fcfe = reduce(
        lambda left, right: left.join(right, on='dt_refer', how='inner'), lst_df
    )

    # Calculando o FCFE
    df_fcfe = df_fcfe.with_columns((
        (
            pl.col('lucro_liquido') + pl.col('depreciacao') - pl.col('capex') - pl.col('change_in_non_cash_wc') + (pl.col('new_borrowing') - pl.col('debt_paid'))
        )
        .alias('fcfe')
    )).select('dt_refer', 'fcfe')

    return df_fcfe


def indicador_fcfe_trimestral(
    df_dre: pl.DataFrame, 
    df_dfc: pl.DataFrame,
    cod_cvm: int, 
    cod_ll: str, 
    cod_depreciacao: str,
    df_capex: pl.DataFrame,
    df_change_in_non_cash_wc: pl.DataFrame,
    df_new_borrowing: pl.DataFrame,
    df_debt_paid: pl.DataFrame,
) -> pl.DataFrame:
    """
    Calcula o FCFE da empresa.

    Parameters
    ----------
    df_dre: pl.DataFrame
        DataFrame da demonstração do resultado do exercício.
    df_dfc: pl.DataFrame
        DataFrame do demonstrativo de fluxo de caixa.
    cod_cvm: int
        Código da empresa na B3.
    cod_ll: str
        Código do lucro líquido na dre.
    cod_depreciacao: str
        Código da depreciação na dfc.
    df_capex: pl.DataFrame
        DataFrame do capex da empresa.
    df_change_in_non_cash_wc: pl.DataFrame
        DataFrame das variações nos ativos e passivos do fluxo de caixa da empresa.
    df_new_borrowing: pl.DataFrame
        DataFrame dos empréstimos captados pela empresa.
    df_debt_paid: pl.DataFrame
        DataFrame dos empréstimos pagos pela empresa.

    Returns
    -------
    df_fcfe: pl.DataFrame
        DataFrame do FCFE da empresa.
    """
    # Selecionando a dre
    dre_empresa = df_dre.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o lucro líquido
    lucro_liquido = dre_empresa.filter(pl.col('cd_conta') == cod_ll).select('dt_refer', 'dt_ini_exerc', 'vl_conta')

    # Existem linhas duplicadas na coluna 'dt_refer', removendo as linhas duplicadas, mantendo pela data mais recente da 'dt_ini_exerc'
    lucro_liquido = (lucro_liquido
        .sort(['dt_refer', 'dt_ini_exerc'], descending=[False, True])
        .unique(subset='dt_refer', keep='first')
        .sort('dt_refer')
    ).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    lucro_liquido = lucro_liquido.rename({'vl_conta': 'lucro_liquido'})

    # Calculando o lucro líquido trimestral acumulado
    expr_lucro_liquido_acum = (
        pl.col('lucro_liquido')
        .rolling_sum(window_size=4, min_periods=4) 
        .round(2)                                   
    )

    df_lucro_liquido_acum = lucro_liquido.with_columns([expr_lucro_liquido_acum]).select('dt_refer', 'lucro_liquido').fill_null(0)

    # Selecionando a dfc
    dfc_empresa = df_dfc.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando a depreciação
    depreciacao = dfc_empresa.filter(pl.col('cd_conta') == cod_depreciacao).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    depreciacao = depreciacao.rename({'vl_conta':'depreciacao'})

    # Calculando a depreciaçao não acumulada para os meses 06 e 09
    expr_deprec_nao_acum = (
        pl.when(
            (pl.col('dt_refer').dt.month() == 6) | (pl.col('dt_refer').dt.month() == 9)
        )
        .then(
            # Verifica se depreciacao é 0 para manter 0. Caso contrário, faz a subtração.
            pl.when(pl.col('depreciacao') == 0)
            .then(0)
            .otherwise(pl.col('depreciacao') - pl.col('depreciacao').shift(1))
        )
        .otherwise(pl.col('depreciacao')) 
        .alias('depreciacao_nao_acum')
    )

    depreciacao_nao_acum = depreciacao.with_columns([expr_deprec_nao_acum]).select('dt_refer', 'depreciacao_nao_acum')

    # Calculando a depreciação trimestral acumulada
    expr_depreciacao_acum = (
        pl.col('depreciacao_nao_acum')
        .rolling_sum(window_size=4, min_periods=4) 
        .round(2)    
        .alias('depreciacao')                               
    )

    df_depreciacao_acum = depreciacao_nao_acum.with_columns([expr_depreciacao_acum]).select('dt_refer', 'depreciacao').fill_null(0)

    # Lista dos dfs para juntar em apenas um df
    lst_df = [
        df_lucro_liquido_acum, 
        df_depreciacao_acum, 
        df_capex, 
        df_change_in_non_cash_wc, 
        df_new_borrowing, 
        df_debt_paid
    ]

    # Juntando os dfs
    df_fcfe = reduce(
        lambda left, right: left.join(right, on='dt_refer', how='inner'), lst_df
    )

    # Calculando o FCFE
    df_fcfe = df_fcfe.with_columns((
        (
            pl.col('lucro_liquido') + pl.col('depreciacao') - pl.col('capex') - pl.col('change_in_non_cash_wc') + (pl.col('new_borrowing') - pl.col('debt_paid'))
        )
        .alias('fcfe')
    )).select('dt_refer', 'fcfe')

    return df_fcfe


def indicador_fcff(
    df_dre: pl.DataFrame, 
    df_dfc: pl.DataFrame,
    cod_cvm: int, 
    cod_ebit: str, 
    cod_depreciacao: str, 
    df_taxes_on_operating_income: pl.DataFrame,
    df_capex: pl.DataFrame,
    df_change_in_non_cash_wc: pl.DataFrame
) -> pl.DataFrame:
    """
    Calcula o FCFF da empresa.

    Parameters
    ----------
    df_dre: pl.DataFrame
        DataFrame da demonstração do resultado do exercício.
    df_dfc: pl.DataFrame
        DataFrame do demonstrativo de fluxo de caixa.
    cod_cvm: int
        Código da empresa na B3.
    cod_ebit: str
        Código do ebit na dre.
    cod_depreciacao: str
        Código da depreciação na dfc.
    df_taxes_on_operating_income: pl.DataFrame
        DataFrame do imposto sobre a receita operacional (ebit).
    df_capex: pl.DataFrame
        DataFrame do capex da empresa.
    df_change_in_non_cash_wc: pl.DataFrame
        DataFrame das variações nos ativos e passivos do fluxo de caixa da empresa.

    Returns
    -------
    df_fcff: pl.DataFrame
        DataFrame do FCFF da empresa.
    """
    # Selecionando a dre
    dre_empresa = df_dre.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o ebit
    ebit = dre_empresa.filter(pl.col('cd_conta') == cod_ebit).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    ebit = ebit.rename({'vl_conta': 'ebit'})

    # Selecionando a dfc
    dfc_empresa = df_dfc.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando a depreciação
    depreciacao = dfc_empresa.filter(pl.col('cd_conta') == cod_depreciacao).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    depreciacao = depreciacao.rename({'vl_conta': 'depreciacao'})

    # Lista dos dfs para juntar em apenas um df
    lst_df = [
        ebit, 
        df_taxes_on_operating_income, 
        depreciacao, 
        df_capex, 
        df_change_in_non_cash_wc, 
    ]

    # Juntando os dfs
    df_fcff = reduce(
        lambda left, right: left.join(right, on='dt_refer', how='inner'), lst_df
    )

    # Calculando o FCFE
    df_fcff = df_fcff.with_columns((
        (
            pl.col('ebit') - pl.col('taxes_on_operating_income') + pl.col('depreciacao') - pl.col('capex') - pl.col('change_in_non_cash_wc')
        )
        .alias('fcff')
    )).select('dt_refer', 'fcff')

    return df_fcff


def indicador_fcff_trimestral(
    df_dre: pl.DataFrame, 
    df_dfc: pl.DataFrame,
    cod_cvm: int, 
    cod_ebit: str, 
    cod_depreciacao: str, 
    df_taxes_on_operating_income: pl.DataFrame,
    df_capex: pl.DataFrame,
    df_change_in_non_cash_wc: pl.DataFrame
) -> pl.DataFrame:
    """
    Calcula o FCFF da empresa.

    Parameters
    ----------
    df_dre: pl.DataFrame
        DataFrame da demonstração do resultado do exercício.
    df_dfc: pl.DataFrame
        DataFrame do demonstrativo de fluxo de caixa.
    cod_cvm: int
        Código da empresa na B3.
    cod_ebit: str
        Código do ebit na dre.
    cod_depreciacao: str
        Código da depreciação na dfc.
    df_taxes_on_operating_income: pl.DataFrame
        DataFrame do imposto sobre a receita operacional (ebit).
    df_capex: pl.DataFrame
        DataFrame do capex da empresa.
    df_change_in_non_cash_wc: pl.DataFrame
        DataFrame das variações nos ativos e passivos do fluxo de caixa da empresa.

    Returns
    -------
    df_fcff: pl.DataFrame
        DataFrame do FCFF da empresa.
    """
    # Selecionando a dre
    dre_empresa = df_dre.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando o ebit
    ebit = dre_empresa.filter(pl.col('cd_conta') == cod_ebit).select('dt_refer', 'dt_ini_exerc', 'vl_conta')

    # Existem linhas duplicadas na coluna 'dt_refer', removendo as linhas duplicadas, mantendo pela data mais recente da 'dt_ini_exerc'
    ebit = (ebit
        .sort(['dt_refer', 'dt_ini_exerc'], descending=[False, True])
        .unique(subset='dt_refer', keep='first')
        .sort('dt_refer')
    ).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    ebit = ebit.rename({'vl_conta': 'ebit'})


    # Calculando o ebit trimestral acumulado
    expr_ebit_acum = (
        pl.col('ebit')
        .rolling_sum(window_size=4, min_periods=4) 
        .round(2)                                   
    )

    df_ebit_acum = ebit.with_columns([expr_ebit_acum]).select('dt_refer', 'ebit').fill_null(0)

    # Selecionando a dfc
    dfc_empresa = df_dfc.filter(pl.col('cd_cvm') == cod_cvm)

    # Selecionando a depreciação
    depreciacao = dfc_empresa.filter(pl.col('cd_conta') == cod_depreciacao).select('dt_refer', 'vl_conta')

    # Renomeando a coluna
    depreciacao = depreciacao.rename({'vl_conta':'depreciacao'})

    # Calculando a depreciaçao não acumulada para os meses 06 e 09
    expr_deprec_nao_acum = (
        pl.when(
            (pl.col('dt_refer').dt.month() == 6) | (pl.col('dt_refer').dt.month() == 9)
        )
        .then(
            # Verifica se depreciacao é 0 para manter 0. Caso contrário, faz a subtração.
            pl.when(pl.col('depreciacao') == 0)
            .then(0)
            .otherwise(pl.col('depreciacao') - pl.col('depreciacao').shift(1))
        )
        .otherwise(pl.col('depreciacao')) 
        .alias('depreciacao_nao_acum')
    )

    depreciacao_nao_acum = depreciacao.with_columns([expr_deprec_nao_acum]).select('dt_refer', 'depreciacao_nao_acum')

    # Calculando a depreciação trimestral acumulada
    expr_depreciacao_acum = (
        pl.col('depreciacao_nao_acum')
        .rolling_sum(window_size=4, min_periods=4) 
        .round(2)    
        .alias('depreciacao')                               
    )

    df_depreciacao_acum = depreciacao_nao_acum.with_columns([expr_depreciacao_acum]).select('dt_refer', 'depreciacao').fill_null(0)

    # Lista dos dfs para juntar em apenas um df
    lst_df = [
        df_ebit_acum, 
        df_taxes_on_operating_income, 
        df_depreciacao_acum, 
        df_capex, 
        df_change_in_non_cash_wc, 
    ]

    # Juntando os dfs
    df_fcff = reduce(
        lambda left, right: left.join(right, on='dt_refer', how='inner'), lst_df
    )

    # Calculando o FCFE
    df_fcff = df_fcff.with_columns((
        (
            pl.col('ebit') - pl.col('taxes_on_operating_income') + pl.col('depreciacao') - pl.col('capex') - pl.col('change_in_non_cash_wc')
        )
        .alias('fcff')
    )).select('dt_refer', 'fcff')

    return df_fcff