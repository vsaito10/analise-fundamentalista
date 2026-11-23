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
import streamlit.components.v1 as components
from curl_cffi.requests import RequestsError
from plotly.subplots import make_subplots
from scipy.stats import t as student_t
from yfinance.exceptions import YFException

RAIZ_PADRAO = r"C:\B3\historico-arquivos\indicadores_fundamentalistas_eua"
RAIZ_VALUATIONS = r"C:\B3\historico-arquivos\valuations\atualizado"
RAIZ_PROJECOES = r"C:\B3\historico-arquivos\projecoes"
RAIZ_GRAFO_RECEITA = r"C:\B3\historico-arquivos\grafo_receita"
ALTURA_GRAFO_RECEITA = 1450
# Diferenca de ate meio centavo entre LPA e projecao conta como "em linha"
# (as estimativas sao cotadas em centavos).
TOLERANCIA_LPA = 0.005

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
        return {"usd_bi": "US$ bi", "pct": "%", "pct_decimal": "%", "x": "x", "usd": "US$", "dias": "dias"}[self.tipo]


# Demonstrativos em US$ milhões; valor de mercado em US$ milhares.
MILHAO_PARA_BILHAO = 1e3
MILHAR_PARA_BILHAO = 1e6
METRICAS = {
    "pl_damodaran": Metrica("P/L Damodaran", "x", 1, "Preço sobre lucro pelo modelo Damodaran"),
    "pl": Metrica("P/L", "x"),
    "lpa": Metrica("LPA", "usd", 1, "Lucro por ação diluído (básico quando não há o diluído), "
                   "ajustado por desdobramentos"),
    "lp": Metrica("L/P (earnings yield)", "pct"),
    "ebitda": Metrica("EBITDA", "usd_bi", MILHAO_PARA_BILHAO),
    "margem_liquida": Metrica("Margem líquida", "pct"),
    "margem_bruta": Metrica("Margem bruta", "pct"),
    "margem_operacional": Metrica("Margem operacional", "pct"),
    "margem_ebitda": Metrica("Margem EBITDA", "pct"),
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
    "free_cash_flow_yield": Metrica("Free cash flow yield", "pct"),
    "adjusted_free_cash_flow": Metrica("Adjusted Free Cash Flow", "usd_bi", MILHAO_PARA_BILHAO),
    "sbc": Metrica("SBC", "usd_bi", MILHAO_PARA_BILHAO),
    "capex": Metrica("Capex", "usd_bi", MILHAO_PARA_BILHAO),
    "net_capex": Metrica("Net capex", "usd_bi", MILHAO_PARA_BILHAO),
    "rd": Metrica("P&D", "usd_bi", MILHAO_PARA_BILHAO),
    "adj_net_capex": Metrica("Net capex ajustado", "usd_bi", MILHAO_PARA_BILHAO),
    "working_capital": Metrica("Capital de giro", "usd_bi", MILHAO_PARA_BILHAO),
    "dso": Metrica("DSO", "dias"),
    "dio": Metrica("DIO", "dias"),
    "dpo": Metrica("DPO", "dias"),
    "cfo_ll": Metrica("CFO/LL", "x"),
    "cfo_ebitda": Metrica("CFO/EBITDA", "x"),
    "accrual_ratio": Metrica("Accrual Ratio", "pct_decimal", 0.01),
    "fcf_ll": Metrica("FCF/LL", "x"),
    "recebiveis_receita": Metrica("Recebíveis/Receita", "pct_decimal", 0.01),
    "capex_depreciacao": Metrica("CAPEX/Depreciação", "x"),
    "reinvestment_rate": Metrica("Taxa de reinvestimento", "pct"),
    "fcfe": Metrica("FCFE", "usd_bi", MILHAO_PARA_BILHAO),
    "fcff": Metrica("FCFF", "usd_bi", MILHAO_PARA_BILHAO),
    "buyback": Metrica("Recompra de ações", "usd_bi", MILHAO_PARA_BILHAO),
    "dpa": Metrica("Dividendos por ação", "usd"),
    "payout": Metrica("Payout", "pct"),
}

RANKING = {
    "valor_mercado": "maior", "lp": "maior", "pl": "menor", "ev_ebitda": "menor", "roe": "maior", "roic": "maior",
    "margem_bruta": "maior", "margem_operacional": "maior", "margem_ebitda": "maior", "margem_liquida": "maior",
    "fco": "maior", "free_cash_flow": "maior", "adjusted_free_cash_flow": "maior", "free_cash_flow_yield": "maior",
    "dpa": "maior", "buyback": "maior",
    "dl_ebitda": "menor", "dl_pl": "menor",
}
SO_POSITIVOS = ("pl", "ev_ebitda")
MARGENS = ("margem_bruta", "margem_operacional", "margem_ebitda", "margem_liquida")

# Alguns arquivos antigos trazem caracteres acentuados corrompidos. A normalização
# abaixo identifica a coluna pelo conteúdo e produz nomes internos estáveis.
COLUNAS = {
    "p/l damodaran": "pl_damodaran", "p/l": "pl", "l/p": "lp", "ebitda": "ebitda",
    "margem liquida": "margem_liquida", "margem bruta": "margem_bruta",
    "margem operacional": "margem_operacional", "margem ebitda": "margem_ebitda",
    "p/vpa": "pvp", "valor de mercado": "valor_mercado",
    "divida bruta": "divida_bruta", "caixa e equivalentes": "caixa", "divida liquida": "divida_liquida",
    "divida liquida/ebitda": "dl_ebitda", "divida liquida/pl": "dl_pl", "ev/ebitda": "ev_ebitda",
    "roe": "roe", "roic": "roic", "fco": "fco", "fci": "fci", "fcf": "fcf",
    "free cash flow": "free_cash_flow", "free cash flow yield": "free_cash_flow_yield",
    "adjusted free cash flow": "adjusted_free_cash_flow", "sbc": "sbc",
    "capex": "capex", "net capex": "net_capex",
    "r&d": "rd", "adj net capex": "adj_net_capex", "working capital": "working_capital",
    "dso": "dso", "dio": "dio", "dpo": "dpo",
    "fco/ll": "cfo_ll", "fco/ebitda": "cfo_ebitda", "accrual ratio": "accrual_ratio",
    "free cash flow/ll": "fcf_ll", "recebiveis/receita": "recebiveis_receita",
    "capex/depreciacao": "capex_depreciacao",
    "reinvestment rate": "reinvestment_rate", "fcfe": "fcfe", "fcff": "fcff", "buyback": "buyback",
    "dpa": "dpa", "payout": "payout",
}


# Mes de encerramento do 1o trimestre fiscal de cada empresa, conforme as anotacoes do
# notebook sec-api-eua.ipynb ("1Q -> Lancamento em ..."). Os demais trimestres seguem de
# 3 em 3 meses; empresas fora da lista usam o ano-calendario (1T = marco).
MES_1T_FISCAL = {
    "nvda": 4, "amd": 3, "intc": 3, "mu": 12, "avgo": 2, "mrvl": 5, "qcom": 12,
    "cdns": 3, "snps": 1,
    "amat": 1, "klac": 9, "lrcx": 9,
    "ceg": 3, "vst": 3, "tln": 3, "oklo": 3, "nrg": 3,
    "amzn": 3, "aapl": 12, "googl": 3, "meta": 3, "msft": 9, "nflx": 3,
    "pltr": 3, "net": 3, "ddog": 3,
}


def periodo_trimestral(datas, ticker: str) -> pd.Series:
    """Rotulo ``AAAATq`` do trimestre fiscal da empresa.

    ``AAAA`` e o ano em que o exercicio fiscal termina (o mesmo dos rotulos anuais do
    10-K) e ``q`` o trimestre fiscal. Ex.: NVIDIA (1T = abril), periodo encerrado em
    julho/2026 -> ``2027T2``.
    """
    datas = pd.to_datetime(pd.Series(datas), errors="coerce").reset_index(drop=True)
    mes_1t = MES_1T_FISCAL.get(ticker.lower(), 3)
    # A data bruta da SEC pode cair alguns dias depois (mes seguinte) ou antes (mes
    # anterior) do mes de encerramento: aproxima para o mes do ciclo da empresa.
    ajuste = ((datas.dt.month - mes_1t) % 3).map({0: 0, 1: -1, 2: 1})
    meses = datas.dt.year * 12 + datas.dt.month - 1 + ajuste
    ano, mes = meses // 12, meses % 12 + 1
    trimestre = ((mes - mes_1t) % 12) // 3 + 1
    mes_fim_exercicio = (mes_1t + 8) % 12 + 1
    ano_fiscal = ano + (mes > mes_fim_exercicio).astype("Int64")
    rotulo = ano_fiscal.astype("Int64").astype(str) + "T" + trimestre.astype("Int64").astype(str)
    return rotulo.where(datas.notna())


# Texto curto da aba "Sobre a empresa", por ticker (nome do arquivo <ticker>_indicators.xlsx).
SOBRE_EMPRESA = {
    "cdns": "A Cadence Design Systems desenvolve softwares de EDA (automação de projeto eletrônico), usados por fabricantes "
            "como Nvidia, Apple e Qualcomm para projetar, simular e verificar chips antes da fabricação. Também vende "
            "hardware de emulação e prototipagem, licencia blocos de propriedade intelectual (IP) de semicondutores e tem "
            "softwares de análise de sistemas (térmica, eletromagnética e fluidodinâmica). A receita vem em grande parte de "
            "licenças recorrentes de longo prazo, e a empresa divide com a Synopsys a liderança desse mercado.",
    "snps": "A Synopsys é a maior empresa de softwares de EDA (automação de projeto eletrônico), ferramentas usadas para "
            "projetar e verificar chips, e uma das maiores fornecedoras de blocos de propriedade intelectual (IP) de "
            "semicondutores, como interfaces USB, PCIe e memória. Em 2025 concluiu a compra da Ansys, líder em software de "
            "simulação de engenharia, ampliando sua atuação do projeto do chip para o projeto do sistema completo. A receita "
            "vem principalmente de licenças recorrentes, e a empresa divide com a Cadence a liderança desse mercado.",
    "ceg": "A Constellation Energy é a maior geradora de energia livre de carbono dos Estados Unidos e a maior operadora de "
           "usinas nucleares do país. Vende eletricidade no atacado e diretamente para empresas, governos e residências, e "
           "fecha contratos de longo prazo com grandes consumidores, como data centers de empresas de tecnologia. Também tem "
           "usinas a gás natural, hidrelétricas, eólicas e solares, e ampliou sua base de geração com a compra da Calpine, "
           "concluída em 2026.",
    "nrg": "A NRG Energy é uma empresa integrada de energia que gera eletricidade e a vende no varejo para residências e "
           "empresas, principalmente no Texas e no Leste dos Estados Unidos, com marcas como Reliant, Green Mountain e "
           "Direct Energy. Também oferece serviços residenciais pela Vivint (segurança e automação) e vende gás natural. "
           "O modelo combina usinas próprias, principalmente a gás, com uma grande carteira de clientes de varejo.",
    "oklo": "A Oklo desenvolve pequenos reatores nucleares de fissão rápida (a linha Aurora) para gerar eletricidade e calor "
            "para data centers, indústrias e bases militares. Seu modelo é construir, ser dona e operar as usinas, vendendo "
            "a energia por contratos de longo prazo, em vez de vender os reatores. Também trabalha com reciclagem de "
            "combustível nuclear e produção de radioisótopos. A empresa ainda está em fase pré-operacional, sem receita "
            "relevante, à espera das licenças da NRC para a primeira usina.",
    "tln": "A Talen Energy é uma produtora independente de energia que opera usinas nucleares e a gás natural, "
           "principalmente na região PJM, no Leste dos Estados Unidos. Seu principal ativo é a usina nuclear de Susquehanna, "
           "na Pensilvânia, que tem um contrato de longo prazo de fornecimento de energia para data centers da Amazon (AWS). "
           "A receita vem da venda de energia e de capacidade no mercado atacadista e de contratos bilaterais.",
    "vst": "A Vistra é uma das maiores empresas de energia dos Estados Unidos, com geração e varejo integrados. Tem uma grande "
           "frota de usinas a gás natural, nucleares (que cresceram com a compra da Energy Harbor) e a carvão, além de "
           "projetos solares e de baterias. No varejo, vende eletricidade para milhões de clientes residenciais e "
           "comerciais, principalmente no Texas, com marcas como TXU Energy.",
    "amat": "A Applied Materials é a maior fabricante de equipamentos para a produção de semicondutores do mundo. Suas "
            "máquinas fazem etapas como deposição de materiais, corrosão (etch), implantação de íons, planarização (CMP) e "
            "inspeção nas fábricas de chips de clientes como TSMC, Samsung e Intel. Também fabrica equipamentos para "
            "displays e presta serviços, peças e atualizações para a base instalada de máquinas.",
    "klac": "A KLA é líder mundial em equipamentos de inspeção e metrologia para a fabricação de semicondutores, que medem e "
            "encontram defeitos nos wafers durante a produção para aumentar o rendimento (yield) das fábricas. Tem "
            "participação dominante nesse nicho e também vende equipamentos para encapsulamento de chips e placas de "
            "circuito. Uma parte relevante da receita vem de serviços para a base instalada.",
    "lrcx": "A Lam Research fabrica equipamentos para a produção de semicondutores e é líder em corrosão (etch) e forte em "
            "deposição de materiais, etapas essenciais para chips de memória (NAND e DRAM) e para lógica avançada. Seus "
            "principais clientes são fabricantes como Samsung, TSMC, Micron e SK Hynix. Uma parte relevante da receita vem do "
            "segmento de serviços e peças para a base instalada de equipamentos.",
    "bwxt": "A BWX Technologies fabrica reatores nucleares e componentes nucleares para a Marinha dos EUA, sendo a única "
            "fornecedora dos reatores que equipam submarinos e porta-aviões nucleares americanos. Também produz combustível "
            "nuclear, opera instalações do Departamento de Energia e fornece componentes e serviços para usinas nucleares "
            "civis, além de isótopos para medicina. A maior parte da receita vem de contratos de longo prazo com o governo "
            "americano.",
    "be": "A Bloom Energy fabrica e instala servidores de energia baseados em células a combustível de óxido sólido (SOFC), "
          "que geram eletricidade no próprio local do cliente a partir de gás natural, biogás ou hidrogênio, sem combustão. "
          "Vende para data centers, indústrias, hospitais, varejistas e concessionárias, e ganhou destaque como fonte de "
          "energia rápida de instalar para data centers de IA, com acordos com empresas como Oracle, AEP e Brookfield. Também "
          "desenvolve eletrolisadores para produzir hidrogênio. A receita vem da venda e instalação dos sistemas, de "
          "contratos de serviço e manutenção de longo prazo e da venda de energia.",
    "gev": "A GE Vernova, separada da General Electric em 2024, fabrica equipamentos para geração e transmissão de energia. "
           "Seu maior negócio são as turbinas a gás e os serviços para usinas; também produz turbinas eólicas (em terra e no "
           "mar), equipamentos de rede elétrica (transformadores, subestações e softwares de controle) e pequenos reatores "
           "nucleares. Uma grande parte da receita e do lucro vem de contratos de serviço para a base instalada de turbinas.",
    "amd": "A AMD (Advanced Micro Devices) projeta processadores e placas de vídeo sem fábrica própria (modelo fabless; a "
           "fabricação é da TSMC). Vende CPUs Ryzen para PCs e EPYC para servidores, GPUs Radeon e aceleradores Instinct para "
           "inteligência artificial em data centers, além de chips customizados para consoles (PlayStation e Xbox) e chips "
           "programáveis (FPGAs, vindos da compra da Xilinx). Concorre com a Intel em CPUs e com a Nvidia em GPUs.",
    "avgo": "A Broadcom é uma das maiores empresas de semicondutores e de software de infraestrutura do mundo. Em chips, vende "
            "aceleradores de IA customizados (ASICs) para grandes empresas de tecnologia, chips de rede para data centers, "
            "componentes de radiofrequência para smartphones (como os da Apple) e chips de banda larga e armazenamento. Em "
            "software, oferece produtos para empresas, com destaque para a VMware, comprada em 2023, além de CA e Symantec.",
    "intc": "A Intel projeta e fabrica processadores, sendo líder histórica em CPUs para PCs (linha Core) e servidores (linha "
            "Xeon). Diferente da maioria das concorrentes, tem fábricas próprias e vem investindo pesado para recuperar a "
            "liderança tecnológica de fabricação e oferecer fabricação para outras empresas (Intel Foundry). Também tem "
            "participação na Mobileye (chips para carros autônomos) e na Altera (chips programáveis).",
    "mrvl": "A Marvell Technology projeta chips sem fábrica própria (modelo fabless) para infraestrutura de dados. Seu maior "
            "negócio é o de data centers: chips ópticos e eletro-ópticos que conectam servidores e aceleradores de IA, "
            "chips customizados (ASICs) para grandes empresas de nuvem e controladores de armazenamento. Também vende chips "
            "para redes de telecomunicações, equipamentos corporativos e automóveis.",
    "mu": "A Micron é uma das três maiores fabricantes de memória do mundo, ao lado da Samsung e da SK Hynix, e a única grande "
          "fabricante americana. Produz memória DRAM (incluindo HBM, usada em aceleradores de IA) e memória NAND flash, "
          "vendidas para data centers, PCs, smartphones, carros e indústria. Por ser um mercado de commodity, a receita e as "
          "margens oscilam bastante com o ciclo de preços de memória.",
    "nvda": "A Nvidia projeta GPUs (unidades de processamento gráfico) e plataformas de computação acelerada sem fábrica "
            "própria (modelo fabless). É a líder absoluta em chips para treinamento e execução de modelos de inteligência "
            "artificial em data centers, vendendo GPUs, sistemas completos, equipamentos de rede (NVLink, InfiniBand e "
            "Ethernet) e a plataforma de software CUDA. Também vende placas de vídeo GeForce para games e chips para "
            "automóveis e robótica.",
    "qcom": "A Qualcomm projeta chips sem fábrica própria (modelo fabless), sendo líder em processadores e modems para "
            "smartphones (linha Snapdragon). Além da venda de chips (QCT), tem um negócio muito lucrativo de licenciamento "
            "de patentes de tecnologias de comunicação sem fio, como 4G e 5G (QTL), cobrado de quase todos os fabricantes "
            "de celular. Vem diversificando para chips de automóveis, PCs, internet das coisas e data centers.",
    "net": "A Cloudflare opera uma das maiores redes globais de internet, com data centers em centenas de cidades, e vende "
           "serviços na nuvem que deixam sites e aplicativos mais rápidos e seguros. Seus produtos incluem rede de "
           "distribuição de conteúdo (CDN), proteção contra ataques DDoS, firewall de aplicações, segurança de acesso para "
           "empresas (Zero Trust) e uma plataforma para desenvolvedores rodarem código e IA na borda da rede. A receita vem "
           "de assinaturas.",
    "adbe": "A Adobe desenvolve softwares para criação de conteúdo, documentos digitais e marketing. O segmento Digital "
            "Media, cerca de três quartos da receita, reúne o Creative Cloud (Photoshop, Illustrator, Premiere Pro, "
            "Lightroom e Adobe Express) e o Document Cloud (Acrobat, o leitor e editor de PDF, e o Acrobat Sign, de "
            "assinatura eletrônica). O segmento Digital Experience vende a empresas ferramentas de análise de clientes, "
            "gestão de conteúdo e automação de marketing. A receita vem quase toda de assinaturas, e a Adobe vem "
            "colocando IA generativa (Firefly) nos seus produtos.",
    "ddog": "A Datadog oferece uma plataforma de monitoramento e segurança na nuvem para equipes de tecnologia. Ela reúne em "
            "um só lugar métricas de infraestrutura, desempenho de aplicações (APM), logs, experiência do usuário e "
            "segurança, permitindo acompanhar e resolver problemas em sistemas na nuvem. A receita vem de assinaturas "
            "cobradas pelo uso, e a empresa cresce vendendo mais produtos para a mesma base de clientes.",
    "pltr": "A Palantir desenvolve softwares de análise e integração de dados e de inteligência artificial para governos e "
            "empresas. Suas plataformas são o Gotham (usado por agências de defesa e inteligência), o Foundry (para "
            "operações de empresas), o AIP (que aplica modelos de IA aos dados e processos do cliente) e o Apollo (para "
            "implantação de software). O governo dos EUA é seu maior cliente, mas o segmento comercial vem crescendo "
            "rápido.",
    "now": "A ServiceNow vende uma plataforma de software na nuvem (Now Platform) que automatiza fluxos de trabalho dentro das "
           "empresas. Começou com gestão de serviços de TI (ITSM) e se expandiu para atendimento ao cliente, recursos "
           "humanos, segurança, finanças e desenvolvimento de aplicações sem código, com agentes de IA integrados. A receita "
           "vem de assinaturas, principalmente de grandes empresas e governos.",
    "amzn": "A Amazon é a maior empresa de comércio eletrônico do mundo e também a maior provedora de computação em nuvem, "
            "pela AWS (Amazon Web Services), que gera a maior parte do lucro operacional. Além das vendas próprias, cobra "
            "comissões e serviços de logística de vendedores terceiros e tem um negócio de publicidade em rápido "
            "crescimento. Também oferece a assinatura Prime (frete, vídeo e música) e tem lojas físicas, como a Whole Foods.",
    "aapl": "A Apple projeta e vende aparelhos eletrônicos de consumo, com destaque para o iPhone, que gera cerca de metade da "
            "receita, além de Mac, iPad, Apple Watch, AirPods e Vision Pro. O segmento de serviços (App Store, iCloud, "
            "Apple Music, Apple TV+, Apple Pay e o acordo do Google como buscador padrão) cresce mais rápido e tem margens "
            "bem maiores. O modelo integra hardware, software próprio (iOS e macOS) e chips próprios.",
    "googl": "A Alphabet é a controladora do Google. A maior parte da receita vem de publicidade na Busca, no YouTube e na "
             "rede de sites parceiros. Também tem o Google Cloud (infraestrutura de nuvem, IA e Workspace), assinaturas e "
             "dispositivos (YouTube Premium, Google One, Android e Pixel) e os chips de IA próprios (TPUs). Em Other Bets ficam "
             "projetos de longo prazo, como a Waymo (carros autônomos).",
    "meta": "A Meta é dona das redes sociais Facebook, Instagram, WhatsApp, Messenger e Threads, usadas por mais de 3 bilhões "
            "de pessoas por dia. Quase toda a receita vem de publicidade nesses aplicativos. Também investe pesado em "
            "inteligência artificial (modelos Llama e assistente Meta AI) e no Reality Labs, que desenvolve óculos de "
            "realidade virtual e aumentada (Quest e óculos Ray-Ban Meta) e ainda dá prejuízo.",
    "msft": "A Microsoft é uma das maiores empresas de software e de nuvem do mundo. Seus negócios são a nuvem Azure, o "
            "Office/Microsoft 365, o Windows, o LinkedIn, o Dynamics (software de gestão) e a área de games (Xbox e "
            "Activision Blizzard). É também a principal parceira e investidora da OpenAI, e vende ferramentas de IA como o "
            "Copilot integradas aos seus produtos. A maior parte da receita é recorrente, vinda de assinaturas e da nuvem.",
    "nflx": "A Netflix é a maior plataforma de streaming de vídeo por assinatura do mundo, com centenas de milhões de "
            "assinantes em mais de 190 países. Produz e licencia séries, filmes, documentários e programas ao vivo, e "
            "também tem jogos. A receita vem principalmente das assinaturas mensais, complementada por um plano mais "
            "barato com anúncios, que fez a publicidade virar uma nova fonte de receita.",
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
        ticker = Path(caminho).stem.removesuffix("_indicators")
        df["periodo"] = periodo_trimestral(datas, ticker).to_numpy()
        df["_data"] = datas
        df = df.sort_values("_data")
    if "valor_mercado" in df.columns and _acoes_em_milhares(caminho, aba):
        df["valor_mercado"] = pd.to_numeric(df["valor_mercado"], errors="coerce") / 1000
    return df.reset_index(drop=True)


def _acoes_em_milhares(caminho: str, aba: str) -> bool:
    """O notebook grava o valor de mercado como preco x acoes x 1000, na unidade das acoes
    do income statement: com acoes em milhoes o resultado sai em US$ mil (padrao do app);
    com acoes em milhares (ex.: CDNS, KLAC, PLTR) sai em US$. A maior empresa em milhoes
    tem ~25 mil (NVDA) e a menor em milhares ~90 mil (BWXT), dai o corte de 50 mil."""
    ticker = Path(caminho).stem.removesuffix("_indicators")
    demonstrativo = Path(caminho).with_name(f"{ticker}_{aba.removeprefix('indicators_')}.xlsx")
    try:
        dre = pd.read_excel(demonstrativo, sheet_name="income_statement", index_col=0)
    except (OSError, ValueError):
        return False
    if "Number of Shares - Basic" not in dre.index:
        return False
    acoes = pd.to_numeric(dre.loc[["Number of Shares - Basic"]].iloc[0], errors="coerce")
    return bool(acoes.median() > 50_000)


def escala(df: pd.DataFrame, coluna: str) -> pd.Series:
    return pd.to_numeric(df[coluna], errors="coerce") / METRICAS[coluna].divisor


@st.cache_data(show_spinner=False)
def carregar_receita(caminho: str, tipo: str, _mtime: float) -> pd.DataFrame:
    """Carrega a receita do demonstrativo e a associa aos períodos dos indicadores."""
    demonstrativo = pd.read_excel(caminho, sheet_name="income_statement", index_col=0)
    linha_receita = next((indice for indice in demonstrativo.index if _chave_coluna(indice) in {"revenue", "revenues", "net revenue", "net revenues"}), None)
    if linha_receita is None:
        return pd.DataFrame(columns=["periodo", "receita"])
    receita = pd.to_numeric(demonstrativo.loc[linha_receita], errors="coerce")
    datas = pd.to_datetime(receita.index, errors="coerce")
    if tipo == "indicators_10k":
        periodos = datas.year.astype("Int64").astype(str)
    else:
        periodos = periodo_trimestral(datas, Path(caminho).stem.rsplit("_", 1)[0]).to_numpy()
    return pd.DataFrame({"periodo": periodos, "receita": receita.to_numpy()}).dropna(subset=["periodo", "receita"])


# Prazo maximo de entrega a SEC (large accelerated filers), em dias apos o fim do periodo.
PRAZO_ENTREGA = {"indicators_10k": 60, "indicators_10q": 40}


def _linha_demonstrativo(demonstrativo: pd.DataFrame, *nomes: str) -> pd.Series | None:
    """Primeira linha encontrada (pelo nome normalizado) convertida para numero."""
    chaves = {_chave_coluna(indice): indice for indice in demonstrativo.index}
    nome = next((chaves[n] for n in nomes if n in chaves), None)
    if nome is None:
        return None
    valores = demonstrativo.loc[nome]
    if isinstance(valores, pd.DataFrame):          # linha repetida: fica com a primeira
        valores = valores.iloc[0]
    return pd.to_numeric(valores, errors="coerce")


@st.cache_data(show_spinner=False)
def carregar_lpa(caminho: str, tipo: str, _mtime: float) -> pd.DataFrame:
    """LPA do demonstrativo de resultados, como publicado (sem ajuste de desdobramento).

    Usa o LPA diluido e cai para o basico quando o diluido falta ou vem 0,00 com basico
    diferente de zero. ``escala`` compara o LPA com lucro / acoes da mesma linha
    (normalizado pela mediana do arquivo): ~1 quando os dois vieram do mesmo relatorio,
    ~1/razao quando o LPA ja foi reapresentado depois de um desdobramento.
    """
    vazio = pd.DataFrame(columns=["periodo", "data_fim", "lpa", "escala"])
    demonstrativo = pd.read_excel(caminho, sheet_name="income_statement", index_col=0)
    diluido = _linha_demonstrativo(demonstrativo, "earnings per share, diluted")
    basico = _linha_demonstrativo(demonstrativo, "earnings per share, basic")
    if diluido is None and basico is None:
        return vazio
    if diluido is None:
        lpa = basico
    elif basico is None:
        lpa = diluido
    else:
        lpa = diluido.where(diluido.notna() & ~((diluido == 0) & (basico != 0)), basico)

    lucro = _linha_demonstrativo(demonstrativo, "net income")
    acoes = _linha_demonstrativo(demonstrativo, "number of shares - diluted", "number of shares - basic")
    if lucro is not None and acoes is not None:
        relacao = (lpa / (lucro / acoes)).where((lpa.abs() > 0.02) & (lucro != 0) & (acoes > 0))
        escala = relacao / relacao.median()
    else:
        escala = pd.Series(np.nan, index=lpa.index)

    datas = pd.to_datetime(lpa.index, errors="coerce")
    if tipo == "indicators_10k":
        periodos = datas.year.astype("Int64").astype(str)
    else:
        periodos = periodo_trimestral(datas, Path(caminho).stem.rsplit("_", 1)[0]).to_numpy()
    df = pd.DataFrame({"periodo": periodos, "data_fim": datas, "lpa": lpa.to_numpy(),
                       "escala": escala.reindex(lpa.index).to_numpy()})
    return df.dropna(subset=["data_fim", "lpa"]).sort_values("data_fim").drop_duplicates("periodo", keep="last")


@st.cache_data(ttl=86400, show_spinner=False)
def carregar_desdobramentos(ticker: str) -> pd.Series | None:
    """Historico de desdobramentos do Yahoo (razao novas/antigas acoes); None se falhar."""
    import yfinance as yf

    yf.cache.set_cache_location(str(Path(tempfile.gettempdir()) / "analise-fundamentalista-yfinance"))
    try:
        desdobramentos = yf.Ticker(ticker).splits
    except (KeyError, RequestsError, ValueError, YFException):
        return None
    if desdobramentos.empty:
        return pd.Series(dtype=float)
    desdobramentos.index = pd.DatetimeIndex(desdobramentos.index).tz_localize(None)
    return desdobramentos[desdobramentos > 0]


def ajustar_desdobramentos(lpa: pd.DataFrame, desdobramentos: pd.Series, tipo: str) -> pd.Series:
    """Divide o LPA publicado antes de cada desdobramento pela razao do desdobramento.

    O relatorio ja vem ajustado quando e entregue depois do desdobramento, mesmo que o
    periodo tenha terminado antes. Como os arquivos nao trazem a data de entrega, um
    periodo so conta como publicado antes se terminou mais que o prazo legal da SEC
    antes do desdobramento. Linhas cujo LPA ja foi reapresentado por um relatorio
    posterior (``escala`` ~ 1/razao acumulada) nao sao divididas de novo.
    """
    prazo = pd.Timedelta(days=PRAZO_ENTREGA[tipo])
    ordenados = sorted(desdobramentos.items())
    ajustado = []
    for linha in lpa.itertuples():
        razoes = [razao for data, razao in ordenados if linha.data_fim + prazo < data]
        acumulados = np.cumprod([1.0, *razoes])
        ja_aplicado = 1.0
        if pd.notna(linha.escala) and linha.escala > 0:
            distancias = np.abs(np.log(linha.escala * acumulados))
            if distancias.min() < np.log(1.25):
                ja_aplicado = acumulados[distancias.argmin()]
        ajustado.append(linha.lpa / (acumulados[-1] / ja_aplicado))
    return pd.Series(ajustado, index=lpa.index, dtype=float)


def numero(valor: float) -> str:
    if valor is None or pd.isna(valor):
        return "—"
    return f"{valor:,.2f}"


def fmt(valor: float, tipo: str) -> str:
    texto = numero(valor)
    if texto == "—":
        return texto
    return {"usd_bi": f"US$ {texto} bi", "pct": f"{texto}%", "pct_decimal": f"{texto}%", "x": f"{texto}x", "usd": f"US$ {texto}", "dias": f"{texto} dias"}[tipo]


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


def linhas_com_referencia_zero(df: pd.DataFrame, colunas: list[str], titulo: str, cores: dict) -> go.Figure:
    """Desenha linhas de indicadores e destaca o nível zero."""
    fig = linhas(df, colunas, titulo, cores)
    fig.add_hline(y=0, line={"color": cores["negativo"], "width": 1.5})
    return fig


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


def receita_dso(df: pd.DataFrame, receita: pd.DataFrame, cores: dict) -> go.Figure | None:
    """Compara receita e prazo médio de recebimento, preservando as unidades de cada série."""
    dados = df[["periodo", "dso"]].merge(receita, on="periodo", how="inner")
    if dados.empty:
        return None
    valores_receita = dados["receita"] / MILHAO_PARA_BILHAO
    valores_dso = pd.to_numeric(dados["dso"], errors="coerce")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=dados["periodo"], y=valores_receita, name="Receita", marker={"color": cores["serie"][0]},
        hovertemplate="Receita: US$ %{y:,.2f} bi<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=dados["periodo"], y=valores_dso, name="DSO", mode="lines+markers",
        line={"color": cores["serie"][1], "width": 2}, marker={"size": 8, "color": cores["serie"][1]},
        hovertemplate="DSO: %{y:,.2f} dias<extra></extra>",
    ), secondary_y=True)
    _layout(fig, cores, "Receita e DSO (contas a receber)", "US$ bi", 2)
    fig.update_yaxes(title_text="US$ bi", secondary_y=False)
    fig.update_yaxes(title_text="dias", secondary_y=True, showgrid=False)
    return fig


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
    metrica = METRICAS[coluna]
    # Valores fora do ranking ficam no fim, com uma barra cinza curta e um rotulo:
    # P/L e EV/EBITDA zero, negativo (prejuizo) ou ausente; margens negativas.
    excluidas, rotulo, explicacao = dados.iloc[0:0], "", ""
    if coluna in SO_POSITIVOS:
        valido = dados[coluna] > 0
        excluidas = dados[~valido].sort_values("Empresa")
        serie = dados[valido].copy()
        rotulo, explicacao = "prejuízo / n.d.", "não calculado (prejuízo ou dado ausente)"
    elif coluna in MARGENS:
        serie = dados.dropna(subset=[coluna])
        excluidas = serie[serie[coluna] < 0].sort_values(coluna, ascending=False)
        serie = serie[serie[coluna] >= 0].copy()
        rotulo, explicacao = "margem negativa", "margem negativa"
    else:
        serie = dados.dropna(subset=[coluna]).copy()
    serie = serie.sort_values(coluna, ascending=RANKING[coluna] == "menor")
    sentido = "menor é melhor" if RANKING[coluna] == "menor" else "maior é melhor"
    fig = go.Figure(go.Bar(x=serie[coluna], y=serie["Empresa"], orientation="h",
                           marker={'color': [cores["serie"][0] if empresa == destaque else cores["neutro"] for empresa in serie["Empresa"]]},
                           text=[numero(v) for v in serie[coluna]], textposition="outside", cliponaxis=False,
                           customdata=serie[["Setor", "Período"]].values,
                           hovertemplate=f"<b>%{{y}}</b><br>{metrica.label}: %{{text}}<br>%{{customdata[0]}} · %{{customdata[1]}}<extra></extra>"))
    if not excluidas.empty:
        comprimento = serie[coluna].max() * 0.06 if not serie.empty and serie[coluna].max() > 0 else 1
        # Nas margens o valor real vai junto no hover; em P/L e EV/EBITDA ele nao tem significado.
        valores = [f" ({numero(v)})" if coluna in MARGENS else "" for v in excluidas[coluna]]
        fig.add_trace(go.Bar(x=[comprimento] * len(excluidas), y=excluidas["Empresa"], orientation="h",
                             marker={'color': [cores["serie"][0] if empresa == destaque else cores["neutro"] for empresa in excluidas["Empresa"]],
                                     'opacity': 0.35, 'line': {'width': 0}},
                             text=[rotulo] * len(excluidas), textposition="outside", cliponaxis=False,
                             textfont={'color': cores["texto_2"]},
                             customdata=np.column_stack([excluidas[["Setor", "Período"]].values, valores]),
                             hovertemplate=(f"<b>%{{y}}</b><br>{metrica.label}: {explicacao}%{{customdata[2]}}"
                                            "<br>%{customdata[0]} · %{customdata[1]}<extra></extra>")))
    total = len(serie) + len(excluidas)
    fig.update_layout(title={'text': f"{metrica.label} · {sentido}", 'font': {'size': 15, 'color': cores["texto"]}, 'x': 0, 'xanchor': "left"},
                      height=max(260, 26 * total + 90), margin={'l': 8, 'r': 64, 't': 52, 'b': 8}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font={'color': cores["texto_2"], 'size': 12}, showlegend=False, bargap=0.3)
    fig.update_xaxes(showgrid=True, gridcolor=cores["grade"], zeroline=True, zerolinecolor=cores["grade"], title=metrica.unidade)
    fig.update_yaxes(type="category", autorange="reversed", showgrid=False,
                     categoryorder="array", categoryarray=[*serie["Empresa"], *excluidas["Empresa"]])
    return fig


# ---------------------------------------------------------------------------
# Valuation e preco de mercado
# ---------------------------------------------------------------------------
def arquivo_valuation_mais_recente(setor: str, empresa: str) -> Path | None:
    """Localiza o Excel mais recente em ``<setor>/<empresa>``.

    A convencao de nome e o prefixo MMYYYY. Arquivos fora dela (por exemplo
    ``DCF_PLTR_2026.xlsx``) entram pela data de modificacao, com prioridade menor
    que a dos nomeados pela convencao.
    """
    pasta = Path(RAIZ_VALUATIONS) / setor / empresa
    candidatos = []
    for arquivo in pasta.glob("*.xlsx"):
        if arquivo.name.startswith("~$"):          # arquivo temporario do Excel aberto
            continue
        try:
            data, pela_convencao = datetime.strptime(arquivo.name[:6], "%m%Y"), True  # noqa: DTZ007
        except ValueError:
            data, pela_convencao = datetime.fromtimestamp(arquivo.stat().st_mtime), False  # noqa: DTZ006
        candidatos.append((pela_convencao, data, arquivo.name, arquivo))
    return max(candidatos, default=(None, None, None, None))[3]


def _numero_da_celula(valor: object) -> float | None:
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        return None
    return numero if pd.notna(numero) else None


def _valor_valuation(caminho: Path) -> float | None:
    """Le o preco-alvo nos layouts de valuation disponiveis."""
    livro = openpyxl.load_workbook(caminho, read_only=True, data_only=True)

    # Modelos com layout 03_DCF: preco-alvo na celula especifica dessa aba.
    celulas_03_dcf = {
        "amd": "B51", "avgo": "B51", "broadcom": "B51",
        "amat": "B51", "applied_materials": "B51",
        "ceg": "B51", "constellation": "B51", "nrg": "B51", "tln": "B51",
        "kla_corporation": "B51", "klac": "B51",
        "lam_research": "B51", "lrcx": "B51",
        "talen": "B51", "vst": "B51", "vistra": "B51", "oklo": "B55",
        "cadence": "B51", "cdns": "B51",
        "intel": "B51", "intc": "B51", "micron": "B51", "mu": "B51",
        "marvell": "B51", "mrvl": "B51",
        "netflix": "B51", "nflx": "B51",
        "palantir": "B51", "pltr": "B51",
        "cloudflare": "B51", "net": "B51",
        "datadog": "B51", "ddog": "B51",
        "adobe": "B51", "adbe": "B51",
        "servicenow": "B51", "now": "B51",
        "ge_vernova": "B51", "gev": "B51",
        "bwxt": "B51",
        "bloom_energy": "B51", "be": "B51",
        "qualcomm": "B51", "qcom": "B51",
        "synopsys": "B51", "snps": "B51",
    }
    identificadores = [caminho.parent.name.casefold(), *caminho.stem.casefold().split("_")]
    celula_03_dcf = next((celulas_03_dcf[identificador] for identificador in identificadores
                           if identificador in celulas_03_dcf), None)
    if celula_03_dcf and "03_DCF" in livro.sheetnames:
        valor_03_dcf = _numero_da_celula(livro["03_DCF"][celula_03_dcf].value)
        if valor_03_dcf is not None:
            return valor_03_dcf

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
def carregar_precos_fechamento(tickers: tuple[str, ...], periodo: str, auto_adjust: bool = True) -> pd.DataFrame:
    """Baixa os fechamentos ajustados dos ativos selecionados em uma unica consulta."""
    import yfinance as yf

    yf.cache.set_cache_location(str(Path(tempfile.gettempdir()) / "analise-fundamentalista-yfinance"))
    try:
        dados = yf.download(list(tickers), period=periodo, interval="1d", auto_adjust=auto_adjust,
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


def tabela_dcf_valuation(catalogo: dict) -> pd.DataFrame:
    """Consolida o ultimo fechamento e o DCF mais recente de cada empresa."""
    empresas = [
        (Path(caminho).stem.removesuffix("_indicators").upper(), setor, empresa)
        for setor, empresas_setor in catalogo.items()
        for empresa, caminho in empresas_setor.items()
    ]
    tickers = tuple(ticker for ticker, _, _ in empresas)
    precos = carregar_precos_fechamento(tickers, "5d", auto_adjust=False) if tickers else pd.DataFrame()
    registros = []
    for ticker, setor_empresa, empresa in sorted(empresas):
        preco_fechamento = (
            pd.to_numeric(precos[ticker], errors="coerce").dropna().iloc[-1]
            if ticker in precos and precos[ticker].notna().any() else float("nan")
        )
        arquivo_valuation = arquivo_valuation_mais_recente(setor_empresa, empresa)
        valor_valuation = (
            carregar_valuation(str(arquivo_valuation), arquivo_valuation.stat().st_mtime)
            if arquivo_valuation is not None else None
        )
        variacao = (
            (preco_fechamento / valor_valuation - 1) * 100
            if valor_valuation is not None and valor_valuation != 0 and pd.notna(preco_fechamento)
            else float("nan")
        )
        status = (
            "Sobrevalorizada" if variacao > 0 else "Subvalorizada" if variacao < 0 else "Em linha"
        ) if pd.notna(variacao) else "—"
        upside_downside = (
            (valor_valuation / preco_fechamento - 1) * 100
            if status in {"Subvalorizada", "Sobrevalorizada"} else float("nan")
        )
        registros.append({
            "Ticker": ticker,
            "Preço de fechamento": preco_fechamento,
            "Valuation": valor_valuation,
            "Upside/Downside": upside_downside,
            "Status": status,
            "Fonte": arquivo_valuation.name if arquivo_valuation is not None else "—",
        })
    return pd.DataFrame(registros)


def cor_variacao_valuation(valor: float) -> str:
    """Define a cor do texto conforme o sinal da variação para a tabela de DCF."""
    if pd.isna(valor):
        return ""
    if valor > 0:
        return "color: #1baf7a; font-weight: 600"
    if valor < 0:
        return "color: #e34948; font-weight: 600"
    return ""


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


JANELA_BOLLINGER = 20


def _rgba(cor_hex: str, alfa: float) -> str:
    """Converte uma cor #rrggbb da paleta em rgba, para preenchimentos discretos."""
    r, g, b = (int(cor_hex.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alfa})"


def bandas_bollinger(fechamento: pd.Series, janela: int = JANELA_BOLLINGER) -> pd.DataFrame:
    """Media movel do fechamento e bandas de 2 e 3 desvios-padrao."""
    valores = pd.to_numeric(fechamento, errors="coerce")
    media = valores.rolling(janela, min_periods=janela).mean()
    desvio = valores.rolling(janela, min_periods=janela).std(ddof=0)
    return pd.DataFrame({
        "media": media,
        "sup_2": media + 2 * desvio, "inf_2": media - 2 * desvio,
        "sup_3": media + 3 * desvio, "inf_3": media - 3 * desvio,
    }, index=valores.index)


def candlestick_bollinger(precos: pd.DataFrame, ticker: str, cores: dict,
                          janela: int = JANELA_BOLLINGER) -> go.Figure:
    """Candlestick diario com as bandas de Bollinger de 2 e 3 desvios-padrao."""
    bandas = bandas_bollinger(precos["Close"], janela)
    fig = go.Figure()

    # As bandas entram primeiro para ficarem atras dos candles; o traco pontilhado
    # e a legenda distinguem 3 de 2 desvios sem competir com as cores dos candles.
    for desvios, tracejado, largura in ((3, "dot", 1), (2, "solid", 1.2)):
        for extremo, sinal, preenchimento in (("sup", "+", None), ("inf", "-", "tonexty" if desvios == 2 else None)):
            fig.add_trace(go.Scatter(
                x=bandas.index, y=bandas[f"{extremo}_{desvios}"],
                name=f"Bollinger {desvios}σ", legendgroup=f"bollinger_{desvios}",
                showlegend=extremo == "sup", mode="lines",
                line={"color": cores["neutro"], "width": largura, "dash": tracejado},
                fill=preenchimento, fillcolor=_rgba(cores["neutro"], 0.10),
                hovertemplate=f"{sinal}{desvios}σ: US$ %{{y:,.2f}}<extra></extra>",
            ))
    fig.add_trace(go.Scatter(
        x=bandas.index, y=bandas["media"], name=f"Média móvel ({janela} pregões)", mode="lines",
        line={"color": cores["serie"][3], "width": 2},
        hovertemplate="Média: US$ %{y:,.2f}<extra></extra>",
    ))
    fig.add_trace(go.Candlestick(
        x=precos.index, open=precos["Open"], high=precos["High"], low=precos["Low"],
        close=precos["Close"], name=ticker, showlegend=False,
        increasing_line_color=cores["positivo"], decreasing_line_color=cores["negativo"],
        increasing_fillcolor=cores["positivo"], decreasing_fillcolor=cores["negativo"],
        line={"width": 1},
    ))

    fig.update_layout(
        title={'text': f"{ticker} — preços e bandas de Bollinger", 'font': {'size': 15, 'color': cores["texto"]},
               'x': 0, 'xanchor': "left"},
        height=520, margin={'l': 8, 'r': 64, 't': 56, 'b': 58},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={'color': cores["texto_2"], 'size': 12}, hovermode="closest",
        legend={'orientation': "h", 'yanchor': "top", 'y': -0.12, 'xanchor': "left", 'x': 0,
                'bgcolor': "rgba(0,0,0,0)", 'font': {'color': cores["texto_2"]}},
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(type="date", showgrid=False, linecolor=cores["grade"], tickcolor=cores["grade"], ticks="outside")
    fig.update_yaxes(showgrid=True, gridcolor=cores["grade"], zeroline=False,
                     title={'text': "US$ por ação", 'font': {'size': 11, 'color': cores["texto_2"]}})
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
        title={'text': f"{ticker} - preço de mercado e valuation", 'font': {'size': 15, 'color': cores["texto"]}, 'x': 0, 'xanchor': "left"},
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


RESULTADO_LPA = {
    "Acima": "Acima da projeção",
    "Abaixo": "Abaixo da projeção",
    "Em linha": "Em linha com a projeção",
}


@st.cache_data(show_spinner=False)
def carregar_projecoes(caminho: str, ticker: str, _mtime: float) -> pd.DataFrame | None:
    """LPA divulgado e projecao do mercado de uma empresa em ``projecoes_<setor>.xlsx``.

    Cada aba e uma empresa (nome da aba = ticker). A primeira linha traz o ticker e a
    segunda, o cabecalho ``Data | Exercício | LPA | Projeção``. O exercicio vem como
    texto ``MM/AAAA`` ou como data, conforme o arquivo. Devolve None se nao houver aba.
    """
    aba = next((nome for nome in pd.ExcelFile(caminho).sheet_names
                if nome.strip().lower() == ticker.lower()), None)
    if aba is None:
        return None
    bruto = pd.read_excel(caminho, sheet_name=aba, header=1)
    bruto.columns = [_chave_coluna(coluna) for coluna in bruto.columns]
    exercicio = bruto["exercicio"].map(
        lambda valor: pd.Timestamp(valor) if isinstance(valor, datetime)
        else pd.to_datetime(str(valor).strip(), format="%m/%Y", errors="coerce"))
    df = pd.DataFrame({
        "divulgacao": pd.to_datetime(bruto["data"], errors="coerce"),
        "exercicio": exercicio,
        "lpa": pd.to_numeric(bruto["lpa"], errors="coerce"),
        "projecao": pd.to_numeric(bruto["projecao"], errors="coerce"),
    }).dropna(subset=["exercicio", "lpa", "projecao"]).sort_values("exercicio").reset_index(drop=True)

    limite = TOLERANCIA_LPA + 1e-9          # folga para o arredondamento de ponto flutuante
    df["periodo"] = df["exercicio"].dt.strftime("%m/%Y")
    df["surpresa"] = df["lpa"] - df["projecao"]
    # Percentual so quando a projecao nao e praticamente zero; senao ele explode.
    df["surpresa_pct"] = (df["surpresa"] / df["projecao"].abs()).where(df["projecao"].abs() >= limite) * 100
    df["resultado"] = np.select([df["surpresa"] > limite, df["surpresa"] < -limite],
                                ["Acima", "Abaixo"], default="Em linha")
    return df


def grafico_lpa_projecao(projecoes: pd.DataFrame, ticker: str, cores: dict) -> go.Figure:
    """LPA divulgado contra a projecao do mercado em cada trimestre."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=projecoes["periodo"], y=projecoes["projecao"], name="Projeção do mercado", mode="lines+markers",
        line={"color": cores["neutro"], "width": 2, "dash": "dash"},
        marker={"symbol": "circle-open", "size": 9, "color": cores["neutro"], "line": {"width": 2}},
        hovertemplate="Projeção: US$ %{y:,.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=projecoes["periodo"], y=projecoes["lpa"], name="LPA divulgado", mode="lines+markers",
        line={"color": cores["serie"][0], "width": 2}, marker={"size": 8, "color": cores["serie"][0]},
        customdata=projecoes["resultado"].map(RESULTADO_LPA),
        hovertemplate="LPA: US$ %{y:,.2f} · %{customdata}<extra></extra>",
    ))
    _layout(fig, cores, f"{ticker} — LPA divulgado e projeção do mercado — Site Investing", "US$ por ação", 2)
    fig.update_layout(height=420)
    return fig


def grafico_surpresa_lpa(projecoes: pd.DataFrame, cores: dict) -> go.Figure:
    """Diferenca entre LPA divulgado e projecao; a cor indica acima, abaixo ou em linha."""
    cor = {"Acima": cores["positivo"], "Abaixo": cores["negativo"], "Em linha": cores["neutro"]}
    percentual = [f"{valor:+.2f}%" if pd.notna(valor) else "projeção ≈ 0"
                  for valor in projecoes["surpresa_pct"]]
    surpresa = [f"US$ {float(valor):+,.2f}" if pd.notna(valor) else "—" for valor in projecoes["surpresa"]]
    fig = go.Figure(go.Bar(
        x=projecoes["periodo"], y=projecoes["surpresa"], name="Surpresa",
        marker={"color": projecoes["resultado"].map(cor).tolist(), "line": {"width": 0}},
        customdata=np.column_stack([projecoes["resultado"].map(RESULTADO_LPA), percentual,
                                    projecoes["divulgacao"].dt.strftime("%d/%m/%Y").fillna("—"), surpresa]),
        hovertemplate=("%{customdata[0]}<br>Surpresa: %{customdata[3]} (%{customdata[1]})"
                       "<br>Divulgado em %{customdata[2]}<extra></extra>"),
    ))
    return _layout(fig, cores, "Surpresa do LPA (divulgado − projeção)", "US$ por ação", 1)


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
ticker_arquivo = arquivo.stem.removesuffix("_indicators")
arquivo_resultados = arquivo.with_name(f"{ticker_arquivo}_{tipo.removeprefix('indicators_')}.xlsx")
aviso_lpa = None
if arquivo_resultados.is_file():
    lpa_publicado = carregar_lpa(str(arquivo_resultados), tipo, arquivo_resultados.stat().st_mtime)
    if not lpa_publicado.empty:
        desdobramentos = carregar_desdobramentos(ticker_arquivo.upper())
        if desdobramentos is None:
            aviso_lpa = ("Não foi possível obter o histórico de desdobramentos no Yahoo Finance; "
                         "o LPA aparece como publicado, sem ajuste.")
            desdobramentos = pd.Series(dtype=float)
        lpa_publicado = lpa_publicado.assign(lpa=ajustar_desdobramentos(lpa_publicado, desdobramentos, tipo))
        df = df.merge(lpa_publicado[["periodo", "lpa"]], on="periodo", how="left")
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

aba_sobre, *abas = st.tabs(["Sobre a empresa", "Valuation", "Rentabilidade", "Endividamento", "Fluxo de caixa", "Proventos", "Fraude", "Explorar", "Dados", "Comparação múltiplos", "Indicadores gerais", "DCF valuation", "DCF valuation tabela", "LPA vs projeção"])

def existentes(*colunas: str) -> list[str]:
    return [coluna for coluna in colunas if coluna in disponiveis]

with aba_sobre:
    st.subheader(f"{titulo_empresa} ({ticker_arquivo.upper()})")
    st.write(SOBRE_EMPRESA.get(ticker_arquivo.lower(),
                               "Ainda não há texto para esta empresa. Inclua-o em `SOBRE_EMPRESA`, no início do app."))
    # Grafo de receita: <raiz>/<setor>/<empresa>/<ticker>_grafo_receita_<periodo>.html (o mais recente).
    pasta_grafo = Path(RAIZ_GRAFO_RECEITA) / setor / empresa
    grafos = sorted(pasta_grafo.glob(f"{ticker_arquivo.lower()}_grafo_receita_*.html"), key=lambda p: p.stat().st_mtime)
    if grafos:
        st.subheader("Como a empresa gera receita")
        st.caption(f"Grafo interativo: passe o mouse ou clique nos nós para ver o detalhe · {grafos[-1].name}")
        components.html(grafos[-1].read_text(encoding="utf-8"), height=ALTURA_GRAFO_RECEITA, scrolling=True)

with abas[0]:
    bloco(df, cores, [(linhas, existentes("pl_damodaran", "pl", "pvp", "ev_ebitda"), "Múltiplos de mercado"),
                       (linhas, existentes("lp"), "Earnings yield (L/P)"),
                       (barras, existentes("valor_mercado"), "Valor de mercado"),
                       (barras, existentes("lpa"), "LPA (ajustado por desdobramentos)")])
    if aviso_lpa:
        st.caption(aviso_lpa)

with abas[10]:
    arquivo_valuation = arquivo_valuation_mais_recente(setor, empresa)
    st.subheader("Preço de mercado e valuation")
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
            intervalo_candlestick = st.selectbox("Históricos de preços", list(opcoes_candlestick), index=1)
            ticker_acao = arquivo.stem.rsplit("_", 1)[0].upper()
            with st.spinner(f"Baixando precos de {ticker_acao}..."):
                precos = carregar_ohlc(ticker_acao, opcoes_candlestick[intervalo_candlestick])
            if precos.empty:
                st.warning(f"Nao foi possivel obter as cotacoes OHLC de {ticker_acao}.")
            else:
                preco_fechamento = float(precos["Close"].iloc[-1])
                if preco_fechamento > valor_valuation:
                    situacao = "Sobrevalorizada"
                    rotulo_percentual = "Sobrevalorização"
                elif preco_fechamento < valor_valuation:
                    situacao = "Subvalorizada"
                    rotulo_percentual = "Subvalorização"
                else:
                    situacao = "Em linha com o valuation"
                    rotulo_percentual = "Diferença para o valuation"
                diferenca_percentual = abs(preco_fechamento / valor_valuation - 1) if valor_valuation else float("nan")
                coluna_preco, coluna_valuation, coluna_situacao, coluna_percentual = st.columns(4)
                coluna_preco.metric("Preço de fechamento", f"US$ {preco_fechamento:,.2f}")
                coluna_valuation.metric("Valuation", f"US$ {valor_valuation:,.2f}")
                coluna_situacao.metric("Situação", situacao)
                coluna_percentual.metric(rotulo_percentual, f"{diferenca_percentual:.2%}" if pd.notna(diferenca_percentual) else "—")
                st.plotly_chart(candlestick(precos, ticker_acao, valor_valuation, arquivo_valuation, cores), width="stretch")

with abas[11]:
    st.subheader("Tabela de DCF Valuation")
    with st.spinner("Lendo valuations e preços de fechamento..."):
        tabela_valuation = tabela_dcf_valuation(catalogo)
    st.dataframe(
        tabela_valuation.style.map(cor_variacao_valuation, subset=["Upside/Downside"]),
        width="stretch",
        hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker"),
            "Preço de fechamento": st.column_config.NumberColumn("Preço de fechamento", format="US$ %.2f"),
            "Valuation": st.column_config.NumberColumn("Valuation", format="US$ %.2f"),
            "Upside/Downside": st.column_config.NumberColumn("Upside/Downside", format="%.2f%%"),
            "Status": st.column_config.TextColumn("Status"),
            "Fonte": st.column_config.TextColumn("Fonte"),
        },
    )

with abas[1]:
    bloco(df, cores, [(linhas, existentes("roe", "roic"), "ROE e ROIC"),
                       (linhas, existentes("margem_liquida", "margem_bruta", "margem_operacional", "margem_ebitda"), "Margens"),
                       (barras, existentes("ebitda"), "EBITDA"),
                       (linhas, existentes("reinvestment_rate"), "Taxa de reinvestimento")])
with abas[2]:
    bloco(df, cores, [(barras, existentes("divida_bruta", "caixa", "divida_liquida"), "Dívida bruta, caixa e dívida líquida"),
                       (linhas, existentes("dl_ebitda", "dl_pl"), "Alavancagem")])
with abas[3]:
    bloco(df, cores, [(barras, existentes("fco", "fci", "fcf"), "Fluxos de caixa (FCO, FCI, FCF)"),
                       (barras, existentes("free_cash_flow", "adjusted_free_cash_flow", "sbc"), "Free cash flow e SBC"),
                       (linhas_com_referencia_zero, existentes("free_cash_flow_yield"), "Free cash flow yield"),
                       (barras, existentes("capex", "net_capex", "rd", "adj_net_capex"), "Capex e P&D"),
                       (barras, existentes("fcfe", "fcff"), "FCFE e FCFF"),
                       (barras, existentes("working_capital"), "Capital de giro"),
                       (linhas, existentes("dso", "dio", "dpo"), "Prazos operacionais (DSO, DIO, DPO)")])
with abas[4]:
    bloco(df, cores, [(barras, existentes("buyback"), "Recompra de ações"),
                       (linhas, existentes("dpa"), "Dividendos por ação"),
                       (linhas, existentes("payout"), "Payout")])
with abas[5]:
    bloco(df, cores, [
        (linhas, existentes("cfo_ll", "fcf_ll"), "Qualidade dos lucros"),
        (linhas, existentes("cfo_ebitda", "capex_depreciacao"), "Conversão de caixa e reinvestimento"),
        (linhas, existentes("accrual_ratio", "recebiveis_receita"), "Accruals e recebíveis"),
    ])
    if "dso" in disponiveis:
        sufixo_demonstrativo = tipo.removeprefix("indicators_")
        arquivo_demonstrativo = arquivo.with_name(f"{arquivo.stem.removesuffix('_indicators')}_{sufixo_demonstrativo}.xlsx")
        if arquivo_demonstrativo.is_file():
            receita = carregar_receita(
                str(arquivo_demonstrativo), tipo, arquivo_demonstrativo.stat().st_mtime
            )
            grafico_receita_dso = receita_dso(df, receita, cores)
            if grafico_receita_dso is not None:
                st.plotly_chart(grafico_receita_dso, width="stretch")
            else:
                st.info("Não há períodos em comum entre Receita e DSO.")
        else:
            st.info("O demonstrativo de resultados necessário para comparar Receita e DSO não foi encontrado.")
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
            st.caption(f"Candlesticks diários com bandas de Bollinger de 2 e 3 desvios-padrão "
                       f"sobre a média móvel de {JANELA_BOLLINGER} pregões.")
            for ticker_acao in tickers:
                with st.spinner(f"Baixando cotações OHLC de {ticker_acao}..."):
                    ohlc = carregar_ohlc(ticker_acao, periodos_mercado[periodo_mercado])
                if ohlc.empty:
                    st.warning(f"Não foi possível obter as cotações OHLC de {ticker_acao}.")
                else:
                    st.plotly_chart(candlestick_bollinger(ohlc, ticker_acao, cores),
                                    width="stretch", key=f"candlestick_bollinger_{ticker_acao}")
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

with abas[12]:
    ticker_projecao = arquivo.stem.removesuffix("_indicators")
    arquivo_projecoes = Path(RAIZ_PROJECOES) / f"projecoes_{setor}.xlsx"
    projecoes = (carregar_projecoes(str(arquivo_projecoes), ticker_projecao, arquivo_projecoes.stat().st_mtime)
                 if arquivo_projecoes.is_file() else None)
    if projecoes is None or projecoes.empty:
        st.info(f"Não há projeções de LPA para {ticker_projecao.upper()} em `{arquivo_projecoes}`.")
    else:
        ultimo = projecoes.iloc[-1]
        acima = int((projecoes["resultado"] == "Acima").sum())
        abaixo = int((projecoes["resultado"] == "Abaixo").sum())
        coluna_lpa, coluna_projecao, coluna_batidas, coluna_mediana = st.columns(4)
        coluna_lpa.metric(f"LPA divulgado · {ultimo['periodo']}", f"US$ {ultimo['lpa']:,.2f}",
                          f"{ultimo['surpresa']:+,.2f} vs projeção")
        coluna_projecao.metric(f"Projeção do mercado · {ultimo['periodo']}", f"US$ {ultimo['projecao']:,.2f}")
        coluna_batidas.metric("Trimestres acima da projeção", f"{acima} de {len(projecoes)}",
                              help=f"{abaixo} abaixo e {len(projecoes) - acima - abaixo} em linha.")
        mediana = projecoes["surpresa_pct"].median()
        coluna_mediana.metric("Surpresa mediana", f"{mediana:+.1f}%" if pd.notna(mediana) else "—",
                              help="Mediana da surpresa percentual; menos sensível a trimestres "
                                   "com projeção próxima de zero do que a média.")

        st.plotly_chart(grafico_lpa_projecao(projecoes, ticker_projecao.upper(), cores),
                        width="stretch", key="lpa_projecao")
        st.plotly_chart(grafico_surpresa_lpa(projecoes, cores), width="stretch", key="lpa_surpresa")
        st.caption("Exercício = mês de encerramento do trimestre fiscal. Barras azuis: LPA acima da projeção; "
                   "vermelhas: abaixo; cinza: em linha (diferença de até meio centavo). "
                   f"Fonte: {arquivo_projecoes.name}, aba {ticker_projecao}.")

        with st.expander("Tabela"):
            st.dataframe(
                projecoes[["divulgacao", "periodo", "lpa", "projecao", "surpresa", "surpresa_pct", "resultado"]]
                .iloc[::-1],
                width="stretch", hide_index=True,
                column_config={
                    "divulgacao": st.column_config.DateColumn("Divulgação", format="DD/MM/YYYY"),
                    "periodo": st.column_config.TextColumn("Exercício"),
                    "lpa": st.column_config.NumberColumn("LPA (US$)", format="%.3f"),
                    "projecao": st.column_config.NumberColumn("Projeção (US$)", format="%.3f"),
                    "surpresa": st.column_config.NumberColumn("Surpresa (US$)", format="%+.3f"),
                    "surpresa_pct": st.column_config.NumberColumn("Surpresa (%)", format="%+.1f%%"),
                    "resultado": st.column_config.TextColumn("Resultado"),
                },
            )
