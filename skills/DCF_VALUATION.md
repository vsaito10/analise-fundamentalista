---
name: dcf-valuation
description: "Construa modelos completos de valuation DCF (Discounted Cash Flow) para empresas de capital aberto ou fechado. Use esta skill sempre que o usuário mencionar valuation, DCF, fluxo de caixa descontado, WACC, FCFF, valor terminal, equity value, preço-alvo, análise de sensibilidade, comparáveis (comps), múltiplos EV/EBITDA, ou qualquer tarefa relacionada a precificação de ativos e modelagem financeira. Acione também quando o usuário quiser montar uma planilha de valuation, projetar DRE/balanço/fluxo de caixa, calcular custo de capital ou avaliar upside de uma ação."
---
 
# DCF Valuation — Guia de Modelagem
 
Este guia define a estrutura completa de um modelo DCF profissional. O modelo possui **5 abas** organizadas da seguinte forma:
 
---
 
## Estrutura das Abas (ordem obrigatória)
 
```
01_Cover            → Identificação do ativo
02_Historicals      → DRE, Balanço e DFC históricos (extraídos de SEC 10-K / 10-Q)
03_DCF              → WACC, Terminal Value, Equity Bridge, UFCF Projections, DCF Valuation
04_IS_Estimates     → Income Statement projetado
05_CFS_Estimates    → Cash Flow Statement projetado
```
 
---
 
## 01 — Cover
 
Aba informativa. Não contém fórmulas.
 
| Campo       | Exemplo            |
|-------------|--------------------|
| Empresa     | Apple Inc.         |
| Ticker      | AAPL               |
| Modelo      | DCF Valuation      |
| Data        | =TODAY()           |
| Fonte       | SEC EDGAR (10-K/10-Q) |
| Analista    | (nome do usuário)  |
 
---
 
## 02 — Historicals
 
> **Fonte de dados:** SEC EDGAR — filings 10-K (anual) e 10-Q (trimestral).
> Claude **deve usar web search e web fetch** para acessar os links da SEC fornecidos pelo usuário (ou buscar no EDGAR) e extrair os dados financeiros reais dos filings.
 
Cobrir **3 a 5 anos** históricos, organizados em 3 blocos na mesma aba:
 
### 2A — Income Statement (Demonstração de Resultado)
 
```
Revenue (Receita Líquida)
  (−) Cost of Revenue (COGS)
= Gross Profit (Lucro Bruto)
  (−) Operating Expenses (SG&A, R&D, etc.)
= Operating Income (EBIT)
  (−/+) Interest Expense / Income
  (−/+) Other Income/Expense
= Pre-Tax Income (EBT)
  (−) Income Tax Expense
= Net Income (Lucro Líquido)
 
Métricas derivadas:
  Gross Margin %
  Operating Margin %
  Net Margin %
  EBITDA = EBIT + D&A
  EBITDA Margin %
```
 
### 2B — Balance Sheet (Balanço Patrimonial)
 
```
ASSETS
  Cash & Equivalents
  Short-term Investments
  Accounts Receivable
  Inventories
  Other Current Assets
  Total Current Assets
  PP&E (net)
  Goodwill & Intangibles
  Other Non-Current Assets
  Total Assets
 
LIABILITIES & EQUITY
  Accounts Payable
  Short-term Debt
  Other Current Liabilities
  Total Current Liabilities
  Long-term Debt
  Other Non-Current Liabilities
  Total Liabilities
  Total Stockholders' Equity
  Total Liabilities & Equity
 
Check: Total Assets − Total Liabilities & Equity = 0
```
 
### 2C — Cash Flow Statement (Demonstração de Fluxo de Caixa)
 
```
Cash from Operating Activities
  Net Income
  D&A
  Stock-Based Compensation
  Changes in Working Capital
  Other Operating Adjustments
  Net Cash from Operations (CFO)
 
Cash from Investing Activities
  Capital Expenditures (Capex)
  Acquisitions
  Purchases/Sales of Investments
  Net Cash from Investing (CFI)
 
Cash from Financing Activities
  Debt Issuance / Repayment
  Share Repurchases
  Dividends Paid
  Net Cash from Financing (CFF)
 
Net Change in Cash
Beginning Cash
Ending Cash
```
 
> **Regra:** Todos os valores devem vir diretamente dos filings da SEC. Não inventar, estimar ou arredondar. Se um dado não estiver disponível, deixar a célula com "N/A" em vermelho.
 
---
 
## 03 — DCF
 
Esta aba consolida toda a mecânica de valuation em **5 seções**:
 
### 3A — WACC Assumptions
 
Todos os inputs de WACC ficam aqui, em **texto azul** (hardcoded inputs).
 
| Parâmetro                         | Célula | Nota                            |
|-----------------------------------|--------|---------------------------------|
| Risk-Free Rate (Rf)               |        | US 10Y Treasury Yield — **BUSCAR VIA WEB SEARCH no ETF TNX (CBOE 10 YR TREASURY NOTE YIELD). Se for uma empresa brasileira,  BUSCAR VIA WEB SEARCH a Taxa Selic.** |
| Equity Risk Premium (ERP)         |        | Damodaran ou fonte equivalente  |
| Beta (Levered)                    |        | Yahoo Finance / Bloomberg       |
| Cost of Debt (Rd)                 |        | Interest Expense / Total Debt   |
| Tax Rate (T)                      |        | Média da Effective Tax Rate dos anos projetados em `04_IS_Estimates` — calcular como `AVERAGE` das células de `Tax Expense / Pre-Tax Income` dos 5 anos projetados; deve ser uma fórmula referenciando `04_IS_Estimates`, nunca um valor hardcoded |
| Equity Weight — E/(D+E)           |        | Market cap based                |
| Debt Weight — D/(D+E)             |        | Total debt based                |
 
**Cálculos derivados (texto preto — fórmulas):**
 
> ⚠️ **REGRA CRÍTICA — TAX RATE (T) NO WACC:**
> O parâmetro **Tax Rate (T)** da seção 3A deve ser obrigatoriamente a **média da Effective Tax Rate dos anos projetados** da aba `04_IS_Estimates` — **não** a alíquota dos históricos.
> - Effective Tax Rate por ano projetado = `Tax Expense / Pre-Tax Income` (para cada coluna de projeção em `04_IS_Estimates`)
> - Tax Rate (T) no WACC = `AVERAGE(ETR_Y1, ETR_Y2, ETR_Y3, ETR_Y4, ETR_Y5)`
> - A célula deve ser uma **fórmula com link para `04_IS_Estimates`** — zero hardcode.
> - Isso garante que qualquer mudança nas premissas de impostos em `04_IS_Estimates` se propague automaticamente para o WACC e, em cadeia, para o Target Price.
 
> ⚠️ **REGRA CRÍTICA — RISK-FREE RATE (Rf):**
> Claude **DEVE obrigatoriamente buscar o valor atual do Risk-Free Rate via web search** usando o ticker **TNX** (CBOE 10 YR Treasury Note Yield) antes de montar o modelo.
> - Buscar a cotação atual do TNX, que representa o yield do título do Tesouro Americano de 10 anos em percentual (ex.: 4.32 = 4,32%).
> - Converter para decimal no modelo: `Rf = TNX_valor / 100`
> - Registrar a data e o valor obtido em comentário na célula correspondente.
> - Se a busca falhar, deixar a célula em **vermelho com "INSERIR Rf MANUALMENTE (TNX)"** e alertar o usuário.
> - **Nunca** usar um valor estimado ou fixo sem verificação — o Rf deve sempre refletir o yield corrente do mercado.
 
```
Cost of Equity (Re) = Rf + β × ERP
After-Tax Cost of Debt = Rd × (1 − T)
WACC = Re × [E/(D+E)] + Rd×(1−T) × [D/(D+E)]
```
 
### 3B — Terminal Value Assumptions
 
| Parâmetro                         | Célula | Nota                            |
|-----------------------------------|--------|---------------------------------|
| Long-term Growth Rate (g)         |        | Tipicamente 2–3% (≤ GDP growth) |
| Exit EV/EBITDA Multiple           |        | Para cross-check via múltiplos  |
 
**Cálculos:**
 
```
Método Gordon Growth:
  TV = FCFF_último_ano × (1 + g) / (WACC − g)
 
Método Múltiplos (cross-check):
  TV = EBITDA_último_ano × Exit Multiple
 
PV(Terminal Value) = TV / (1 + WACC)^n
```
 
> Usar Gordon Growth como método principal e múltiplos como sanity check.
 
### 3C — Equity Bridge Assumptions
 
| Parâmetro                         | Célula | Nota                            |
|-----------------------------------|--------|---------------------------------|
| Total Debt                        |        | Do último balanço (02_Historicals) |
| Cash & Equivalents                |        | Do último balanço (02_Historicals) |
| Minority Interest                 |        | Se aplicável                    |
| Shares Outstanding                |        | Diluted shares (10-K)           |
| Current Share Price               |        | **BUSCAR VIA WEB SEARCH — JAMAIS INVENTAR** |
 
```
Enterprise Value = Σ PV(FCFF) + PV(Terminal Value)
(−) Net Debt = Total Debt − Cash
(−) Minority Interest
= Equity Value
 
Equity Value ÷ Shares Outstanding = Target Price
Upside % = (Target Price / Current Price) − 1
```
 
> ⚠️ **REGRA CRÍTICA — PREÇO ATUAL DA AÇÃO:**
> Claude **DEVE obrigatoriamente buscar o preço de mercado via web search** antes de montar o modelo.
> Se a busca falhar, deixar a célula em **vermelho com "INSERIR PREÇO MANUALMENTE"** e alertar o usuário.
> Nunca preencher com valor aproximado — ou é real e verificado, ou fica vazio.
 
### 3D — Unlevered Free Cash Flow (UFCF) Projections
 
Horizonte de projeção: **5 anos** (ex.: 2026–2030).
 
O UFCF representa o caixa gerado pelo negócio **antes do efeito da estrutura de capital** — ou seja, como se a empresa fosse 100% equity-financed. É o fluxo descontado pelo WACC para se chegar ao Enterprise Value.
 
**Fórmula central:**
 
```
UFCF = EBIT × (1 − T) + D&A − Capex − ΔWorking Capital
     = NOPAT + D&A − Capex − ΔWC
```
 
**Estrutura linha a linha (para cada ano projetado):**
 
```
EBIT  (Operating Income)
  × (1 − Tax Rate)
= NOPAT  (Net Operating Profit After Tax)
  (+) D&A  (Depreciation & Amortization — non-cash add-back)
  (−) Capex  (Capital Expenditures)
  (−) ΔWorking Capital  (aumento de WC = saída de caixa; queda = entrada)
= UFCF
```
 
**Fontes de cada linha:**
 
| Linha              | Fonte                                      |
|--------------------|--------------------------------------------|
| EBIT               | Aba `04_IS_Estimates` — linha EBIT         |
| Tax Rate (T)       | Alíquota efetiva projetada (de `04_IS_Estimates`) — usar por ano ou média dos 5 anos |
| D&A                | Aba `04_IS_Estimates` — linha D&A          |
| Capex              | Aba `05_CFS_Estimates` — linha Capex       |
| ΔWorking Capital   | Aba `05_CFS_Estimates` — variação de WC    |
 
> Todos os valores devem ser **fórmulas referenciando as abas de estimativas** — evitar hardcodes no bloco de UFCF.
 
**Boas práticas:**
 
- **D&A:** sempre positivo (add-back); vem da DRE ou do CFS histórico.
- **Capex:** sempre negativo (saída de caixa); usar valor bruto do CFS.
- **ΔWC:** positivo quando o WC aumenta (caixa sai); negativo quando diminui. Calcular como `WC_ano_N − WC_ano_N-1`, onde `WC = Current Assets (ex-cash) − Current Liabilities (ex-debt)`.
- **Tax Rate:** usar a alíquota efetiva projetada (Tax Expense / Pre-Tax Income). Para consistência, pode-se usar a média dos 5 anos projetados como taxa única, ou aplicar a alíquota ano a ano se houver variação relevante nas premissas.
- **Normalização:** se o EBIT do último ano histórico contiver itens não recorrentes, ajustá-lo antes de usar como base de projeção.
**Teste de integridade:** alterar uma premissa de receita ou margem em `04_IS_Estimates` deve propagar automaticamente para o UFCF e, em seguida, para o Target Price final na seção 3E/3C.
 
### 3E — DCF Valuation (Consolidação)
 
```
Discount Factor por ano: 1 / (1 + WACC)^t
 
PV(UFCF) = UFCF × Discount Factor   (para cada ano)
 
Sum of PV(UFCF)
+ PV(Terminal Value)
= Enterprise Value
 
Aplicar Equity Bridge (seção 3C) → Target Price e Upside %
```
 
Incluir **tabela de sensibilidade** bidimensional:
- Linhas: g (growth rate) — variar ±100bps
- Colunas: WACC — variar ±150bps
- Célula central = cenário-base (destacar com preenchimento amarelo/laranja e borda espessa)
- A célula central da tabela de sensibilidade deve ser obrigatoriamente linkada com a célula **TARGET PRICE** (seção 3C, linha `Equity Value ÷ Shares Outstanding`) — nunca um valor calculado diretamente.
---
 
## 04 — IS Estimates (Income Statement Projetado)
 
Projetar a DRE para o horizonte de 5 anos, lado a lado com os anos históricos (vindos de 02_Historicals).
 
```
                        Hist Y-2 | Hist Y-1 | Hist Y0 | Proj Y1 | Proj Y2 | Proj Y3 | Proj Y4 | Proj Y5
Revenue                     ...       ...       ...       ...       ...       ...       ...       ...
  Revenue Growth %
(−) COGS
= Gross Profit
  Gross Margin %
(−) SG&A
(−) R&D
(−) Other OpEx
= EBIT
  Operating Margin %
(+) D&A (abaixo da linha)
= EBITDA
  EBITDA Margin %
(−) Interest Expense
(−) Tax Expense
= Net Income
  Net Margin %
```
 
**Premissas de projeção** (em azul, no topo da aba):
- Revenue Growth % por ano
- Gross Margin % alvo
- OpEx como % da Receita (SG&A, R&D)
- D&A como % da Receita
- Tax Rate efetiva
> Todas as células de projeção devem ser **fórmulas** referenciando as premissas. Zero hard-codes.
 
---
 
## 05 — CFS Estimates (Cash Flow Statement Projetado)
 
Projetar o fluxo de caixa para o mesmo horizonte de 5 anos.
 
```
                        Hist Y-2 | Hist Y-1 | Hist Y0 | Proj Y1 | Proj Y2 | Proj Y3 | Proj Y4 | Proj Y5
OPERATING ACTIVITIES
  Net Income (de 04_IS)
  (+) D&A
  (+) SBC
  (+/−) ΔWorking Capital
    ΔAccounts Receivable
    ΔInventories
    ΔAccounts Payable
  = Cash from Operations (CFO)
 
INVESTING ACTIVITIES
  (−) Capital Expenditures
  (−/+) Other Investing
  = Cash from Investing (CFI)
 
FINANCING ACTIVITIES
  (+/−) Debt Issuance/Repayment
  (−) Share Repurchases
  (−) Dividends
  = Cash from Financing (CFF)
 
Net Change in Cash = CFO + CFI + CFF
Beginning Cash
Ending Cash
```
 
**Premissas de projeção** (em azul, no topo da aba):
- Capex como % da Receita
- DSO, DIO, DPO (para calcular ΔWC)
- SBC como % da Receita
- Dividend per share ou payout ratio
- Share repurchase budget (se aplicável)
> Net Income deve ser **link** para `04_IS_Estimates`. Capex é usado em `03_DCF` seção 3D.
 
---
 
## Boas Práticas de Modelagem
 
| # | Regra |
|---|-------|
| 1 | **Zero hard-coded nas fórmulas.** Todo número editável fica com texto azul e claramente identificado como premissa. |
| 2 | **Fluxo esquerda → direita:** Histórico à esquerda, projeções à direita. |
| 3 | **Codificação por cor:** inputs em azul, fórmulas em preto, links entre abas em verde. |
| 4 | **Check de balanço:** `Total Assets − Total L&E = 0` em todos os anos. |
| 5 | **Documentar fontes:** cada dado histórico com referência ao filing da SEC (10-K/10-Q, página/seção). |
| 6 | **Versionar o arquivo** com data: `DCF_AAPL_20260316_v1.xlsx`. |
| 7 | **Dados históricos reais:** usar web search/fetch nos links da SEC — nunca inventar números. |
| 8 | **Preço da ação real:** buscar via web search — nunca alucinar. |
| 9 | **Risk-Free Rate (Rf) real:** buscar o yield atual do TNX (CBOE 10 YR Treasury Note Yield) via web search — nunca usar valor fixo ou desatualizado. |
 
---
 
## Checklist de Revisão
 
Antes de entregar o modelo, verificar:
 
- [ ] Dados históricos extraídos de SEC filings reais (10-K / 10-Q)
- [ ] Todos os inputs/premissas identificados em azul
- [ ] Nenhum hard-code em fórmulas de cálculo
- [ ] WACC calculado a partir dos inputs (não digitado)
- [ ] Terminal Value usa `g` < WACC (condição de convergência)
- [ ] Sensibilidade cobre ±150bps em WACC e ±100bps em g
- [ ] Preço atual da ação buscado via web search — nunca alucinado
- [ ] **Risk-Free Rate (Rf) buscado via web search no TNX (CBOE 10 YR Treasury Note Yield)** — nunca usando valor fixo ou desatualizado
- [ ] Check de balanço fecha em todos os anos
- [ ] **Links entre abas funcionando:** EBIT e D&A puxados de `04_IS_Estimates`; Capex e ΔWC puxados de `05_CFS_Estimates` — zero valores hardcoded no bloco UFCF de `03_DCF`
- [ ] **TARGET PRICE linkado à célula central da tabela de sensibilidade** — alterar WACC ou g deve atualizar ambos em cadeia
- [ ] Arquivo recalculado com `scripts/recalc.py` sem erros