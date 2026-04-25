---
name: canvas-model
description: Construa um Business Model Canvas completo de uma empresa específica, preenchendo os 9 blocos com base em informações públicas (site, releases, relatórios anuais, notícias). Use esta skill sempre que o usuário mencionar "canvas", "business model canvas", "BMC", "modelo de negócio", "quadro de modelo de negócios", pedir para "mapear o modelo de negócio" de uma empresa, ou quiser entender visualmente como uma empresa gera, entrega e captura valor. Acione também quando o usuário citar o nome de uma empresa seguido de pedidos como "monte o canvas", "faça o BMC" ou "estrutura de negócio".
---
 
# Canvas Model
 
Skill para construção de um Business Model Canvas (Osterwalder & Pigneur) de uma empresa específica, com base em evidências públicas.
 
## Quando usar
 
Acione sempre que o usuário pedir um Canvas, BMC ou análise do modelo de negócios de uma empresa — aberta ou fechada. Se a empresa não for informada, pergunte antes de começar.
 
## Princípios
 
1. **Empresa-específica, não genérica**: cada bloco deve refletir a empresa real, com nomes de produtos, segmentos, parceiros e canais concretos. Evite generalidades ("clientes diversos", "vários canais").
2. **Baseado em evidência**: use site institucional, relatório anual (10-K, Formulário de Referência, Annual Report), releases de resultados, apresentações a investidores e notícias recentes. Cite fontes quando relevante.
3. **Sucinto**: cada bloco com 3–6 bullets curtos. O Canvas é um mapa, não um relatório.
4. **Atual**: se informações podem ter mudado, faça web search antes de preencher.
## Passo a passo
 
### 1. Coleta
Antes de preencher, reúna:
- O que a empresa vende (produtos/serviços principais)
- Para quem vende (B2B, B2C, segmentos, geografia)
- Como monetiza (assinatura, venda única, comissão, publicidade, licenciamento)
- Estrutura operacional (própria, terceirizada, plataforma, franquia)
- Principais parceiros e fornecedores estratégicos
Se faltar informação, use web search. Para empresas listadas, priorize RI, 10-K/20-F ou Formulário de Referência.
 
### 2. Preencha os 9 blocos (nesta ordem)
 
Preencha nesta sequência — ela reflete a lógica de fora para dentro (cliente → entrega → operação → finanças):
 
**1. Segmentos de Clientes (Customer Segments)**
Para quem a empresa cria valor? Liste segmentos distintos com características específicas (ex.: "PMEs brasileiras com 10–200 funcionários", "consumidores premium 25–45 anos").
 
**2. Proposta de Valor (Value Propositions)**
Que problema resolve ou que necessidade atende para cada segmento? Seja concreto: "redução de X%", "entrega em Y horas", "preço Z% abaixo do mercado". Se há múltiplos segmentos, relacione cada proposta ao segmento correspondente.
 
**3. Canais (Channels)**
Como a empresa alcança, vende e entrega para os clientes? Inclua canais de comunicação, venda, distribuição e pós-venda (ex.: lojas próprias, e-commerce, marketplaces, força de vendas B2B, app).
 
**4. Relacionamento com Clientes (Customer Relationships)**
Que tipo de relação mantém com cada segmento? Autoatendimento, assistência pessoal, comunidade, cocriação, programas de fidelidade, SLA dedicado.
 
**5. Fontes de Receita (Revenue Streams)**
Como e quanto cobra? Identifique o mecanismo (venda de ativo, assinatura, licenciamento, publicidade, comissão, taxa de uso) e, se possível, a participação relativa de cada linha na receita total.
 
**6. Recursos-Chave (Key Resources)**
Ativos essenciais para entregar a proposta de valor: físicos (fábricas, frota, lojas), intelectuais (marca, patentes, dados, algoritmos), humanos (talento especializado) e financeiros.
 
**7. Atividades-Chave (Key Activities)**
O que a empresa precisa fazer bem? Produção, desenvolvimento de software/plataforma, logística, gestão de rede, marketing, P&D.
 
**8. Parcerias-Chave (Key Partnerships)**
Parceiros e fornecedores estratégicos sem os quais o modelo não funciona: alianças, joint ventures, fornecedores críticos, distribuidores.
 
**9. Estrutura de Custos (Cost Structure)**
Principais categorias de custo (CMV, logística, marketing, P&D, pessoal, infraestrutura). Indique se o modelo é dirigido por custo (cost-driven) ou por valor (value-driven), e a proporção fixo vs. variável quando relevante.
 
### 3. Formato de saída
 
Entregue o Canvas como uma **tabela 3x3 em markdown** (ou HTML quando o contexto pedir visual), seguindo o layout clássico:
 
| Parcerias-Chave | Atividades-Chave | Proposta de Valor | Relacionamento | Segmentos de Clientes |
|---|---|---|---|---|
|  | **Recursos-Chave** |  | **Canais** |  |
 
E abaixo, em linha cheia:
 
| Estrutura de Custos | Fontes de Receita |
 
Alternativa compacta: 9 seções em markdown com títulos `###`, cada uma com bullets curtos. Use esta opção quando a tabela ficar ilegível (mobile, muitos bullets).
 
### 4. Finalização
 
Após o Canvas, inclua em 2–4 linhas:
- **Lógica central do modelo**: uma frase sobre como os blocos se conectam (ex.: "marketplace dois-lados cuja escala no lado da oferta sustenta a proposta de valor no lado da demanda").
- **Pontos de atenção**: dependências críticas, riscos ou tensões visíveis no modelo (ex.: concentração em poucos fornecedores, receita dependente de publicidade, CAPEX intensivo).
## O que evitar
 
- Bullets genéricos que serviriam para qualquer empresa do setor.
- Copiar texto institucional da empresa sem síntese.
- Confundir Canvas com análise SWOT, Porter ou valuation — o Canvas descreve **como o negócio funciona hoje**, não avalia sua competitividade ou valor.
- Inventar segmentos, parcerias ou receitas sem base. Se não souber, diga "não divulgado publicamente" em vez de adivinhar.