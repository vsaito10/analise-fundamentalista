---
name: porter-five-forces
description: >
  Realize análises completas e estruturadas das Cinco Forças de Porter para empresas de capital
  aberto negociadas em bolsa. Use esta skill sempre que o usuário mencionar: forças de Porter,
  análise competitiva, rivalidade setorial, poder de barganha, ameaça de entrantes, produtos
  substitutos, análise da indústria, posicionamento estratégico, análise setorial, vantagem
  competitiva, moat, ou qualquer variação dessas expressões aplicada a uma empresa ou setor.
  Acione também quando o usuário pedir para "analisar a competitividade", "entender o setor",
  "avaliar barreiras de entrada", ou solicitar um relatório estratégico de uma empresa listada.
  Combine com dados públicos (DRE, releases, apresentações de investidores) para fundamentar
  cada força com evidências quantitativas e qualitativas.
---
 
# Análise das Cinco Forças de Porter — Empresas de Capital Aberto
 
Framework para análise estratégica completa de empresas listadas em bolsa, combinando o modelo
clássico de Porter com dados públicos disponíveis (ITR/DFP, release de resultados, apresentações
para investidores, notícias setoriais).
 
> **Nota importante:** Michael Porter definiu **cinco** forças (não quatro). O framework
> completo inclui: Rivalidade entre Concorrentes, Poder dos Fornecedores, Poder dos Clientes,
> Ameaça de Novos Entrantes e Ameaça de Substitutos. Sempre aplique as cinco.
 
---
 
## Fluxo de Trabalho
 
### 1. Identificação da Empresa e Setor
 
Antes de iniciar a análise:
 
- Confirme o **ticker** e a **bolsa** (ex: PETR4 na B3, VALE3 na B3)
- Identifique o **setor GICS** ou **segmento de atuação** (ex: Petróleo & Gás Integrado)
- Defina o **mercado relevante** (Brasil? América Latina? Global?)
- Liste os **principais concorrentes diretos** (ao menos 3–5 empresas comparáveis)
Se o usuário não informar a empresa, pergunte antes de prosseguir.
 
---
 
### 2. Coleta de Dados Públicos
 
Para fundamentar a análise, busque e utilize:
 
| Fonte | O que buscar |
|---|---|
| RI da empresa (site) | Apresentações para investidores, guidance, relatório anual |
| CVM / SEC (ADRs) | DFP, ITR, Formulário de Referência (seção de riscos e concorrência) |
| Release de resultados | Margens, receita por segmento, capex, endividamento |
| Notícias setoriais | M&As, novos entrantes, regulação, tecnologia |
| Dados setoriais (IBGE, ANP, Anatel, etc.) | Concentração de mercado, tamanho do setor |
 
**Use web_search** para obter dados recentes quando disponível. Priorize fontes primárias.
 
---
 
### 3. Análise das Cinco Forças
 
Para **cada força**, siga esta estrutura:
 
```
### [Nome da Força]
 
**Intensidade:** Alta / Média / Baixa  
**Tendência:** Crescente / Estável / Decrescente
 
**Fatores que aumentam a força:**
- [evidência com dado ou fonte]
 
**Fatores que diminuem a força:**
- [evidência com dado ou fonte]
 
**Impacto na empresa analisada:**
[parágrafo com implicação estratégica e financeira]
```
 
---
 
#### Força 1 — Rivalidade entre Concorrentes Existentes
 
Avalie:
- **Concentração do setor**: HHI (Herfindahl-Hirschman Index) ou participação de mercado dos top-3
- **Taxa de crescimento do setor**: setores estagnados amplificam a rivalidade
- **Diferenciação de produto/serviço**: commodities = alta rivalidade; marcas fortes = menor
- **Barreiras de saída**: ativos especializados, custos trabalhistas, regulação
- **Comportamento de precificação**: guerras de preço, disciplina de CAPEX no setor
- **Benchmarks financeiros**: comparação de ROIC, margem EBITDA, market share entre concorrentes
**Dados-chave a citar:** market share (%), margem EBITDA vs peers, histórico de pricing.
 
---
 
#### Força 2 — Poder de Barganha dos Fornecedores
 
Avalie:
- **Concentração de fornecedores**: poucos fornecedores = maior poder
- **Custo de troca (switching cost)**: quão fácil é trocar de fornecedor?
- **Existência de substitutos para os insumos**: matérias-primas com alternativas vs exclusivas
- **Ameaça de integração vertical para frente** pelos fornecedores
- **Relevância da empresa para o fornecedor**: se a empresa representa grande % da receita do fornecedor, seu poder aumenta
**Dados-chave a citar:** % do CPV representado pelos principais insumos, contratos de longo prazo, dependência geográfica de insumos.
 
---
 
#### Força 3 — Poder de Barganha dos Clientes
 
Avalie:
- **Concentração de clientes**: poucos clientes grandes = maior poder (verificar no Formulário de Referência — clientes que representam >10% da receita são obrigados a ser declarados)
- **Custo de troca para o cliente**: serviços críticos com alto lock-in reduzem o poder do cliente
- **Sensibilidade a preço**: setores B2B com margens apertadas tendem a pressionar mais
- **Disponibilidade de informação**: clientes mais informados negociam melhor
- **Ameaça de integração vertical para trás** pelos clientes
**Dados-chave a citar:** % da receita do maior cliente, churn rate (quando divulgado), NPS/satisfação, contratos recorrentes.
 
---
 
#### Força 4 — Ameaça de Novos Entrantes
 
Avalie:
- **Economias de escala**: quanto maior a escala necessária, maior a barreira
- **Requisitos de capital**: CAPEX intensivo é barreira natural (ex: siderurgia, telecomunicações)
- **Acesso a canais de distribuição**: redes consolidadas são difíceis de replicar
- **Vantagens de custo independentes de escala**: patentes, localização, acesso a matérias-primas
- **Regulação e licenças**: setores regulados (bancos, seguros, energia) têm barreiras formais
- **Efeitos de rede**: quanto mais usuários, maior o valor (ex: plataformas digitais)
- **Reação esperada dos incumbentes**: histórico de agressividade defensiva
**Dados-chave a citar:** CAPEX/Receita do setor, número de players novos nos últimos 5 anos, exigências regulatórias, patentes.
 
---
 
#### Força 5 — Ameaça de Produtos ou Serviços Substitutos
 
Avalie:
- **Relação custo-benefício dos substitutos**: o substituto entrega valor similar a custo menor?
- **Propensão do comprador a substituir**: hábitos, custos de transição, status
- **Desempenho relativo do substituto**: tecnologias disruptivas vs produtos maduros
- **Tendências de longo prazo**: adoção de energias renováveis, digitalização, mudanças regulatórias
**Dados-chave a citar:** crescimento do mercado de substitutos, exemplos de migração de clientes, elasticidade-preço estimada.
 
---
 
### 4. Síntese Estratégica
 
Após analisar as cinco forças, produza:
 
#### 4.1 Mapa de Intensidade
 
```
Força                          | Intensidade | Tendência
-------------------------------|-------------|----------
Rivalidade entre Concorrentes  |    Alta     | Estável
Poder dos Fornecedores         |    Média    | Decrescente
Poder dos Clientes             |    Baixa    | Estável
Ameaça de Novos Entrantes      |    Baixa    | Crescente
Ameaça de Substitutos         |    Média    | Crescente
```
 
#### 4.2 Conclusão Estratégica
 
Responda objetivamente:
1. **A indústria é estruturalmente atrativa?** (lucratividade esperada acima do custo de capital?)
2. **Qual é o posicionamento competitivo da empresa** frente às forças?
3. **Quais são os principais riscos estratégicos** identificados?
4. **Quais são os diferenciais competitivos (moat)** que protegem a empresa?
#### 4.3 Implicações para o Investidor
 
- Como as forças identificadas se refletem nos múltiplos de valuation? (ex: setor com baixa rivalidade tende a ser negociado com prêmio de P/E)
- Existem catalisadores ou riscos não precificados identificados pela análise?
- A posição competitiva da empresa justifica ou questiona o valuation atual?
---
 
### 5. Formato de Entrega
 
Por padrão, entregue a análise como:
- **Relatório em markdown** estruturado, com tabelas e seções claras
- Se o usuário solicitar arquivo, use a skill `docx` para gerar um Word profissional
- Se solicitado, gere também um **arquivo XLSX** com o mapa de forças e scoring (use a skill `xlsx`)
**Extensão esperada:** análise completa = 800–1.500 palavras. Análise resumida (quando solicitado) = 300–500 palavras.
 
---
 
## Boas Práticas
 
- **Evite generalidades**: toda afirmação deve ter respaldo em dado, fonte ou lógica setorial explícita
- **Diferencie a indústria da empresa**: as forças descrevem o setor; o posicionamento da empresa dentro dele é consequência
- **Seja prospectivo**: analise tendências, não apenas o estado atual
- **Cite dados financeiros** (margens, ROIC, market share) sempre que disponíveis
- **Não confunda força com fraqueza interna**: as Cinco Forças são externas à empresa (para análise interna, use SWOT ou Value Chain)
---
 
## Referência Rápida — Sinais de Alta vs Baixa Atratividade
 
| Força | Sinal de Alta Atratividade | Sinal de Baixa Atratividade |
|---|---|---|
| Rivalidade | Poucos players, setor crescendo, diferenciação | Muitos players, setor maduro, guerra de preços |
| Fornecedores | Muitos fornecedores, insumos commodity | Fornecedor único, insumo crítico sem substituto |
| Clientes | Muitos clientes fragmentados, alto switching cost | Poucos clientes grandes, produto commodity |
| Novos Entrantes | Alto CAPEX, regulação forte, economias de escala | Baixo CAPEX, setor desregulado, fácil replicação |
| Substitutos | Sem alternativas viáveis, alto custo de troca | Substitutos mais baratos e acessíveis crescendo |
 
---
 
## Integração com Outras Skills
 
- Use **`dcf-valuation`** após esta análise para conectar o posicionamento estratégico às premissas de crescimento e margem no modelo de valuation
- Use **`docx`** para entregar o relatório em formato Word profissional
- Use **`pdf`** para entregar o relatório em formato PDF profissional
- Use **`xlsx`** para criar um scorecard interativo das cinco forças