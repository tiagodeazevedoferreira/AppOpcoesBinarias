# Radical 99 Edge — Estado Atual

## Objetivo
Construir uma estratégia seletiva para opções binárias que priorize **precisão extrema out-of-sample**, com meta de 99%+, sem transformar a métrica em um artefato de abstention.

## Estado confirmado
- Dataset observado no último experimento: `frxEURUSD`, ~420 mil ticks.
- Horizonte de referência: 60s.
- O modelo Radical Selective Edge atual produziu **0 decisões** para 99%, 99,5% e 99,9%.
- Isso significa que o filtro estatístico está conservador demais para executar, não que exista um edge de 99%.
- A persistência de ~90,96% em 60s e ~96,46% em 300s deve **não ser usada como baseline executável**: a implementação legada usa o `label` da linha anterior, que já incorpora informação futura em relação ao instante de decisão.
- O baseline observável correto é baseado somente no histórico de preços disponível no instante da decisão.

## Mudança de direção atual
A próxima etapa não é baixar arbitrariamente o threshold de 99%. É descobrir **bolsões observáveis de previsibilidade**.

Pipeline atual da pesquisa:

`dados → baseline observável → diagnóstico de regimes → bolsões de edge → seletor estatístico → walk-forward → promoção`

O novo diagnóstico `persistence_edge.py` procura estados definidos apenas por:
- direção do retorno observado em múltiplos lookbacks;
- consistência direcional;
- eficiência do movimento;
- volatilidade recente;
- hora do dia.

Cada célula é avaliada com:
- quantidade de evidências;
- acertos;
- acurácia observada;
- limite inferior Wilson unilateral de 99%.

## Próximo ponto de decisão
A execução automática deve responder:
1. Existem células observáveis com acurácia alta?
2. Quantas têm evidência suficiente?
3. O limite Wilson continua alto fora da amostra?
4. O efeito aparece em 60s, 120s ou 300s?
5. O efeito é estável nos folds walk-forward?
6. Existe cobertura não-trivial?

## Regra de integridade
Nunca promover uma estratégia com base em holdout isolado. A promoção exige estabilidade temporal e ausência de informação futura.

## Última ação de engenharia
Foi criado o workflow `Research Persistence Edge Diagnostic`, que executa testes e o diagnóstico automaticamente em push na branch `research/radical-99`.

## Próxima ação manual esperada
Somente quando o workflow terminar, analisar o artefato `persistence-edge-diagnostic` e escolher o próximo experimento com base nos bolsões observados. Não alterar thresholds às cegas.
