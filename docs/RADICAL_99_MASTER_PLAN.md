# Radical 99 Edge — Master Plan

## North Star
Chegar a uma decisão executável com precisão >=99% out-of-sample **sem depender de vazamento, seleção retrospectiva ou cobertura artificialmente zero**.

## Fases

### 1. Integridade estatística
- [x] Identificar o baseline de persistência baseado em `label` como não-executável.
- [x] Manter baseline observável baseado apenas em preço.
- [x] Preservar separação temporal.
- [x] Usar Wilson unilateral de 99% para avaliar robustez.

### 2. Descoberta de bolsões de previsibilidade
- [x] Criar diagnóstico de persistência observável.
- [x] Avaliar múltiplos horizontes.
- [x] Avaliar múltiplos lookbacks.
- [x] Condicionar por estado observável de mercado.
- [ ] Executar diagnóstico em dados reais.
- [ ] Selecionar estados candidatos com evidência suficiente.

### 3. Especialistas de estado
- [ ] Construir especialistas por regime/estado.
- [ ] Separar tendência, reversão, compressão e expansão.
- [ ] Avaliar consenso entre especialistas.
- [ ] Evitar células excessivamente granulares.

### 4. Meta-seletor
- [ ] Aprender quando cada especialista deve ser ativado.
- [ ] Permitir abstention, mas medir cobertura explicitamente.
- [ ] Otimizar precisão primeiro e cobertura depois.
- [ ] Calibrar confiança fora da amostra.

### 5. Validação radical
- [ ] Walk-forward em todos os folds.
- [ ] Teste não sobreposto.
- [ ] Teste por período/hora/regime.
- [ ] Wilson lower bound >= meta quando aplicável.
- [ ] Verificar degradação temporal.

### 6. Promoção
Uma estratégia somente pode ser promovida quando:
- não usa informação futura;
- supera o baseline observável ou demonstra edge independente;
- mantém a meta em todos os folds exigidos;
- possui número mínimo de decisões;
- possui cobertura mensurável e não-zero;
- passa os testes unitários e lint;
- o resultado é reproduzível.

## Princípio de projeto
A meta de 99% é uma **restrição de projeto**, não uma instrução para manipular o filtro até que a métrica apareça.

Se 99% só ocorrer com zero decisões, o resultado é rejeitado.

Se nenhum bolsão chegar a 99%, devemos procurar uma transformação estrutural do problema: horizonte, estado, evento, instrumento, especialização ou composição de sinais.

## Próximo experimento
O workflow `Research Persistence Edge Diagnostic` deve produzir o mapa inicial dos estados observáveis. A decisão seguinte será tomada a partir desse mapa, e não por ajuste manual de thresholds.
