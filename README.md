# Hackathon Alliage - Desafio 04: registro rigido de capturas 3D

## Dados e unidade

`malha 1.ply` e a referencia; `malha 2.ply` e a movel. Ambas sao nuvens de pontos PLY binarias com RGB e normais, sem faces: 79.037 pontos na referencia e 84.887 na movel.

Os arquivos PLY nao declaram unidade fisica. Os resultados sao reportados na **unidade nativa dos arquivos**. A conversao ou interpretacao em milimetros requer confirmacao da origem dos dados.

Sem faces, a analise e point-wise: nao ha distancia ponto-superficie nem mapa triangular.

## Convencao

Vetores coluna homogeneos e matriz 4x4:

```text
p_reference = T @ p_moving
```

`resultados/transformacao_final.json` salva a matriz, `R`, `t` em unidade nativa, a convencao e verificacoes de rigidez (`det(R)` e erro de ortonormalidade).

## Registro

```text
PCA com 24 hipoteses de orientacao propria
-> mascara estavel heuristica por residuos NN apos PCA
-> ICP point-to-plane em tres escalas
-> trimming de 20% dos residuos plano-a-ponto e kernel Cauchy
```

A mascara estavel nao e anatomica ou clinica. Ela seleciona os 65% menores residuos `M->R` apos o alinhamento grosseiro de PCA, antes do ICP, e esses pontos moveis sao os usados pelo ICP.

## Metricas finais

Para cada ponto fonte `p_i`, a distancia e Euclidiana ao vertice mais proximo da nuvem alvo:

```text
d_i = min_j ||p_i - q_j||_2
```

As estatisticas direcionais usam todos os pontos fonte, sem filtrar outliers:

```text
RMS = sqrt(mean(d_i^2))
media = mean(d_i)
mediana = percentile(d_i, 50)
P95 = percentile(d_i, 95)
P99 = percentile(d_i, 99)
maximo NN direcionado = max(d_i)
desvio padrao = std(d_i)
```

O limiar e usado somente para cobertura:

```text
coverage_within_threshold = 100 * count(d_i <= threshold) / N
```

O RMS bidirecional combina somas de quadrados e contagens, nunca uma media simples de RMS:

```text
RMS_bi = sqrt((sum(d_MR^2) + sum(d_RM^2)) / (N_M + N_R))
```

## Resultados da ultima execucao

| Medida final | Valor na unidade nativa |
| --- | ---: |
| RMS M->R, 84.887 pontos | 0,348088 |
| RMS R->M, 79.037 pontos | 0,215443 |
| RMS bidirecional, 163.924 pontos | 0,291761 |
| Mediana M->R | 0,001228 |
| P95 M->R | 1,014004 |
| P99 M->R | 1,606912 |
| Maximo NN direcionado M->R | 2,041977 |
| Cobertura M->R <= 0,385317 | 91,04% |

A regiao candidata a alteracao e uma heuristica diferente da mascara estavel: sao os 8.489 vertices moveis cujo residuo final `M->R` e estritamente maior que P90 (`0,281503`). Ela nao e espacialmente contigua por construcao, nem uma segmentacao anatomica.

O mapa em `resultados/mapa_distancia.png` usa exatamente o mesmo vetor final de residuos `M->R` usado pelas metricas, com colorbar em unidade nativa. `mapa_distancia_vertices.ply` guarda essa mesma cor por vertice.

## Rastreabilidade da execucao

Uma execucao remove e recria os artefatos produzidos: `antes.png`, `depois.png`, `mapa_distancia.png`, `malha2_alinhada.ply`, `mapa_distancia_vertices.ply`, `metrics.json`, `relatorio.txt` e `transformacao_final.json`.

O terminal, `relatorio.txt`, `metrics.json`, a transformacao e as imagens sao alimentados por uma unica execucao e pela mesma transformacao final. `metrics.json` e a fonte detalhada e inclui as somas de quadrados que permitem recomputar o RMS bidirecional.

## Variabilidade e limitacoes

A variabilidade e estimada por seis bootstraps deterministas de 75% dos pontos. Na ultima rodada, RMS medio `0,348103` e desvio `0,000012`, ambos em unidade nativa. Isso descreve estabilidade numerica do pipeline, nao incerteza clinica.

O metodo pode degradar com sobreposicao pequena, grande mudanca geometrica ou mascara de consenso que inclua uma area alterada. Como os dados nao possuem faces nem unidade fisica declarada, nao e apropriado interpretar os valores como distancia de superficie ou milimetros sem informacao externa.

## Execucao

```powershell
python -m pip install -r requirements.txt
python run.py
```
