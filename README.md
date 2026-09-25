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

O RMS global apos o registro nao deve ser interpretado automaticamente como "erro do registro". Ele e uma distancia NN entre a nuvem movel transformada e a referencia e pode combinar residuo de registro, diferenca geometrica entre capturas, ruido, diferencas de densidade e regioes ausentes. A metrica da `stable_region` e o indicador do residuo na regiao heuristica usada pelo ICP. A metrica da `candidate_discrepancy_region` descreve a discrepancia geometrica observada na regiao dos maiores residuos; nao confirma alteracao anatomica ou clinica.

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
| Cobertura de correspondencia M->R dentro de 8x o espacamento estimado | 91,04% |

A regiao estavel heuristica e medida novamente apos o registro usando exatamente os residuos finais `M->R` dos pontos selecionados antes do ICP. O JSON e o terminal salvam numero de pontos, RMS, media, mediana, P95, P99, maximo e desvio padrao. Isso e um indicador do residuo na regiao usada para estimar o registro, nao uma prova de estabilidade anatomica.

A regiao candidata a alteracao e uma heuristica diferente da mascara estavel: sao os pontos moveis com os 10% maiores residuos finais `M->R` (estritamente acima de P90). Ela nao e espacialmente contigua por construcao, nao e uma regiao anatomica comprovada e nao e uma segmentacao clinica. O JSON e o terminal salvam threshold P90, numero de pontos, percentual, media, mediana, RMS, P95, P99, maximo e desvio padrao para descrever a discrepancia geometrica observada.

O limiar de cobertura e um parametro explicito do experimento: `8 * spacing_estimate_native`, em que `spacing_estimate_native` e a mediana da distancia ao segundo vizinho mais proximo em uma amostra da referencia. Ele e chamado de **cobertura de correspondencia dentro de 8x o espacamento estimado** e serve somente para `count(d <= threshold) / N`; nao remove pontos de RMS, mediana, percentis ou maximo e nao e tolerancia clinica, erro clinico ou limite de precisao. A unidade fisica do PLY nao e declarada.

O mapa em `resultados/mapa_distancia.png` e um mapa point-wise de distancia NN: usa exatamente o mesmo vetor final de residuos `M->R` usado pelas metricas, com colorbar em unidade nativa. Nao e mapa de superficie continua, triangular ou ponto-superficie. O PLY `mapa_distancia_vertices.ply` guarda essa mesma cor por vertice e `regiao_candidata.png` destaca a heuristica P90.

## Rastreabilidade da execucao

Uma execucao remove e recria os artefatos produzidos: `antes.png`, `depois.png`, `mapa_distancia.png`, `malha2_alinhada.ply`, `mapa_distancia_vertices.ply`, `metrics.json`, `relatorio.txt` e `transformacao_final.json`.

O terminal, `relatorio.txt`, `metrics.json`, a transformacao e as imagens sao alimentados por uma unica execucao e pela mesma transformacao final. `metrics.json` e a fonte detalhada e inclui as somas de quadrados que permitem recomputar o RMS bidirecional.

## Variabilidade e limitacoes

A variabilidade e estimada por seis repeticoes deterministas, usando subconjuntos independentes de 75% dos pontos de cada nuvem. O JSON salva o RMS de cada repeticao, media, desvio padrao, fração usada e ruido aplicado (zero neste teste). Isso descreve variabilidade do RMS sob perturbacoes, nao intervalo de confiança, incerteza metrológica ou incerteza clínica.

A robustez é auditada separando: (A) ruido gaussiano com sigma de 0,5 spacing; (B) ausencia espacial dos 25% e dos 50% superiores em Z; e (C) uma perturbação de inicialização de 8 graus e translação. Os testes usam cópias/subconjuntos e não alteram os PLYs. Um caso é marcado como potencialmente não confiável se a cobertura cair pelo menos 20 pontos percentuais em relação ao resultado original ou se o RMS atingir pelo menos duas vezes o RMS original. O critério e todas as métricas ficam em `metrics.json`.

O método pode degradar com sobreposição pequena, grande mudança geométrica ou máscara de consenso que inclua uma área alterada. Os testes de ruído, ausência espacial e perturbação de inicialização são testes sintéticos de robustez, não validação clínica. Como os dados não possuem faces nem unidade física declarada, não é apropriado interpretar os valores como distância de superfície ou milímetros sem informação externa. A comparação entre regiões é uma diferença geométrica observada/residual geométrico, não uma alteração clínica.

Os tempos em `metrics.json` são separados em inspeção, preprocessamento, registro, métricas, análise de robustez/variabilidade, visualização e total. Isso permite distinguir o custo do registro dos testes e artefatos adicionais.

## Execucao

Para abrir a interface gráfica e escolher os dois arquivos PLY:

```powershell
python interface.py
```

No Linux, também é possível iniciar pelo arquivo executável `./iniciar_interface.sh`.

A interface exibe um resumo e o mapa point-wise; os arquivos completos são salvos em `resultados/`. Ela aceita nuvens PLY compatíveis com o esquema XYZ/RGB/normais usado neste projeto. Imagens 2D comuns não são entradas para este algoritmo.

Também é possível informar os arquivos diretamente:

```powershell
python run.py --referencia "caminho/referencia.ply" --movel "caminho/movel.ply" --saida "resultados"
```

Sem argumentos, o comando abaixo continua usando `malha 1.ply` e `malha 2.ply`:

```powershell
python -m pip install -r requirements.txt
python run.py
```
