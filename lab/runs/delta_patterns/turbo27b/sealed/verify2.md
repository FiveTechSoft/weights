# Verificación por segundo método del blind test turbo27b (2026-11-11)

`turbo27b_verify2.py` (CPU, solo ficheros sellados/stats) recalcula los tres números del desenlace
escritos en `plan_turbo27b.md` ("Result after unsealing") con caminos independientes:

| Métrica | Claim del unsealing | Segundo método | Concordancia |
|---|---|---|---|
| P1 Jaccard (capas cambiadas de D_heretic vs predicción 26-63) | 0.79 | **0.7895** (30 capas, 26-55, ninguna fuera del rango documentado) | ✓ |
| P2 energy_top1 mediana (60 tensores residual-writer) | 0.34 | **0.338** | ✓ |
| P2 k90 mediana sobre 60 ("direcciones para 90% de energía") | 13 | **13.0** (24/60 con k90>16 — exacto) | ✓ |
| P3 Spearman del perfil por tipo de D_rest vs mediana 9B | 0.927 | **0.9286** (7 tipos; 4 mejoras 9B: mimo/neohorse/tmax/fara) | ✓ |

Notas de método (diferencias que hubo que resolver):
1. k90 exige el espectro COMPLETO como denominador; el stats solo guarda top-16 → total reconstruido como
   s1²/energy_top1. La mediana de 60 se toma con los 24 k90>16 ordenados por encima (la mediana cae en el
   30º/31º valor conocido = 13). La mediana de los 36 conocidos solos es 8 (no es la cifra del claim).
2. P3 es por TIPO de tensor (7 cubetas: mlp_down/mlp_gate/attn_qkv/attn_o/ssm_zab/ssm_state/norm), no por
   familia (a nivel familia el Spearman solo da 0.5: agrupar ssm entero destruye el orden).
3. Espectro top-16 de stats + energy_top1/energy_topk replicados: coherentes entre sí (spot-check L35 o_proj).

**Conclusión: los tres veredictos del blind test (P1 pass-with-noted-end, P2 fail, P3 pass) quedan
independientemente confirmados.** Fichero de salida: `sealed/verify2.json`.
