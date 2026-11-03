// ============ head decompilado ============
// layer 23, q-head #8 de 16 (kv-group 2, kv-heads 4, GDA ratio 4)
// dims por head: Q 256d, GATE 256d sigmoid, K 256d, V 256d, writeback 256x4096
// pre-rope model (rope real: 64 dims longitudinisin secciones multimo)
const float SCALE = 0.062500000;

// --- por cada fila t del residual x[t]:
  q_t = RMS_norm( Wq @ x[t] )
  k_s = RMS_norm( Wk @ x[s] )        // compartido por group
  v_s = Wv @ x[s]
  g_t = sigmoid( Wgate @ x[t] )      // gate multiplicative del head

// --- scores y mezcla (causal)
  score[t][s] = SCALE * dot(q_t, k_s)
  p[t][*] = softmax(score)
  o_t = sum_s p[t][s] * v_s
  o_t *= g_t                          // QWEN35: gate sigmoid

// --- writeback
  residual[t] += Wo(row-block head #8) @ o_t

// --- diagnostico QK (pre-rope estatico): M0 = Wq^T @ Wk (normas escaladas)
//     c1(M0)=0.0506; nota: RoPE (64 dims de 256) altera el score por offset
//     ||Wq|| vs ||Wgate||: 15.25 vs 15.20 (ratio gate 0.996)