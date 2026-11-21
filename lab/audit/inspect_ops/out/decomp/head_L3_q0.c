// ============ head decompilado ============
// layer 3, q-head #0 de 16 (kv-group 0, kv-heads 4, GDA ratio 4)
// dims por head: Q 256d, GATE 256d sigmoid, K 256d, V 256d, writeback 256x1024
// pre-rope model (el rope real rota 64 de 256 dims; MRoPE multimodal, freq 1e7)
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
  residual[t] += Wo(row-block head #0) @ o_t

// --- diagnostico QK (pre-rope estatico): M0 = Wq^T @ Wk (normas escaladas)
//     c1(M0)=0.0945; nota: RoPE (64 dims de 256) altera el score por offset
//     ||Wq|| vs ||Wgate||: 10.30 vs 9.93 (ratio gate 0.964)