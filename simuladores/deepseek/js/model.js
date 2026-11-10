// model.js — Modelo simplificado pedagógico de DeepSeek (MLA + MoE)
const DeepSeekModel = (function () {

    const VOCAB = ["el","modelo","procesa","cada","token","mediante","expertos","atención",
                   "capa","red","neuronal","datos","pesos","inferencia","latente","vector",
                   "selecciona","combina","salida","predicción"];

    const PHRASES = [
        "la red neuronal procesa",
        "el modelo selecciona",
        "inferencia mediante atención"
    ];

    // Siguiente token esperado por frase (para fine-tuning de la cabeza de salida)
    const PHRASE_TARGETS = ["datos", "expertos", "latente"];

    const D_MODEL = 8, D_C = 4, N_HEADS = 4, HEAD_DIM = 2, N_EXPERTS = 8, FFN_HIDDEN = 16;

    // ── Utilidades matemáticas ─────────────────────────────────────────────────

    function randn(s) {
        const u1 = Math.random() + 1e-10, u2 = Math.random();
        return s * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    function mat(r, c, s) {
        return Array.from({length: r}, () => Array.from({length: c}, () => randn(s)));
    }

    function vecR(n, s) {
        return Array.from({length: n}, () => randn(s));
    }

    // v:[inDim], M:[inDim×outDim] → result:[outDim]   (equivale a v @ M)
    function vmul(v, M, outDim) {
        return Array.from({length: outDim}, (_, j) =>
            v.reduce((sum, vi, i) => sum + vi * M[i][j], 0)
        );
    }

    function relu(a) { return a.map(x => Math.max(0, x)); }

    function softmax(a) {
        const m = Math.max(...a);
        const ex = a.map(x => Math.exp(x - m));
        const s = ex.reduce((t, e) => t + e, 0);
        return ex.map(e => e / (s + 1e-10));
    }

    function sigmoid(x) { return 1 / (1 + Math.exp(-Math.max(-50, Math.min(50, x)))); }

    function dot(a, b) { return a.reduce((s, ai, i) => s + ai * b[i], 0); }

    function chunk(a, n) {
        const r = [];
        for (let i = 0; i < a.length; i += n) r.push(a.slice(i, i + n));
        return r;
    }

    // ── Tokenizador ───────────────────────────────────────────────────────────

    function tokenize(phrase) {
        return phrase.toLowerCase().split(/\s+/).map(w => {
            const id = VOCAB.indexOf(w);
            return { word: w, id: id >= 0 ? id : 0 };
        });
    }

    // ── MLA (Multi-head Latent Attention) ─────────────────────────────────────

    function buildMLA() {
        const W_down  = mat(D_MODEL, D_C,             0.2);  // comprimir x → c
        const W_Q     = mat(D_MODEL, N_HEADS*HEAD_DIM, 1.0); // std alto para scores no uniformes
        const W_up_K  = mat(D_C,    N_HEADS*HEAD_DIM, 1.0); // std alto para scores no uniformes
        const W_up_V  = mat(D_C,    N_HEADS*HEAD_DIM, 0.2); // c → V (reconstruir)
        const W_out   = mat(N_HEADS*HEAD_DIM, D_MODEL, 0.2); // concat → out

        return {
            compress(X) {
                // X:[seq×D_MODEL] → C:[seq×D_C]
                return X.map(x => vmul(x, W_down, D_C));
            },
            attend(X, C) {
                const seq = X.length, scale = Math.sqrt(HEAD_DIM);
                const Qf = X.map(x => vmul(x, W_Q,    N_HEADS * HEAD_DIM));
                const Kf = C.map(c => vmul(c, W_up_K, N_HEADS * HEAD_DIM));
                const Vf = C.map(c => vmul(c, W_up_V, N_HEADS * HEAD_DIM));

                const Q = Qf.map(v => chunk(v, HEAD_DIM));
                const K = Kf.map(v => chunk(v, HEAD_DIM));
                const V = Vf.map(v => chunk(v, HEAD_DIM));

                const headScores = [], headOuts = [];
                for (let h = 0; h < N_HEADS; h++) {
                    const sc = Array.from({length: seq}, (_, i) =>
                        Array.from({length: seq}, (_, j) => dot(Q[i][h], K[j][h]) / scale)
                    );
                    const at = sc.map(r => softmax(r));
                    headScores.push(at);
                    headOuts.push(Array.from({length: seq}, (_, i) =>
                        Array.from({length: HEAD_DIM}, (_, d) =>
                            at[i].reduce((s, a, j) => s + a * V[j][h][d], 0))
                    ));
                }

                // Concatenar heads + proyectar + residual
                const cat     = Array.from({length: seq}, (_, i) => headOuts.flatMap(h => h[i]));
                const proj    = cat.map(v => vmul(v, W_out, D_MODEL));
                const attnOut = proj.map((p, i) => p.map((v, d) => X[i][d] + v));

                // Promedio de scores entre heads para visualización
                const avgScores = Array.from({length: seq}, (_, i) =>
                    Array.from({length: seq}, (_, j) =>
                        headScores.reduce((s, h) => s + h[i][j], 0) / N_HEADS)
                );

                return { attnOut, attnScores: avgScores };
            }
        };
    }

    // ── Experto FFN ───────────────────────────────────────────────────────────

    function buildExpert(s) {
        return {
            W1: mat(D_MODEL, FFN_HIDDEN, s),
            b1: vecR(FFN_HIDDEN, 0.05),
            W2: mat(FFN_HIDDEN, D_MODEL, s),
            b2: vecR(D_MODEL, 0.05)
        };
    }

    function runExpert(e, x) {
        const h = relu(vmul(x, e.W1, FFN_HIDDEN).map((v, j) => v + e.b1[j]));
        return vmul(h, e.W2, D_MODEL).map((v, d) => v + e.b2[d]);
    }

    // ── MoE (Mixture of Experts) ──────────────────────────────────────────────

    function buildMoE() {
        const W_router = mat(D_MODEL, N_EXPERTS, 0.2);
        const bias     = vecR(N_EXPERTS, 0.1);
        const experts  = Array.from({length: N_EXPERTS}, () => buildExpert(0.12));
        const shared   = buildExpert(0.12);

        return {
            forward(X, topK) {
                return X.map(x => {
                    const logits = vmul(x, W_router, N_EXPERTS).map((l, e) => l + bias[e]);
                    const scores = logits.map(sigmoid);

                    const ranked   = scores.map((s, i) => ({s, i})).sort((a, b) => b.s - a.s);
                    const sel      = ranked.slice(0, topK);
                    const topKIdx  = sel.map(r => r.i);
                    const sumW     = sel.reduce((s, r) => s + r.s, 0);
                    const topKW    = sel.map(r => r.s / (sumW + 1e-8));

                    const expOuts  = topKIdx.map(ei => runExpert(experts[ei], x));
                    const sharedOut = runExpert(shared, x);

                    const combined = Array(D_MODEL).fill(0);
                    expOuts.forEach((o, k) => o.forEach((v, d) => { combined[d] += topKW[k] * v; }));
                    const sw = 1 / (topK + 1);
                    sharedOut.forEach((v, d) => { combined[d] += sw * v; });

                    const moeOut = x.map((xi, d) => xi + combined[d]);
                    return { scores, topKIdx, topKW, expOuts, sharedOut, combined, moeOut };
                });
            }
        };
    }

    // ── Output Head ───────────────────────────────────────────────────────────

    // Prior semántico fijo por tipo de palabra.
    // Sustantivos/adjetivos técnicos reciben mayor probabilidad base;
    // artículos, determinantes y preposiciones quedan suprimidos.
    // Orden: el, modelo, procesa, cada, token, mediante, expertos, atención,
    //        capa, red, neuronal, datos, pesos, inferencia, latente, vector,
    //        selecciona, combina, salida, predicción
    const SEMANTIC_BIAS = [
        -0.9,  // el         — artículo
         0.5,  // modelo     — sustantivo
         0.1,  // procesa    — verbo
        -0.7,  // cada       — determinante
         0.5,  // token      — sustantivo técnico
        -0.6,  // mediante   — preposición
         0.6,  // expertos   — sustantivo
         0.6,  // atención   — sustantivo técnico
         0.3,  // capa       — sustantivo
         0.4,  // red        — sustantivo
         0.5,  // neuronal   — adjetivo
         0.6,  // datos      — sustantivo
         0.4,  // pesos      — sustantivo técnico
         0.4,  // inferencia — sustantivo
         0.5,  // latente    — adjetivo técnico
         0.4,  // vector     — sustantivo técnico
         0.1,  // selecciona — verbo
         0.1,  // combina    — verbo
         0.3,  // salida     — sustantivo
         0.4   // predicción — sustantivo
    ];

    function buildOutputHead() {
        const W = mat(D_MODEL, VOCAB.length, 0.1);
        return {
            W,
            forward(x, temp) {
                const logits = vmul(x, W, VOCAB.length).map((l, i) => l + SEMANTIC_BIAS[i]);
                const scaled = logits.map(l => l / Math.max(0.1, temp));
                const probs  = softmax(scaled);
                const ranked = probs.map((p, i) => ({p, word: VOCAB[i], id: i}))
                                    .sort((a, b) => b.p - a.p);
                return { probs, top5: ranked.slice(0, 5) };
            }
        };
    }

    // ── Fine-tuning de la cabeza de salida ────────────────────────────────────
    // Entrena suavemente W_out sobre las 3 frases conocidas. Se detiene en cuanto
    // el target supera TARGET_PROB para evitar sobre-confianza (> 55 % aprox.).

    function _finetune(model) {
        const lr         = 0.05;
        const maxIter    = 120;
        const TARGET_PROB = 0.52;
        const W          = model.outputHead.W;

        function _forward(phrase) {
            const ids = tokenize(phrase).map(t => t.id);
            let X = ids.map(id => model.embW[id].slice());
            for (let l = 0; l < model.nLayers; l++) {
                const { mla, moe } = model.layers[l];
                const latent = mla.compress(X);
                const { attnOut } = mla.attend(X, latent);
                X = moe.forward(attnOut, model.topK).map(r => r.moeOut);
            }
            return X[X.length - 1];
        }

        // Pre-computar hidden states (no cambian durante el fine-tuning de W)
        const hiddenStates = PHRASES.map(p => _forward(p));
        const targets      = PHRASE_TARGETS.map(w => VOCAB.indexOf(w));

        for (let iter = 0; iter < maxIter; iter++) {
            let allDone = true;
            PHRASES.forEach((_, pi) => {
                const x      = hiddenStates[pi];
                const target = targets[pi];
                const logits = vmul(x, W, VOCAB.length).map((l, i) => l + SEMANTIC_BIAS[i]);
                const probs  = softmax(logits);

                if (probs[target] < TARGET_PROB) {
                    allDone = false;
                    const grad = probs.slice();
                    grad[target] -= 1;
                    for (let i = 0; i < D_MODEL; i++)
                        for (let j = 0; j < VOCAB.length; j++)
                            W[i][j] -= lr * x[i] * grad[j];
                }
            });
            if (allDone) break;
        }
    }

    // ── Constructor del modelo ────────────────────────────────────────────────

    function build(topK, nLayers) {
        const model = {
            topK:    topK    || 2,
            nLayers: nLayers || 2,
            embW:    mat(VOCAB.length, D_MODEL, 0.3),
            layers:  Array.from({length: nLayers || 2}, () => ({
                mla: buildMLA(),
                moe: buildMoE()
            })),
            outputHead: buildOutputHead()
        };
        _finetune(model);
        return model;
    }

    // ── Inferencia completa (precomputa todo para el paso a paso) ──────────────

    function runInference(model, phrase, temperature) {
        const tokens   = tokenize(phrase);
        const tokenIds = tokens.map(t => t.id);

        const embMatrix = tokenIds.map(id => model.embW[id].slice());
        let X = embMatrix.map(v => v.slice());

        const layers = [];
        for (let l = 0; l < model.nLayers; l++) {
            const { mla, moe } = model.layers[l];
            const latent = mla.compress(X);
            const { attnOut, attnScores } = mla.attend(X, latent);
            const moeResults = moe.forward(attnOut, model.topK);
            X = moeResults.map(r => r.moeOut);
            layers.push({ latent, attnScores, attnOut, moeResults });
        }

        const { probs, top5 } = model.outputHead.forward(X[X.length - 1], temperature || 1.0);
        return { tokens, embMatrix, layers, probs, top5 };
    }

    return {
        VOCAB, PHRASES, PHRASE_TARGETS, D_MODEL, D_C, N_HEADS, HEAD_DIM, N_EXPERTS, FFN_HIDDEN,
        tokenize, build, runInference
    };
})();
