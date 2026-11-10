// main.js — Orquestador del simulador DeepSeek (máquina de estados)
(function () {

    // ── Estado global ─────────────────────────────────────────────────────────

    let model    = null;
    let inference = null;   // resultado completo de runInference
    let steps    = [];      // secuencia de pasos a visualizar
    let stepIdx  = 0;       // próximo paso a ejecutar
    let phase    = 'init';  // init | ready | running | done

    let autoTimer   = null;
    let autoPlaying = false;

    // ── Referencias DOM ───────────────────────────────────────────────────────

    const btnStep     = document.getElementById('btn-step');
    const btnLayer    = document.getElementById('btn-layer');
    const btnRun      = document.getElementById('btn-run');
    const btnReset    = document.getElementById('btn-reset');
    const btnAutoPlay = document.getElementById('btn-auto-play');
    const selSpeed    = document.getElementById('auto-speed');
    const btnOverview = document.getElementById('btn-overview');
    const ovOverlay   = document.getElementById('overview-overlay');
    const ovClose     = document.getElementById('overview-close');
    const stepPanel   = document.getElementById('step-panel');

    // ── Hiperparámetros ───────────────────────────────────────────────────────

    function getHP() {
        return {
            phrase:  document.getElementById('sel-phrase').value,
            temp:    parseFloat(document.getElementById('temperature').value),
            topK:    parseInt(document.getElementById('top-k').value, 10),
            nLayers: parseInt(document.getElementById('n-layers').value, 10),
            speed:   parseInt(selSpeed.value, 10)
        };
    }

    // Actualizar labels de sliders en tiempo real
    document.getElementById('temperature').addEventListener('input', function () {
        document.getElementById('temperature-value').textContent = parseFloat(this.value).toFixed(1);
    });

    // ── Inicializar ───────────────────────────────────────────────────────────

    resetSimulation();

    btnStep.addEventListener('click',     () => { stopAutoPlay(); executeStep(); });
    btnLayer.addEventListener('click',    () => { stopAutoPlay(); executeOneLayer(); });
    btnRun.addEventListener('click',      () => { stopAutoPlay(); executeAll(); });
    btnReset.addEventListener('click',    () => { stopAutoPlay(); resetSimulation(); });
    btnAutoPlay.addEventListener('click', toggleAutoPlay);

    btnOverview.addEventListener('click', () => ovOverlay.classList.add('visible'));
    ovClose.addEventListener('click',     () => ovOverlay.classList.remove('visible'));
    ovOverlay.addEventListener('click',   e => { if (e.target === ovOverlay) ovOverlay.classList.remove('visible'); });
    document.addEventListener('keydown',  e => { if (e.key === 'Escape') ovOverlay.classList.remove('visible'); });

    selSpeed.addEventListener('change', () => {
        if (autoPlaying) { stopAutoPlay(); startAutoPlay(); }
    });

    // ── Auto-play ─────────────────────────────────────────────────────────────

    function toggleAutoPlay() {
        autoPlaying ? stopAutoPlay() : startAutoPlay();
    }

    function startAutoPlay() {
        if (autoPlaying || phase === 'done') return;
        autoPlaying = true;
        btnAutoPlay.textContent = '⏸ Pausar';
        btnAutoPlay.classList.add('btn-playing');
        const ms = getHP().speed;
        autoTimer = setInterval(() => {
            if (phase === 'done') { stopAutoPlay(); return; }
            executeStep();
        }, ms);
    }

    function stopAutoPlay() {
        if (autoTimer) { clearInterval(autoTimer); autoTimer = null; }
        autoPlaying = false;
        btnAutoPlay.textContent = '▶ Auto-ejecutar';
        btnAutoPlay.classList.remove('btn-playing');
    }

    // ── Reset ─────────────────────────────────────────────────────────────────

    function resetSimulation() {
        phase    = 'init';
        stepIdx  = 0;
        inference = null;

        const hp = getHP();

        model = DeepSeekModel.build(hp.topK, hp.nLayers);

        // Calcular secuencia de pasos
        steps = buildStepSequence(hp.nLayers, hp.topK);

        DeepSeekArch.render(hp.nLayers);
        DeepSeekArch.clearHighlight();
        DeepSeekUI.renderWelcome(stepPanel);
        DeepSeekUI.updateStepDescription('Seleccioná una frase y presioná "Siguiente paso" para comenzar.');

        phase = 'ready';
    }

    // ── Secuencia de pasos ────────────────────────────────────────────────────

    function buildStepSequence(nLayers, topK) {
        const seq = [
            { phase: 'tokenize',     layer: null, archId: 'tokens',
              desc: 'Paso 1 — Tokenización: la frase se divide en tokens del vocabulario' },
            { phase: 'embed',        layer: null, archId: 'embed',
              desc: 'Paso 2 — Embedding: cada token se mapea a un vector de 8 dimensiones' }
        ];

        for (let l = 0; l < nLayers; l++) {
            const n = l + 1;
            seq.push(
                { phase: 'mla_compress',  layer: l, archId: `mla_${l}`,
                  desc: `Capa ${n} — MLA: compresión al espacio latente (d_model=${DeepSeekModel.D_MODEL} → d_c=${DeepSeekModel.D_C})` },
                { phase: 'mla_attention', layer: l, archId: `mla_${l}`,
                  desc: `Capa ${n} — MLA: atención multi-cabezal con K,V reconstruidos desde el espacio latente` },
                { phase: 'moe_router',    layer: l, archId: `moe_${l}`,
                  desc: `Capa ${n} — MoE: el router asigna scores (sigmoid) a los ${DeepSeekModel.N_EXPERTS} expertos y selecciona Top-${topK}` },
                { phase: 'moe_experts',   layer: l, archId: `moe_${l}`,
                  desc: `Capa ${n} — MoE: los expertos seleccionados + el experto compartido procesan la representación` },
                { phase: 'moe_combine',   layer: l, archId: `moe_${l}`,
                  desc: `Capa ${n} — MoE: combinación ponderada de salidas → residual` }
            );
        }

        seq.push(
            { phase: 'output',     layer: null, archId: 'output',
              desc: 'Salida — proyección al vocabulario (20 palabras) y predicción del siguiente token' },
            { phase: 'comparison', layer: null, archId: 'output',
              desc: 'Comparativa — DeepSeek (MLA + MoE) vs Transformer Tradicional: arquitectura, métricas y eficiencia' }
        );

        return seq;
    }

    // ── executeStep ───────────────────────────────────────────────────────────

    function executeStep() {
        if (phase === 'init' || phase === 'done') return;

        // Primera vez: correr inferencia
        if (!inference) {
            const hp = getHP();
            inference = DeepSeekModel.runInference(model, hp.phrase, hp.temp);
        }

        if (stepIdx >= steps.length) {
            phase = 'done';
            return;
        }

        const step = steps[stepIdx];
        renderStep(step);
        DeepSeekArch.highlight(step.archId);
        DeepSeekUI.updateStepDescription(
            `[${stepIdx + 1}/${steps.length}] ${step.desc}`
        );

        stepIdx++;
        if (stepIdx >= steps.length) phase = 'done';
    }

    function renderStep(step) {
        const { phase: p, layer: l } = step;
        const r = inference;

        switch (p) {
            case 'tokenize':
                DeepSeekUI.renderTokens(r.tokens, stepPanel);
                break;
            case 'embed':
                DeepSeekUI.renderEmbedding(r.embMatrix, r.tokens, stepPanel);
                break;
            case 'mla_compress':
                DeepSeekUI.renderMLACompress(r.layers[l].latent, r.tokens, l, stepPanel);
                break;
            case 'mla_attention':
                DeepSeekUI.renderMLAAttention(r.layers[l].attnScores, r.tokens, l, stepPanel);
                break;
            case 'moe_router':
                DeepSeekUI.renderMoERouter(r.layers[l].moeResults, r.tokens, model.topK, l, stepPanel);
                break;
            case 'moe_experts':
                DeepSeekUI.renderMoEExperts(r.layers[l].moeResults, r.tokens, model.topK, l, stepPanel);
                break;
            case 'moe_combine':
                DeepSeekUI.renderMoECombine(r.layers[l].moeResults, r.tokens, model.topK, l, stepPanel);
                break;
            case 'output':
                DeepSeekUI.renderOutput(r.top5, r.probs, r.tokens, stepPanel);
                break;
            case 'comparison':
                DeepSeekUI.renderComparison(r.tokens, model.topK, stepPanel);
                break;
        }
    }

    // ── executeOneLayer ───────────────────────────────────────────────────────

    function executeOneLayer() {
        if (phase === 'init' || phase === 'done') return;
        if (!inference) {
            const hp = getHP();
            inference = DeepSeekModel.runInference(model, hp.phrase, hp.temp);
        }
        if (stepIdx >= steps.length) return;

        // Si estamos antes de las capas transformer, avanzar hasta fin de tokenize+embed
        if (stepIdx < 2) {
            while (stepIdx < 2 && stepIdx < steps.length) executeStep();
            return;
        }

        // Si estamos en la última etapa (output), ejecutar
        if (steps[stepIdx].phase === 'output') {
            executeStep();
            return;
        }

        // Avanzar hasta el fin de la capa actual (5 sub-pasos por capa)
        const stepsPerLayer = 5;
        const baseStep      = 2;
        const stepsInCurrent = (stepIdx - baseStep) % stepsPerLayer;
        const remaining     = stepsPerLayer - stepsInCurrent;
        const target        = Math.min(stepIdx + remaining, steps.length);

        while (stepIdx < target && phase !== 'done') executeStep();
    }

    // ── executeAll ────────────────────────────────────────────────────────────

    function executeAll() {
        if (phase === 'init') return;
        if (!inference) {
            const hp = getHP();
            inference = DeepSeekModel.runInference(model, hp.phrase, hp.temp);
        }
        while (stepIdx < steps.length) executeStep();
    }

})();
