// ui.js — Funciones de render del simulador DeepSeek
const DeepSeekUI = (function () {

    // ── Colormaps ─────────────────────────────────────────────────────────────

    function divergingColor(v, absMax) {
        if (absMax < 1e-9) return '#ffffff';
        const t = Math.max(-1, Math.min(1, v / absMax));
        if (t < 0) {
            const a = -t;
            return `rgb(${Math.round(255 - 216*a)},${Math.round(255 - 155*a)},255)`;
        }
        return `rgb(255,${Math.round(255 - 228*t)},${Math.round(255 - 228*t)})`;
    }

    function sequentialColor(v, max) {
        if (max < 1e-9) return '#f0f4ff';
        const t = Math.max(0, Math.min(1, v / max));
        return `rgb(${Math.round(240 - 43*t)},${Math.round(244 - 217*t)},${Math.round(255 - 228*t)})`;
    }

    function absMax2D(arr) {
        let m = 1e-9;
        arr.forEach(row => row.forEach(v => { if (Math.abs(v) > m) m = Math.abs(v); }));
        return m;
    }

    function posMax1D(arr) {
        return Math.max(1e-9, ...arr.map(Math.abs));
    }

    // ── Colores de posición para tokens ───────────────────────────────────────

    const TOKEN_COLORS = ['#1565c0','#6a1b9a','#00695c','#bf360c','#4a148c','#1b5e20'];

    // ── renderTokens ──────────────────────────────────────────────────────────

    function renderTokens(tokens, container) {
        container.innerHTML = '';

        const section = _section('Tokenización', '#455a64', 'token-tooltip');
        const chips = _div('token-chips');

        tokens.forEach((t, i) => {
            const chip = _div('token-chip');
            chip.style.borderColor = TOKEN_COLORS[i % TOKEN_COLORS.length];
            chip.innerHTML = `<span class="chip-word">${t.word}</span><span class="chip-id">ID ${t.id}</span>`;
            chips.appendChild(chip);
        });

        section.appendChild(chips);

        const note = _div('step-note');
        note.textContent = `${tokens.length} tokens | Vocabulario: ${DeepSeekModel.VOCAB.length} palabras | Dimensión embedding: ${DeepSeekModel.D_MODEL}`;
        section.appendChild(note);

        const kvNote = _div('step-note step-note--info');
        kvNote.innerHTML = `En un modelo real (DeepSeek-V3), el vocabulario tiene <strong>128 000</strong> tokens y la dimensión de embedding es <strong>7168</strong>.`;
        section.appendChild(kvNote);

        container.appendChild(section);
    }

    // ── renderEmbedding ───────────────────────────────────────────────────────

    function renderEmbedding(embMatrix, tokens, container) {
        container.innerHTML = '';

        const section = _section('Vectores de Embedding', '#1565c0', 'embed-tooltip');
        const info = _div('step-note');
        info.textContent = `Cada token se mapea a un vector de ${DeepSeekModel.D_MODEL} dimensiones. Colormap: azul = negativo, rojo = positivo.`;
        section.appendChild(info);

        const wrap = _div('emb-wrap');

        // Etiquetas de dimensiones (columnas)
        const dimRow = _div('emb-dim-labels');
        dimRow.appendChild(_span('', 'emb-row-label')); // celda vacía
        for (let d = 0; d < DeepSeekModel.D_MODEL; d++) {
            const lbl = _span(`d${d}`, 'emb-dim-lbl');
            dimRow.appendChild(lbl);
        }
        wrap.appendChild(dimRow);

        const am = absMax2D(embMatrix);

        embMatrix.forEach((vec, i) => {
            const row = _div('emb-row');
            const lbl = _span(tokens[i].word, 'emb-row-label');
            lbl.style.color = TOKEN_COLORS[i % TOKEN_COLORS.length];
            row.appendChild(lbl);

            vec.forEach(v => {
                const cell = _div('emb-cell');
                cell.style.background = divergingColor(v, am);
                cell.title = v.toFixed(3);
                row.appendChild(cell);
            });

            wrap.appendChild(row);
        });

        section.appendChild(wrap);
        _appendLegendDiverging(section);
        container.appendChild(section);
    }

    // ── renderMLACompress ─────────────────────────────────────────────────────

    function renderMLACompress(latent, tokens, layerIdx, container) {
        container.innerHTML = '';

        const { N_HEADS, HEAD_DIM, D_C } = DeepSeekModel;
        const seq = tokens.length;
        const stdSize  = N_HEADS * seq * HEAD_DIM;  // tamaño KV estándar
        const mlaSize  = D_C * seq;                  // tamaño KV comprimido

        const section = _section(`MLA — Compresión KV (Capa ${layerIdx + 1})`, '#6a1b9a', 'mla-tooltip');

        const compare = _div('kv-compare');

        // Barra KV estándar
        const stdBox = _div('kv-box kv-box--std');
        stdBox.innerHTML = `
            <div class="kv-box-title">KV estándar (MHA)</div>
            <div class="kv-bar-track">
                <div class="kv-bar-fill kv-bar-fill--std" style="width:100%"></div>
            </div>
            <div class="kv-box-size">${stdSize} valores<br><small>${N_HEADS} heads × ${seq} tokens × ${HEAD_DIM} dim</small></div>`;
        compare.appendChild(stdBox);

        const arrow = _div('kv-arrow');
        arrow.innerHTML = `<span>MLA</span><br>×${(stdSize/mlaSize).toFixed(1)} compresión`;
        compare.appendChild(arrow);

        // Barra KV comprimido
        const mlaBox = _div('kv-box kv-box--mla');
        const pct = Math.round(mlaSize / stdSize * 100);
        mlaBox.innerHTML = `
            <div class="kv-box-title">KV comprimido (MLA)</div>
            <div class="kv-bar-track">
                <div class="kv-bar-fill kv-bar-fill--mla" style="width:${pct}%"></div>
            </div>
            <div class="kv-box-size">${mlaSize} valores<br><small>${D_C} dim × ${seq} tokens</small></div>`;
        compare.appendChild(mlaBox);

        section.appendChild(compare);

        // Vectores latentes por token
        const latTitle = _div('subsection-title');
        latTitle.textContent = `Vectores latentes C (${D_C}D) por token:`;
        section.appendChild(latTitle);

        const am = absMax2D(latent);
        latent.forEach((vec, i) => {
            const row = _div('latent-row');
            const lbl = _span(tokens[i].word, 'latent-lbl');
            lbl.style.color = TOKEN_COLORS[i % TOKEN_COLORS.length];
            row.appendChild(lbl);
            vec.forEach(v => {
                const cell = _div('latent-cell');
                cell.style.background = divergingColor(v, am);
                cell.textContent = v.toFixed(2);
                row.appendChild(cell);
            });
            section.appendChild(row);
        });

        container.appendChild(section);
    }

    // ── renderMLAAttention ────────────────────────────────────────────────────

    function renderMLAAttention(attnScores, tokens, layerIdx, container) {
        container.innerHTML = '';

        const section = _section(`MLA — Atención Multi-cabezal (Capa ${layerIdx + 1})`, '#6a1b9a', 'mla-tooltip');

        const info = _div('step-note');
        info.innerHTML = `Scores promediados entre ${DeepSeekModel.N_HEADS} cabezales. Filas = token consulta (Q), columnas = token clave (K). Cada celda indica cuánta "atención" le presta Q a K.`;
        section.appendChild(info);

        const heatWrap = _div('attn-heatmap-wrap');

        // Encabezado de columnas
        const header = _div('attn-header');
        header.appendChild(_span('', 'attn-corner'));
        tokens.forEach((t, j) => {
            const lbl = _span(t.word, 'attn-col-lbl');
            lbl.style.color = TOKEN_COLORS[j % TOKEN_COLORS.length];
            header.appendChild(lbl);
        });
        heatWrap.appendChild(header);

        // Filas del heatmap
        const maxScore = Math.max(...attnScores.flat());
        attnScores.forEach((row, i) => {
            const rowDiv = _div('attn-row');
            const lbl = _span(tokens[i].word, 'attn-row-lbl');
            lbl.style.color = TOKEN_COLORS[i % TOKEN_COLORS.length];
            rowDiv.appendChild(lbl);
            row.forEach(v => {
                const cell = _div('attn-cell');
                cell.style.background = sequentialColor(v, maxScore);
                cell.textContent = v.toFixed(2);
                rowDiv.appendChild(cell);
            });
            heatWrap.appendChild(rowDiv);
        });

        section.appendChild(heatWrap);
        _appendLegendSequential(section, '0', 'máx');
        container.appendChild(section);
    }

    // ── _renderTokenSelector ─────────────────────────────────────────────────

    function _renderTokenSelector(tokens, selectedIdx, onSelect) {
        const row = _div('token-selector-row');
        row.appendChild(_span('Token:', 'token-selector-lbl'));
        tokens.forEach((t, i) => {
            const chip = _span(t.word, 'token-sel-chip' + (i === selectedIdx ? ' token-sel-chip--active' : ''));
            chip.addEventListener('click', () => onSelect(i));
            row.appendChild(chip);
        });
        const hint = _span('(clic para cambiar)', 'token-selector-hint');
        row.appendChild(hint);
        return row;
    }

    // ── renderMoERouter ───────────────────────────────────────────────────────

    function renderMoERouter(moeResults, tokens, topK, layerIdx, container) {
        container.innerHTML = '';

        const section = _section(`MoE — Router (Capa ${layerIdx + 1})`, '#bf360c', 'router-tooltip');

        const parallelNote = _div('step-note');
        parallelNote.innerHTML = `El router asigna scores a los ${DeepSeekModel.N_EXPERTS} expertos para <strong>cada token por separado</strong>. Seleccioná un token para ver su routing:`;
        section.appendChild(parallelNote);

        const body = _div('moe-body');

        function renderRouterBody(idx) {
            body.innerHTML = '';
            const { scores, topKIdx } = moeResults[idx];

            const info = _div('step-note');
            info.innerHTML = `Scores del router para <strong>"${tokens[idx].word}"</strong> (sigmoid). Los <strong>Top-${topK}</strong> con mayor score son seleccionados.`;
            body.appendChild(info);

            const bars = _div('router-bars');
            scores.forEach((s, ei) => {
                const isSelected = topKIdx.includes(ei);
                const row = _div('router-bar-row' + (isSelected ? ' router-bar-selected' : ''));
                row.appendChild(_span(`Experto ${ei + 1}`, 'router-expert-lbl'));
                const track = _div('bar-track');
                const fill  = _div('bar-fill');
                fill.style.width = `${Math.round(s * 100)}%`;
                fill.style.background = isSelected ? '#e67e22' : '#90a4ae';
                track.appendChild(fill);
                row.appendChild(track);
                row.appendChild(_span(s.toFixed(3), 'bar-pct'));
                if (isSelected) row.appendChild(_span('SELECCIONADO', 'selected-badge'));
                bars.appendChild(row);
            });
            body.appendChild(bars);
        }

        let selIdx = 0;
        const selector = _renderTokenSelector(tokens, selIdx, idx => {
            selIdx = idx;
            selector.querySelectorAll('.token-sel-chip').forEach((c, i) => {
                c.className = 'token-sel-chip' + (i === idx ? ' token-sel-chip--active' : '');
            });
            renderRouterBody(idx);
        });
        section.appendChild(selector);
        section.appendChild(body);
        renderRouterBody(0);

        const moreNote = _div('step-note step-note--info');
        moreNote.innerHTML = `El router usa función <strong>sigmoide</strong> (no softmax), lo que permite seleccionar expertos de forma independiente. En DeepSeek-V3 real hay 256 expertos y se seleccionan 8.`;
        section.appendChild(moreNote);

        container.appendChild(section);
    }

    // ── renderMoEExperts ──────────────────────────────────────────────────────

    function renderMoEExperts(moeResults, tokens, topK, layerIdx, container) {
        container.innerHTML = '';

        const section = _section(`MoE — Expertos Activos (Capa ${layerIdx + 1})`, '#bf360c', 'moe-tooltip');

        const parallelNote = _div('step-note');
        parallelNote.innerHTML = `Cada token activa sus propios expertos. El proceso ocurre en <strong>paralelo para todos los tokens</strong>. Seleccioná un token:`;
        section.appendChild(parallelNote);

        const body = _div('moe-body');

        function renderExpertsBody(idx) {
            body.innerHTML = '';
            const { topKIdx, topKW, expOuts, sharedOut } = moeResults[idx];

            const info = _div('step-note');
            info.innerHTML = `Salida de los <strong>${topK}</strong> expertos seleccionados + el experto compartido (siempre activo) para <strong>"${tokens[idx].word}"</strong>.`;
            body.appendChild(info);

            const boxes = _div('expert-boxes');
            expOuts.forEach((out, k) => {
                const box = _div('expert-box expert-box--selected');
                box.innerHTML = `<div class="expert-box-title">Experto ${topKIdx[k] + 1} <span class="expert-weight">peso ${(topKW[k]*100).toFixed(0)}%</span></div>`;
                box.appendChild(_renderMiniBarList(out, '#e67e22'));
                boxes.appendChild(box);
            });
            const sharedBox = _div('expert-box expert-box--shared');
            sharedBox.innerHTML = `<div class="expert-box-title">Experto Compartido <span class="expert-weight">siempre activo</span></div>`;
            sharedBox.appendChild(_renderMiniBarList(sharedOut, '#8e44ad'));
            boxes.appendChild(sharedBox);
            body.appendChild(boxes);
        }

        let selIdx = 0;
        const selector = _renderTokenSelector(tokens, selIdx, idx => {
            selIdx = idx;
            selector.querySelectorAll('.token-sel-chip').forEach((c, i) => {
                c.className = 'token-sel-chip' + (i === idx ? ' token-sel-chip--active' : '');
            });
            renderExpertsBody(idx);
        });
        section.appendChild(selector);
        section.appendChild(body);
        renderExpertsBody(0);

        container.appendChild(section);
    }

    function _renderMiniBarList(values, color) {
        const wrap = _div('mini-bars');
        const max = posMax1D(values);
        values.forEach((v, d) => {
            const row = _div('mini-bar-row');
            const lbl = _span(`d${d}`, 'mini-bar-lbl');
            const track = _div('bar-track mini-bar-track');
            const fill  = _div('bar-fill');
            fill.style.width = `${Math.round(Math.abs(v) / max * 100)}%`;
            fill.style.background = v >= 0 ? color : '#78909c';
            track.appendChild(fill);
            const val = _span(v.toFixed(2), 'mini-bar-val');
            row.appendChild(lbl);
            row.appendChild(track);
            row.appendChild(val);
            wrap.appendChild(row);
        });
        return wrap;
    }

    // ── renderMoECombine ──────────────────────────────────────────────────────

    function renderMoECombine(moeResults, tokens, topK, layerIdx, container) {
        container.innerHTML = '';

        const section = _section(`MoE — Combinación de Expertos (Capa ${layerIdx + 1})`, '#bf360c', 'moe-tooltip');

        const parallelNote = _div('step-note');
        parallelNote.innerHTML = `La combinación ponderada ocurre en <strong>paralelo para todos los tokens</strong>. Seleccioná un token:`;
        section.appendChild(parallelNote);

        const body = _div('moe-body');
        const colors = ['#e67e22', '#d35400', '#c0392b', '#8e44ad'];

        function renderCombineBody(idx) {
            body.innerHTML = '';
            const { topKIdx, topKW } = moeResults[idx];

            const info = _div('step-note');
            info.innerHTML = `Suma ponderada de las salidas de los expertos para <strong>"${tokens[idx].word}"</strong>. Las contribuciones proporcionales se muestran en la barra apilada.`;
            body.appendChild(info);

            const sharedW = 1 / (topK + 1);
            const totalW  = topKW.reduce((s, w) => s + w, 0) + sharedW;
            const contributions = topKW.map(w => w / totalW).concat([sharedW / totalW]);
            const labels = topKIdx.map((ei, k) => `Exp. ${ei+1} (${(topKW[k]*100).toFixed(0)}%)`).concat([`Compartido (${(sharedW*100).toFixed(0)}%)`]);

            const stackWrap = _div('stacked-bar-wrap');
            const stackBar  = _div('stacked-bar');
            contributions.forEach((w, k) => {
                const seg = _div('stack-seg');
                seg.style.width = `${Math.round(w * 100)}%`;
                seg.style.background = colors[k % colors.length];
                seg.title = labels[k];
                stackBar.appendChild(seg);
            });
            stackWrap.appendChild(stackBar);

            const legend = _div('stack-legend');
            contributions.forEach((w, k) => {
                const item = _div('stack-legend-item');
                const swatch = _div('stack-swatch');
                swatch.style.background = colors[k % colors.length];
                item.appendChild(swatch);
                item.appendChild(_span(labels[k], 'stack-legend-lbl'));
                legend.appendChild(item);
            });
            stackWrap.appendChild(legend);
            body.appendChild(stackWrap);
        }

        let selIdx = 0;
        const selector = _renderTokenSelector(tokens, selIdx, idx => {
            selIdx = idx;
            selector.querySelectorAll('.token-sel-chip').forEach((c, i) => {
                c.className = 'token-sel-chip' + (i === idx ? ' token-sel-chip--active' : '');
            });
            renderCombineBody(idx);
        });
        section.appendChild(selector);
        section.appendChild(body);
        renderCombineBody(0);

        const note = _div('step-note step-note--info');
        note.innerHTML = `La salida combinada se suma a la entrada de cada token (conexión residual). El <strong>último token</strong> acumula contexto de toda la frase vía atención — su representación final se usa para predecir la siguiente palabra.`;
        section.appendChild(note);

        container.appendChild(section);
    }

    // ── renderOutput ──────────────────────────────────────────────────────────

    function renderOutput(top5, probs, tokens, container) {
        container.innerHTML = '';

        const section = _section('Predicción del siguiente token', '#b71c1c', 'output-tooltip');

        // ── Frase de entrada → siguiente token ───────────────────────────────
        const phraseRow = _div('output-phrase-row');
        tokens.forEach(t => {
            const chip = _span(t.word, 'output-phrase-chip');
            phraseRow.appendChild(chip);
            const arrow = _span('→', 'output-phrase-arrow');
            phraseRow.appendChild(arrow);
        });
        const nextSlot = _span('?', 'output-phrase-next');
        phraseRow.appendChild(nextSlot);
        section.appendChild(phraseRow);

        const info = _div('step-note');
        info.innerHTML = `Dado el contexto completo de la frase, el modelo predice cuál es la <strong>siguiente palabra</strong> más probable dentro del vocabulario de 20 palabras.`;
        section.appendChild(info);

        // ── Palabra predicha en grande ────────────────────────────────────────
        const phrase = tokens.map(t => t.word).join(' ');
        const phraseIdx = DeepSeekModel.PHRASES.indexOf(phrase);
        const expectedWord = phraseIdx >= 0 ? DeepSeekModel.PHRASE_TARGETS[phraseIdx] : null;
        const isCorrect = expectedWord && top5[0].word === expectedWord;

        const predicted = _div('predicted-word');
        predicted.textContent = `"${top5[0].word}"`;
        if (isCorrect)  predicted.style.color = '#2e7d32';
        section.appendChild(predicted);

        const predNote = _div('step-note');
        if (expectedWord) {
            const verdict = isCorrect
                ? `<span style="color:#2e7d32;font-weight:800">✓ Correcto</span> — probabilidad <strong>${(top5[0].p * 100).toFixed(1)}%</strong>`
                : `<span style="color:#c0392b;font-weight:800">✗ Incorrecto</span> — predijo <em>"${top5[0].word}"</em> pero se esperaba <em>"${expectedWord}"</em> (prob. ${(probs[DeepSeekModel.VOCAB.indexOf(expectedWord)] * 100).toFixed(1)}%)`;
            predNote.innerHTML = verdict;
        } else {
            predNote.innerHTML = `Probabilidad: <strong>${(top5[0].p * 100).toFixed(1)}%</strong>`;
        }
        section.appendChild(predNote);

        // ── Vocabulario completo (20 palabras) ────────────────────────────────
        const vocabTitle = _div('subsection-title');
        vocabTitle.textContent = 'Probabilidades sobre el vocabulario completo (20 palabras):';
        section.appendChild(vocabTitle);

        const allRanked = DeepSeekModel.VOCAB
            .map((word, i) => ({ word, p: probs[i], isTop: word === top5[0].word }))
            .sort((a, b) => b.p - a.p);

        const barList = _div('pred-bars');
        allRanked.forEach((item) => {
            const row = _div('pred-bar-row' + (item.isTop ? ' pred-bar-row--top' : ''));
            const lbl   = _span(item.word, 'pred-word-lbl');
            const track = _div('bar-track');
            const fill  = _div('bar-fill');
            fill.style.width = `${Math.max(1, Math.round(item.p * 100))}%`;
            fill.style.background = item.isTop ? '#c0392b' : '#e74c3c';
            fill.style.opacity = item.isTop ? 1 : 0.45 + item.p * 2;
            track.appendChild(fill);
            const pct = _span(`${(item.p * 100).toFixed(1)}%`, 'bar-pct');
            row.appendChild(lbl);
            row.appendChild(track);
            row.appendChild(pct);
            barList.appendChild(row);
        });

        section.appendChild(barList);

        const dsNote = _div('step-note step-note--info');
        dsNote.innerHTML = `En DeepSeek-V3 real el vocabulario tiene 128 000 tokens. La generación de texto consiste en repetir este proceso: la palabra predicha se agrega a la frase y se vuelve a correr la inferencia para predecir la siguiente.`;
        section.appendChild(dsNote);

        container.appendChild(section);
    }

    // ── Panel de bienvenida ───────────────────────────────────────────────────

    function renderWelcome(container) {
        container.innerHTML = '';
        const section = _div('welcome-panel');
        section.innerHTML = `
            <div class="welcome-icon">⚡</div>
            <h3>Simulador de Arquitectura DeepSeek</h3>
            <p>Seleccioná una frase y presioná <strong>Siguiente paso</strong> para visualizar<br>
            el flujo de datos a través del modelo paso a paso.</p>
            <ul class="welcome-list">
                <li>Tokenización y embeddings</li>
                <li>Multi-Head Latent Attention (MLA) con KV comprimido</li>
                <li>Mixture of Experts (MoE) con routing y combinación</li>
                <li>Predicción del siguiente token</li>
            </ul>`;
        container.appendChild(section);
    }

    // ── updateStepDescription ─────────────────────────────────────────────────

    function updateStepDescription(text) {
        const el = document.getElementById('step-description');
        if (el) el.textContent = text;
    }

    // ── Helpers de construcción DOM ───────────────────────────────────────────

    function _section(title, accentColor, tooltipId) {
        const sec = _div('step-section');
        const h   = document.createElement('h3');
        h.className = tooltipId ? 'step-title tip-title' : 'step-title';
        h.textContent = title;
        h.style.borderLeftColor = accentColor;
        h.style.color = accentColor;
        if (tooltipId) h.dataset.tooltip = tooltipId;
        sec.appendChild(h);
        return sec;
    }

    function _div(cls) {
        const el = document.createElement('div');
        if (cls) el.className = cls;
        return el;
    }

    function _span(text, cls) {
        const el = document.createElement('span');
        if (cls) el.className = cls;
        el.textContent = text;
        return el;
    }

    function _appendLegendDiverging(parent) {
        const leg = _div('colormap-legend legend-diverging');
        leg.innerHTML = `<span>–</span><div class="legend-bar bar-diverging"></div><span>+</span>`;
        parent.appendChild(leg);
    }

    function _appendLegendSequential(parent, low, high) {
        const leg = _div('colormap-legend legend-sequential');
        leg.innerHTML = `<span>${low}</span><div class="legend-bar bar-sequential"></div><span>${high}</span>`;
        parent.appendChild(leg);
    }

    // ── renderComparison ──────────────────────────────────────────────────────

    function renderComparison(tokens, topK, container) {
        container.innerHTML = '';

        const { D_MODEL, D_C, N_HEADS, HEAD_DIM, N_EXPERTS, FFN_HIDDEN } = DeepSeekModel;
        const seq = tokens.length;

        // ── Métricas simuladas ────────────────────────────────────────────────
        const kvStd   = N_HEADS * HEAD_DIM * 2;               // KV cache estándar por token
        const kvMLA   = D_C;                                   // KV cache MLA por token
        const ffnTot  = N_EXPERTS * FFN_HIDDEN * D_MODEL * 2; // params FFN total MoE
        const ffnAct  = (topK + 1) * FFN_HIDDEN * D_MODEL * 2;// params activos (Top-K + shared)
        const ffnStd  = ffnTot;                                // equiv. denso: todos activos

        // ── Sección principal ─────────────────────────────────────────────────
        const section = _section('Comparativa: DeepSeek vs Transformer Tradicional', '#1a237e', null);

        const intro = _div('step-note');
        intro.innerHTML = `Esta simulación modela las innovaciones clave de DeepSeek: <strong>MLA</strong> (compresión del KV cache) y <strong>MoE</strong> (activación parcial de parámetros). A continuación se compara con un Transformer denso equivalente.`;
        section.appendChild(intro);

        // ── Diagrama de arquitectura comparado ───────────────────────────────
        const archTitle = _div('cmp-section-title');
        archTitle.textContent = 'Arquitectura';
        section.appendChild(archTitle);

        const archGrid = _div('cmp-arch-grid');

        function _archBox(title, color, rows) {
            const box = _div('cmp-arch-box');
            box.style.borderColor = color;
            const h = _div('cmp-arch-box-title');
            h.style.background = color;
            h.textContent = title;
            box.appendChild(h);
            rows.forEach(([label, detail, good]) => {
                const row = _div('cmp-arch-row');
                const lbl = _div('cmp-arch-row-lbl');
                lbl.textContent = label;
                const det = _div('cmp-arch-row-det' + (good ? ' cmp-good' : ' cmp-bad'));
                det.textContent = detail;
                row.appendChild(lbl);
                row.appendChild(det);
                box.appendChild(row);
            });
            return box;
        }

        archGrid.appendChild(_archBox('Transformer Tradicional', '#546e7a', [
            ['Embedding',      'Lookup table — igual', null],
            ['Atención',       'Multi-Head Attention (MHA)', false],
            ['KV Cache',       `${kvStd} valores/token/capa`, false],
            ['FFN',            'Capa densa — todos los pesos activos', false],
            ['Params activos', '100 % en cada forward pass', false],
            ['Escalado',       'Costo ↑ linealmente con parámetros', false],
        ]));

        archGrid.appendChild(_archBox('DeepSeek (MLA + MoE)', '#1565c0', [
            ['Embedding',      'Lookup table — igual', null],
            ['Atención',       'Multi-Head Latent Attention (MLA)', true],
            ['KV Cache',       `${kvMLA} valores/token/capa (vector latente C)`, true],
            ['FFN',            `MoE: Top-${topK} de ${N_EXPERTS} expertos + 1 compartido`, true],
            ['Params activos', `${((ffnAct/ffnTot)*100).toFixed(0)} % por forward pass`, true],
            ['Escalado',       'Params totales ↑ pero costo constante', true],
        ]));

        section.appendChild(archGrid);

        // ── Barras comparativas (modelo simulado) ─────────────────────────────
        const simTitle = _div('cmp-section-title');
        simTitle.textContent = `Métricas — Modelo pedagógico (${seq} tokens, ${tokens.map(t=>'"'+t.word+'"').join(', ')})`;
        section.appendChild(simTitle);

        function _cmpBar(label, valA, valB, maxVal, unitA, unitB, note) {
            const row = _div('cmp-bar-group');

            const lbl = _div('cmp-bar-label');
            lbl.textContent = label;
            row.appendChild(lbl);

            function _bar(val, max, color, tag, unit) {
                const wrap = _div('cmp-bar-wrap');
                const tagEl = _span(tag, 'cmp-bar-tag');
                tagEl.style.background = color;
                wrap.appendChild(tagEl);
                const track = _div('cmp-bar-track');
                const fill = _div('cmp-bar-fill');
                fill.style.width = `${Math.round(val / max * 100)}%`;
                fill.style.background = color;
                track.appendChild(fill);
                wrap.appendChild(track);
                const valEl = _span(`${val} ${unit}`, 'cmp-bar-val');
                wrap.appendChild(valEl);
                return wrap;
            }

            row.appendChild(_bar(valA, maxVal, '#78909c', 'Tradicional', unitA));
            row.appendChild(_bar(valB, maxVal, '#1565c0', 'DeepSeek',    unitB));

            if (note) {
                const n = _div('cmp-bar-note');
                n.innerHTML = note;
                row.appendChild(n);
            }
            return row;
        }

        section.appendChild(_cmpBar(
            'KV Cache por token por capa',
            kvStd, kvMLA, kvStd, 'valores', 'valores',
            `DeepSeek almacena solo el vector latente C (${kvMLA} valores) en vez de todos los K y V por cabeza → <strong>${(kvStd/kvMLA).toFixed(0)}× menos memoria</strong>`
        ));

        section.appendChild(_cmpBar(
            'Parámetros FFN activos por token',
            ffnStd, ffnAct, ffnStd, 'params', 'params',
            `Solo ${topK} expertos + 1 compartido se activan → <strong>${((ffnAct/ffnTot)*100).toFixed(0)}% de los parámetros FFN</strong> por forward pass`
        ));

        section.appendChild(_cmpBar(
            'KV Cache total (frase completa, por capa)',
            kvStd * seq, kvMLA * seq, kvStd * seq, 'valores', 'valores',
            `Con ${seq} tokens en la frase: ${kvStd*seq} valores guardados vs ${kvMLA*seq}`
        ));

        // ── Tabla modelo real ─────────────────────────────────────────────────
        const realTitle = _div('cmp-section-title');
        realTitle.textContent = 'Modelo Real — DeepSeek-V3 vs LLM Denso equivalente';
        section.appendChild(realTitle);

        const realNote = _div('step-note');
        realNote.innerHTML = `Un LLM denso "equivalente" tendría los mismos parámetros de DeepSeek-V3 pero todos activos en cada token (sin MoE ni MLA).`;
        section.appendChild(realNote);

        const tbl = document.createElement('table');
        tbl.className = 'cmp-table';
        const head = tbl.createTHead();
        const hr = head.insertRow();
        ['Métrica', 'LLM Denso equivalente', 'DeepSeek-V3', 'Ventaja'].forEach((h, i) => {
            const th = document.createElement('th');
            th.textContent = h;
            if (i === 2) th.style.color = '#1565c0';
            hr.appendChild(th);
        });

        const rows = [
            ['Parámetros totales',      '671 B',         '671 B',        '—'],
            ['Parámetros activos/token','671 B (100 %)',  '37 B (5.5 %)', '18× menos cómputo'],
            ['Dimensión d_model',       '7 168',          '7 168',        '—'],
            ['Cabezas de atención',     '128 (MHA)',      '128 (MLA)',    '—'],
            ['KV cache/token/capa',     '32 768 valores', '512 valores',  '64× compresión'],
            ['Expertos FFN',            '1 (denso)',       '256 totales, 8 activos', '37 B activos vs 671 B'],
            ['Capas transformer',       '61',             '61',           '—'],
            ['Memoria KV (100k tokens)','~400 GB',        '~6 GB',        '~66× menos'],
        ];

        const body = tbl.createTBody();
        rows.forEach(([met, den, ds, adv]) => {
            const tr = body.insertRow();
            const tdM = tr.insertCell(); tdM.textContent = met; tdM.className = 'cmp-td-metric';
            const tdD = tr.insertCell(); tdD.textContent = den; tdD.className = 'cmp-td-bad';
            const tdS = tr.insertCell(); tdS.textContent = ds;  tdS.className = 'cmp-td-good';
            const tdA = tr.insertCell(); tdA.innerHTML  = adv !== '—' ? `<strong>${adv}</strong>` : '—'; tdA.className = 'cmp-td-adv';
        });

        section.appendChild(tbl);

        // ── Conclusión ────────────────────────────────────────────────────────
        const conclusion = _div('cmp-conclusion');
        conclusion.innerHTML = `
            <div class="cmp-conclusion-title">¿Por qué importa?</div>
            <div class="cmp-conclusion-body">
                MoE permite escalar a 671 B parámetros totales manteniendo solo 37 B activos por token,
                logrando la capacidad de un modelo grande con el costo de uno pequeño.
                MLA comprime el KV cache 64×, reduciendo drásticamente la memoria en inferencia con contextos largos.
                Juntas, estas innovaciones hacen que DeepSeek-V3 sea competitivo con GPT-4 a una
                <strong>fracción del costo de entrenamiento e inferencia</strong>.
            </div>`;
        section.appendChild(conclusion);

        container.appendChild(section);
    }

    return {
        renderTokens,
        renderEmbedding,
        renderMLACompress,
        renderMLAAttention,
        renderMoERouter,
        renderMoEExperts,
        renderMoECombine,
        renderOutput,
        renderComparison,
        renderWelcome,
        updateStepDescription
    };
})();
