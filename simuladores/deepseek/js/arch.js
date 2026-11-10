// arch.js — Diagrama SVG del pipeline DeepSeek estilo académico
const DeepSeekArch = (function () {

    const PALETTE = {
        tokens:    '#455a64',
        embed:     '#1565c0',
        mla:       '#6a1b9a',
        moe:       '#bf360c',
        output:    '#b71c1c',
        arrow:     '#90a4ae',
        text:      '#263238',
        dim:       '#78909c',
        highlight: '#f9a825',
        bg:        '#fafbfc',
        bracket:   '#b0bec5'
    };

    let _svg = null;
    let _activeId = null;

    function render(nLayers) {
        const container = document.getElementById('arch-diagram');
        if (!container) return;
        container.innerHTML = '';

        const VW = 820, VH = 124;
        const CY = 52, LY = VH - 14, SY = VH - 4;

        const svg = _makeSVG(VW, VH);
        _svg = svg;

        // Fondo
        const bg = _el('rect');
        _attr(bg, { x: 0, y: 0, width: VW, height: VH, fill: PALETTE.bg });
        svg.appendChild(bg);

        svg.appendChild(_text('Arquitectura DeepSeek (simplificado pedagógicamente)', 6, 9,
            { fill: PALETTE.dim, 'font-size': 7.5, 'font-style': 'italic', 'text-anchor': 'start' }));

        // Definición de nodos
        const nodes = [
            { id: 'tokens', label: 'Tokens', sub: 'texto → IDs', color: PALETTE.tokens, h: 36, w: 24 },
            { id: 'embed',  label: 'Embedding', sub: 'd = 8',   color: PALETTE.embed,  h: 54, w: 26 },
        ];

        for (let l = 0; l < nLayers; l++) {
            nodes.push(
                { id: `mla_${l}`, label: `MLA ${l+1}`, sub: 'KV comprimido', color: PALETTE.mla, h: 70, w: 28, layerIdx: l },
                { id: `moe_${l}`, label: `MoE ${l+1}`, sub: 'top-K expertos', color: PALETTE.moe, h: 70, w: 28, layerIdx: l }
            );
        }

        nodes.push(
            { id: 'output', label: 'Output', sub: 'next-token', color: PALETTE.output, h: 54, w: 26 }
        );

        // Posicionar nodos equiespaciados
        const margin = 42;
        const step   = (VW - 2 * margin) / (nodes.length - 1);
        nodes.forEach((nd, i) => {
            nd.cx = Math.round(margin + i * step);
            nd.cy = CY;
        });

        // Flechas (antes de los nodos para quedar detrás)
        for (let i = 0; i < nodes.length - 1; i++) {
            const a = nodes[i], b = nodes[i + 1];
            _drawArrow(svg, a.cx + a.w + 1, b.cx - b.w - 1, CY);
        }

        // Corchetes de capa (dashed bracket encima de pares MLA+MoE)
        for (let l = 0; l < nLayers; l++) {
            const mlaNode = nodes.find(n => n.id === `mla_${l}`);
            const moeNode = nodes.find(n => n.id === `moe_${l}`);
            if (mlaNode && moeNode) {
                const x1 = mlaNode.cx - mlaNode.w - 2;
                const x2 = moeNode.cx + moeNode.w + 2;
                const by = CY - mlaNode.h / 2 - 8;
                _drawBracket(svg, x1, x2, by, `Capa ${l + 1}`);
            }
        }

        // Nodos
        nodes.forEach(nd => {
            const g = _el('g');
            g.setAttribute('id', `arch-${nd.id}`);
            g.setAttribute('class', 'arch-layer');
            g.style.transition = 'opacity 0.3s, filter 0.3s';

            const rect = _el('rect');
            _attr(rect, {
                x: nd.cx - nd.w, y: nd.cy - nd.h / 2,
                width: nd.w * 2, height: nd.h,
                fill: nd.color, 'fill-opacity': 0.82,
                rx: 4, stroke: nd.color, 'stroke-width': 0.5, 'stroke-opacity': 0.5
            });
            g.appendChild(rect);

            // Letra inicial dentro del rectángulo
            const letter = nd.label[0];
            g.appendChild(_text(letter, nd.cx, nd.cy + 3,
                { fill: '#fff', 'font-size': 11, 'font-weight': 'bold',
                  'text-anchor': 'middle', opacity: 0.9 }));

            svg.appendChild(g);

            svg.appendChild(_text(nd.label, nd.cx, LY,
                { fill: PALETTE.text, 'font-size': 7.8, 'font-weight': 'bold', 'text-anchor': 'middle' }));
            svg.appendChild(_text(nd.sub, nd.cx, SY,
                { fill: PALETTE.dim, 'font-size': 6.8, 'text-anchor': 'middle' }));
        });

        container.appendChild(svg);

        if (_activeId) highlight(_activeId);
        else clearHighlight();
    }

    function highlight(layerId) {
        _activeId = layerId;
        if (!_svg) return;
        _svg.querySelectorAll('.arch-layer').forEach(g => {
            g.style.opacity = '0.2';
            g.style.filter  = '';
        });
        const target = _svg.querySelector(`#arch-${layerId}`);
        if (target) {
            target.style.opacity = '1';
            target.style.filter  = `drop-shadow(0 0 6px ${PALETTE.highlight})`;
        }
    }

    function clearHighlight() {
        _activeId = null;
        if (!_svg) return;
        _svg.querySelectorAll('.arch-layer').forEach(g => {
            g.style.opacity = '1';
            g.style.filter  = '';
        });
    }

    // ── Helpers de dibujo ─────────────────────────────────────────────────────

    function _drawArrow(parent, x1, x2, y) {
        if (x2 <= x1 + 8) return;
        const line = _el('line');
        _attr(line, { x1, y1: y, x2: x2 - 6, y2: y,
                      stroke: PALETTE.arrow, 'stroke-width': 1.2,
                      'stroke-dasharray': '3,2' });
        parent.appendChild(line);
        _polygon(parent, [[x2, y], [x2-7, y-3], [x2-7, y+3]], PALETTE.arrow, 1);
    }

    function _drawBracket(parent, x1, x2, y, label) {
        const tickH = 5;
        ['left', 'right', 'top'].forEach(part => {
            const line = _el('line');
            if (part === 'left')  _attr(line, { x1, y1: y + tickH, x2: x1, y2: y, stroke: PALETTE.bracket, 'stroke-width': 1, 'stroke-dasharray': '2,2' });
            if (part === 'right') _attr(line, { x1: x2, y1: y + tickH, x2, y2: y, stroke: PALETTE.bracket, 'stroke-width': 1, 'stroke-dasharray': '2,2' });
            if (part === 'top')   _attr(line, { x1, y1: y, x2, y2: y, stroke: PALETTE.bracket, 'stroke-width': 1, 'stroke-dasharray': '2,2' });
            parent.appendChild(line);
        });
        parent.appendChild(_text(label, (x1 + x2) / 2, y - 3,
            { fill: PALETTE.dim, 'font-size': 6.5, 'text-anchor': 'middle', 'font-style': 'italic' }));
    }

    function _polygon(parent, pts, fill, opacity) {
        const poly = _el('polygon');
        poly.setAttribute('points', pts.map(p => p.join(',')).join(' '));
        poly.setAttribute('fill', fill);
        poly.setAttribute('fill-opacity', opacity);
        parent.appendChild(poly);
    }

    function _text(content, x, y, attrs) {
        const t = _el('text');
        t.setAttribute('x', x);
        t.setAttribute('y', y);
        t.setAttribute('font-family', 'Segoe UI, Arial, sans-serif');
        Object.entries(attrs).forEach(([k, v]) => t.setAttribute(k, v));
        t.textContent = content;
        return t;
    }

    function _el(tag) {
        return document.createElementNS('http://www.w3.org/2000/svg', tag);
    }

    function _attr(el, attrs) {
        Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
    }

    function _makeSVG(w, h) {
        const svg = _el('svg');
        _attr(svg, { viewBox: `0 0 ${w} ${h}`, width: '100%', height: '100%',
                     preserveAspectRatio: 'xMidYMid meet' });
        return svg;
    }

    return { render, highlight, clearHighlight };
})();
