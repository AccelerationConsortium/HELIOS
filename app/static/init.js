/* ==========================================================================
   OTbot Campaign Initialization — Vanilla JS Conversation Client
   ========================================================================== */

const API = '/api/v1/init';

// State
let sessionId = null;
let currentRound = null;      // RoundPresentation
let completedRounds = [];     // [1, 2, ...]
let allRoundsDone = false;

// DOM refs
const chatMessages = document.getElementById('chat-messages');
const progressDots = document.querySelectorAll('.progress-dot');
const progressLabel = document.getElementById('progress-label');
const roundSubtitle = document.getElementById('round-subtitle');
const btnBack = document.getElementById('btn-back');
const btnContinue = document.getElementById('btn-continue');
const btnLaunch = document.getElementById('btn-launch');
const confirmOverlay = document.getElementById('confirm-overlay');
const btnCancelLaunch = document.getElementById('btn-cancel-launch');
const btnConfirmLaunch = document.getElementById('btn-confirm-launch');
const diffSummary = document.getElementById('diff-summary');
const diffRows = document.getElementById('diff-rows');

// Card body map: round -> { card, body, status, badge }
const CARD_MAP = {
    1: { card: 'card-goal', body: 'body-goal', status: 'status-goal', badge: 'badge-goal' },
    2: { card: 'card-safety', body: 'body-safety', status: 'status-safety', badge: 'badge-safety' },
    3: { card: 'card-protocol', body: 'body-protocol', status: 'status-protocol', badge: 'badge-protocol' },
    4: { card: 'card-params', body: 'body-params', status: 'status-params', badge: 'badge-params' },
    5: { card: 'card-gate', body: 'body-gate', status: 'status-gate', badge: 'badge-gate' },
};

const KPI_REFS = { card: 'card-kpi', body: 'body-kpi', status: 'status-kpi', badge: 'badge-kpi' };


/* ========================================================================
   Widget Renderers
   ======================================================================== */

const WIDGET_RENDERERS = {
    select(slot) {
        const container = document.createElement('div');
        const sel = document.createElement('select');
        sel.name = slot.name;
        sel.dataset.slotName = slot.name;

        const emptyOpt = document.createElement('option');
        emptyOpt.value = '';
        emptyOpt.textContent = '-- Select --';
        sel.appendChild(emptyOpt);

        (slot.options || []).forEach(opt => {
            const o = document.createElement('option');
            const optVal = typeof opt === 'object' ? (opt.id || opt) : opt;
            o.value = optVal;
            o.textContent = optVal;
            if (String(optVal) === String(slot.current_value)) o.selected = true;
            sel.appendChild(o);
        });

        if (!slot.current_value && slot.default) sel.value = slot.default;
        container.appendChild(sel);
        return container;
    },

    multiselect(slot) {
        const container = document.createElement('div');
        container.className = 'multiselect-group';
        container.dataset.slotName = slot.name;

        const selected = new Set(slot.current_value || slot.default || []);
        (slot.options || []).forEach(opt => {
            const chip = document.createElement('label');
            chip.className = 'multiselect-chip' + (selected.has(opt) ? ' selected' : '');

            const cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.value = opt;
            cb.checked = selected.has(opt);
            cb.addEventListener('change', () => chip.classList.toggle('selected', cb.checked));

            chip.appendChild(cb);
            chip.appendChild(document.createTextNode(opt));
            container.appendChild(chip);
        });
        return container;
    },

    number(slot) {
        const wrapper = document.createElement('div');
        wrapper.className = 'number-input-wrapper';

        const inp = document.createElement('input');
        inp.type = 'number';
        inp.name = slot.name;
        inp.dataset.slotName = slot.name;
        if (slot.min_val != null) inp.min = slot.min_val;
        if (slot.max_val != null) inp.max = slot.max_val;
        if (slot.step_val != null) inp.step = slot.step_val;

        const val = slot.current_value != null ? slot.current_value : slot.default;
        if (val != null) inp.value = val;
        wrapper.appendChild(inp);

        if (slot.unit) {
            const unit = document.createElement('span');
            unit.className = 'slot-unit';
            unit.textContent = slot.unit;
            wrapper.appendChild(unit);
        }
        return wrapper;
    },

    toggle(slot) {
        const wrapper = document.createElement('div');
        wrapper.className = 'toggle-wrapper';
        wrapper.dataset.slotName = slot.name;

        const val = slot.current_value != null ? slot.current_value : (slot.default || false);

        const sw = document.createElement('div');
        sw.className = 'toggle-switch' + (val ? ' active' : '');

        const label = document.createElement('span');
        label.className = 'toggle-label-text';
        label.textContent = val ? 'Yes' : 'No';

        sw.addEventListener('click', () => {
            sw.classList.toggle('active');
            label.textContent = sw.classList.contains('active') ? 'Yes' : 'No';
        });

        wrapper.appendChild(sw);
        wrapper.appendChild(label);
        return wrapper;
    },

    text(slot) {
        const ta = document.createElement('textarea');
        ta.name = slot.name;
        ta.dataset.slotName = slot.name;
        ta.placeholder = slot.hint || '';
        if (slot.current_value) ta.value = slot.current_value;
        return ta;
    },

    param_editor(slot) {
        const container = document.createElement('div');
        container.dataset.slotName = slot.name;
        container.className = 'param-editor-container';

        const params = slot.current_value || [];
        if (!params.length) {
            const empty = document.createElement('div');
            empty.className = 'display-value';
            empty.textContent = 'No parameters available. Select a protocol pattern first.';
            container.appendChild(empty);
            return container;
        }

        const table = document.createElement('table');
        table.className = 'param-editor';

        const thead = document.createElement('thead');
        thead.innerHTML = '<tr><th>Param</th><th>Min</th><th>Max</th><th>Unit</th><th>Optimize</th></tr>';
        table.appendChild(thead);

        const tbody = document.createElement('tbody');
        params.forEach((p, idx) => {
            const tr = document.createElement('tr');
            tr.dataset.paramIndex = idx;

            // Name + description
            const tdName = document.createElement('td');
            const nameDiv = document.createElement('div');
            nameDiv.className = 'param-name';
            nameDiv.textContent = p.param_name;
            tdName.appendChild(nameDiv);
            if (p.description) {
                const descDiv = document.createElement('div');
                descDiv.className = 'param-desc';
                descDiv.textContent = p.description;
                tdName.appendChild(descDiv);
            }
            tr.appendChild(tdName);

            // Min
            const tdMin = document.createElement('td');
            const minInp = document.createElement('input');
            minInp.type = 'number';
            minInp.className = 'param-min';
            minInp.value = p.min_value != null ? p.min_value : '';
            minInp.disabled = p.safety_locked;
            tdMin.appendChild(minInp);
            tr.appendChild(tdMin);

            // Max
            const tdMax = document.createElement('td');
            const maxInp = document.createElement('input');
            maxInp.type = 'number';
            maxInp.className = 'param-max';
            maxInp.value = p.max_value != null ? p.max_value : '';
            maxInp.disabled = p.safety_locked;
            tdMax.appendChild(maxInp);
            tr.appendChild(tdMax);

            // Unit + safety lock icon
            const tdUnit = document.createElement('td');
            tdUnit.innerHTML = '<span class="param-unit">' + (p.unit || '') + '</span>';
            if (p.safety_locked) {
                tdUnit.innerHTML += ' <span class="safety-lock" title="Safety-locked parameter">&#128274;</span>';
            }
            tr.appendChild(tdUnit);

            // Optimizable checkbox
            const tdOpt = document.createElement('td');
            const cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.className = 'optimizable-check';
            cb.checked = p.optimizable && !p.safety_locked;
            cb.disabled = p.safety_locked;
            tdOpt.appendChild(cb);
            tr.appendChild(tdOpt);

            tbody.appendChild(tr);
        });
        table.appendChild(tbody);
        container.appendChild(table);
        return container;
    },

    display(slot) {
        const container = document.createElement('div');
        container.dataset.slotName = slot.name;

        const val = slot.current_value;
        if (Array.isArray(val) && val.length > 0) {
            const listDiv = document.createElement('div');
            listDiv.className = 'display-list';
            val.forEach(item => {
                const tag = document.createElement('span');
                tag.className = 'display-tag';
                tag.textContent = item;
                listDiv.appendChild(tag);
            });
            container.appendChild(listDiv);
        } else {
            const d = document.createElement('div');
            d.className = 'display-value';
            d.textContent = val || 'None';
            container.appendChild(d);
        }
        return container;
    },
};


/* ========================================================================
   Collect Responses from current round widgets
   ======================================================================== */

function collectResponses() {
    if (!currentRound) return {};
    const responses = {};

    currentRound.slots.forEach(slot => {
        const w = slot.widget;

        if (w === 'select') {
            const el = document.querySelector(`select[data-slot-name="${slot.name}"]`);
            if (el) responses[slot.name] = el.value || null;
        }
        else if (w === 'multiselect') {
            const group = document.querySelector(`.multiselect-group[data-slot-name="${slot.name}"]`);
            if (group) {
                responses[slot.name] = Array.from(group.querySelectorAll('input:checked')).map(cb => cb.value);
            }
        }
        else if (w === 'number') {
            const el = document.querySelector(`input[data-slot-name="${slot.name}"]`);
            if (el && el.value !== '') responses[slot.name] = parseFloat(el.value);
            else responses[slot.name] = null;
        }
        else if (w === 'toggle') {
            const wrapper = document.querySelector(`.toggle-wrapper[data-slot-name="${slot.name}"]`);
            if (wrapper) {
                const sw = wrapper.querySelector('.toggle-switch');
                responses[slot.name] = sw ? sw.classList.contains('active') : false;
            }
        }
        else if (w === 'text') {
            const el = document.querySelector(`textarea[data-slot-name="${slot.name}"]`);
            if (el) responses[slot.name] = el.value || '';
        }
        else if (w === 'param_editor') {
            const container = document.querySelector(`.param-editor-container[data-slot-name="${slot.name}"]`);
            if (container) {
                const rows = container.querySelectorAll('tbody tr');
                const params = slot.current_value || [];
                responses[slot.name] = params.map((p, idx) => {
                    const row = rows[idx];
                    if (!row) return p;
                    const minInp = row.querySelector('.param-min');
                    const maxInp = row.querySelector('.param-max');
                    const optCb = row.querySelector('.optimizable-check');
                    return {
                        ...p,
                        min_value: minInp && minInp.value !== '' ? parseFloat(minInp.value) : p.min_value,
                        max_value: maxInp && maxInp.value !== '' ? parseFloat(maxInp.value) : p.max_value,
                        optimizable: optCb ? optCb.checked : p.optimizable,
                    };
                });
            }
        }
        // 'display' widgets are read-only
    });

    return responses;
}


/* ========================================================================
   Render a round into the chat panel
   ======================================================================== */

function renderRound(round) {
    currentRound = round;
    roundSubtitle.textContent = `Round ${round.round_number}: ${round.round_name}`;

    chatMessages.innerHTML = '';

    const msgDiv = document.createElement('div');
    msgDiv.className = 'message message-bot';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = 'OT';
    msgDiv.appendChild(avatar);

    const content = document.createElement('div');
    content.className = 'message-content';

    const roundLabel = document.createElement('div');
    roundLabel.className = 'message-round-name';
    roundLabel.textContent = `Round ${round.round_number} of 5 \u2014 ${round.round_name}`;
    content.appendChild(roundLabel);

    const text = document.createElement('div');
    text.className = 'message-text';
    text.textContent = round.message;
    content.appendChild(text);

    // Slot widgets
    const slotsDiv = document.createElement('div');
    slotsDiv.className = 'slots-container';

    round.slots.forEach(slot => {
        // Skip empty display slots
        if (slot.widget === 'display' && (!slot.current_value || (Array.isArray(slot.current_value) && slot.current_value.length === 0))) {
            return;
        }

        const group = document.createElement('div');
        group.className = 'slot-group' + (slot.error ? ' has-error' : '');

        const label = document.createElement('div');
        label.className = 'slot-label';
        label.textContent = slot.label;
        if (slot.required) {
            const req = document.createElement('span');
            req.className = 'required';
            req.textContent = '*';
            label.appendChild(req);
        }
        group.appendChild(label);

        if (slot.hint) {
            const hint = document.createElement('div');
            hint.className = 'slot-hint';
            hint.textContent = slot.hint;
            group.appendChild(hint);
        }

        const renderer = WIDGET_RENDERERS[slot.widget];
        if (renderer) group.appendChild(renderer(slot));

        if (slot.error) {
            const errDiv = document.createElement('div');
            errDiv.className = 'slot-error';
            errDiv.textContent = '\u26A0 ' + slot.error;
            group.appendChild(errDiv);
        }

        slotsDiv.appendChild(group);
    });

    content.appendChild(slotsDiv);
    msgDiv.appendChild(content);
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    // Update progress and buttons
    updateProgress(round.round_number);
    btnBack.disabled = round.round_number <= 1;

    if (allRoundsDone) {
        btnContinue.textContent = 'All Rounds Complete \u2713';
        btnContinue.disabled = true;
    } else {
        btnContinue.textContent = round.round_number === 5 ? 'Finish \u2713' : 'Continue \u2192';
        btnContinue.disabled = false;
    }

    updateActiveCard(round.round_number);
}


/* ========================================================================
   Pack Panel Updates
   ======================================================================== */

function updatePackPanel(preview) {
    if (!preview) return;

    if (preview.goal) {
        updateCardContent('goal', [
            ['Objective', preview.goal.objective_type],
            ['KPI', preview.goal.objective_kpi],
            ['Direction', preview.goal.direction],
            ['Target', preview.goal.target_value || '\u2014'],
        ]);
        updateCardContent('kpi', [
            ['Primary KPI', preview.goal.objective_kpi],
        ], KPI_REFS);
    }

    if (preview.safety) {
        const instruments = Array.isArray(preview.safety.available_instruments)
            ? preview.safety.available_instruments.join(', ') : '\u2014';
        updateCardContent('safety', [
            ['Instruments', instruments],
            ['Max Temp', (preview.safety.max_temp_c || '\u2014') + ' \u00B0C'],
            ['Max Volume', (preview.safety.max_volume_ul || '\u2014') + ' \u00B5L'],
        ]);
    }

    if (preview.protocol) {
        updateCardContent('protocol', [
            ['Pattern', preview.protocol.pattern_id],
        ]);
    }

    if (preview.param_space) {
        updateCardContent('params', [
            ['Strategy', preview.param_space.strategy],
            ['Batch Size', preview.param_space.batch_size],
            ['Dimensions', preview.param_space.n_params],
        ]);
    }

    if (preview.human_gate) {
        const triggers = Array.isArray(preview.human_gate.human_gate_triggers)
            ? preview.human_gate.human_gate_triggers.join(', ') : '\u2014';
        updateCardContent('gate', [
            ['Max Rounds', preview.human_gate.max_rounds],
            ['Plateau', preview.human_gate.plateau_threshold],
            ['Triggers', triggers],
        ]);
    }
}

function updateCardContent(name, kvPairs, refs) {
    const r = refs || findCardRefs(name);
    if (!r) return;

    const bodyEl = document.getElementById(r.body);
    const statusEl = document.getElementById(r.status);
    const badgeEl = document.getElementById(r.badge);
    const cardEl = document.getElementById(r.card);

    if (bodyEl) {
        bodyEl.innerHTML = kvPairs.map(([k, v]) =>
            `<div class="kv-row"><span class="kv-key">${k}</span><span class="kv-val">${v}</span></div>`
        ).join('');
    }

    if (statusEl) statusEl.className = 'pack-card-status filled';
    if (badgeEl) { badgeEl.textContent = 'filled'; badgeEl.className = 'status-badge completed'; }
    if (cardEl) cardEl.classList.add('filled');
}

function findCardRefs(name) {
    // Map card names to round numbers
    const nameToRound = { goal: 1, safety: 2, protocol: 3, params: 4, gate: 5 };
    const roundNum = nameToRound[name];
    return roundNum ? CARD_MAP[roundNum] : null;
}

function updateActiveCard(roundNum) {
    document.querySelectorAll('.pack-card').forEach(c => c.classList.remove('active'));
    document.querySelectorAll('.pack-card-status').forEach(s => {
        if (!s.classList.contains('filled')) s.className = 'pack-card-status';
    });

    const mapping = CARD_MAP[roundNum];
    if (mapping) {
        const cardEl = document.getElementById(mapping.card);
        const statusEl = document.getElementById(mapping.status);
        if (cardEl && !cardEl.classList.contains('filled')) cardEl.classList.add('active');
        if (statusEl && !statusEl.classList.contains('filled')) statusEl.classList.add('active');
    }
}


/* ========================================================================
   Progress Bar
   ======================================================================== */

function updateProgress(currentNum) {
    progressDots.forEach(dot => {
        const r = parseInt(dot.dataset.round);
        dot.classList.remove('completed', 'active');
        if (completedRounds.includes(r)) dot.classList.add('completed');
        else if (r === currentNum) dot.classList.add('active');
    });
    progressLabel.textContent = `Round ${currentNum}/5`;
}


/* ========================================================================
   API Helpers
   ======================================================================== */

async function apiPost(path, body) {
    const res = await fetch(API + path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: body ? JSON.stringify(body) : undefined,
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'API error');
    }
    return res.json();
}

async function apiGet(path) {
    const res = await fetch(API + path);
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'API error');
    }
    return res.json();
}


/* ========================================================================
   Flow Control
   ======================================================================== */

async function submitRound() {
    if (!sessionId) { showError('No active session'); return; }

    const responses = collectResponses();
    btnContinue.disabled = true;
    btnContinue.innerHTML = '<span class="loading-spinner"></span> Validating\u2026';

    try {
        const result = await apiPost(`/${sessionId}/respond`, { responses });

        if (result.success) {
            // Mark current round completed
            if (currentRound && !completedRounds.includes(currentRound.round_number)) {
                completedRounds.push(currentRound.round_number);
            }

            // Update pack panel
            if (result.injection_pack_preview) updatePackPanel(result.injection_pack_preview);

            // Check if all rounds done
            if (completedRounds.length >= 5) {
                allRoundsDone = true;
                btnLaunch.disabled = false;
                if (result.next_round) renderRound(result.next_round);
                updateProgress(5);
                return;
            }

            // Advance to next round
            if (result.next_round) renderRound(result.next_round);
        } else {
            // Validation error — re-render same round with errors
            if (result.next_round) renderRound(result.next_round);
        }
    } catch (err) {
        showError('Submission failed: ' + err.message);
    } finally {
        if (!allRoundsDone) {
            btnContinue.disabled = false;
            btnContinue.textContent = (currentRound && currentRound.round_number === 5) ? 'Finish \u2713' : 'Continue \u2192';
        }
    }
}

async function goBack() {
    if (!sessionId) return;
    try {
        const round = await apiPost(`/${sessionId}/back`);
        completedRounds = completedRounds.filter(r => r < round.round_number);
        allRoundsDone = false;
        btnLaunch.disabled = true;
        btnContinue.disabled = false;
        renderRound(round);
    } catch (err) {
        showError('Failed to go back: ' + err.message);
    }
}

async function confirmAndLaunch() {
    if (!sessionId) return;

    btnConfirmLaunch.disabled = true;
    btnConfirmLaunch.innerHTML = '<span class="loading-spinner"></span> Creating campaign\u2026';

    try {
        const result = await apiPost(`/${sessionId}/confirm`);
        confirmOverlay.style.display = 'none';

        // Show diff summary
        if (result.diff_summary && result.diff_summary.length > 0) {
            diffSummary.style.display = 'block';
            diffRows.innerHTML = result.diff_summary.map(d =>
                `<div class="diff-row"><span class="diff-field">${d.field}</span><span class="diff-value">${d.actual}</span></div>`
            ).join('');
        }

        // Update launch button
        const shortId = (result.campaign_id || '').substring(0, 8);
        btnLaunch.textContent = '\u2713 Campaign Created: ' + shortId;
        btnLaunch.disabled = true;
        btnLaunch.classList.remove('btn-success');
        btnLaunch.classList.add('btn-secondary');

        // Show warnings
        if (result.warnings && result.warnings.length > 0) {
            const warnDiv = document.createElement('div');
            warnDiv.className = 'slot-error';
            warnDiv.textContent = 'Warnings: ' + result.warnings.join('; ');
            document.getElementById('pack-footer').prepend(warnDiv);
        }

        roundSubtitle.textContent = 'Campaign created successfully!';
    } catch (err) {
        showError('Campaign creation failed: ' + err.message);
    } finally {
        btnConfirmLaunch.disabled = false;
        btnConfirmLaunch.textContent = 'Confirm & Launch';
    }
}


/* ========================================================================
   Utility
   ======================================================================== */

function showError(msg) {
    console.error(msg);
    const errDiv = document.createElement('div');
    errDiv.className = 'message';
    errDiv.innerHTML = `<div class="message-text" style="border-color: var(--accent-danger); color: var(--accent-danger);">\u26A0 ${escapeHtml(msg)}</div>`;
    chatMessages.appendChild(errDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}


/* ========================================================================
   Event Listeners
   ======================================================================== */

btnContinue.addEventListener('click', () => { if (!allRoundsDone) submitRound(); });
btnBack.addEventListener('click', () => goBack());
btnLaunch.addEventListener('click', () => { confirmOverlay.style.display = 'flex'; });
btnCancelLaunch.addEventListener('click', () => { confirmOverlay.style.display = 'none'; });
btnConfirmLaunch.addEventListener('click', () => confirmAndLaunch());


/* ========================================================================
   Boot — start session on page load
   ======================================================================== */

(async function boot() {
    try {
        const data = await apiPost('/start?created_by=user');
        sessionId = data.session_id;

        if (!sessionId) {
            showError('Server did not return a session_id. Please refresh.');
            return;
        }

        renderRound(data);
    } catch (err) {
        showError('Failed to initialize: ' + err.message);
    }
})();
