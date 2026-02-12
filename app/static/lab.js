/**
 * OTbot Lab Agent v2 — Three-Column Pipeline UI with SSE reasoning stream.
 *
 * Layout: Left (chat input) | Middle (agent pipeline) | Right (step context)
 *
 * Flow:
 *   1. Scientist types a paragraph → POST /api/v1/nl/parse
 *   2. Show extracted params → user confirms or edits
 *   3. POST /api/v1/orchestrate/start → campaign_id
 *   4. Connect SSE: GET /api/v1/orchestrate/{id}/events/stream
 *   5. Route SSE events to pipeline steps and context panel
 */

const API = '/api/v1';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let state = {
    phase: 'idle',           // idle | parsing | parsed | running | completed | error
    campaignId: null,
    parsedResult: null,
    eventSource: null,
    pollTimer: null,         // Polling timer ID
    roundsTotal: 0,
    roundsDone: 0,
    bestKpi: null,
    direction: 'minimize',   // from parsed result

    // Pipeline state
    pipeline: {
        steps: [],           // flat array of step objects
        activeStepId: null,  // currently thinking step
        selectedStepId: null,// user-clicked step
    },

    // KPI history for trend visualization
    kpiHistory: [],          // { round, kpi, params }

    // Well allocator info
    wellAllocator: null,

    // Auto-scroll control
    autoScroll: true,

    // Context panel
    contextPanelOpen: true,
};

// ---------------------------------------------------------------------------
// Pipeline Step Model
// ---------------------------------------------------------------------------

function createStep(id, type, agent, label, opts = {}) {
    return {
        id,
        type,           // parse | planner | round | strategy | design | compile | safety | execute | sensing | stop | complete
        agent,          // agent name (for color: planner, design, compiler, safety, executor, sensing, stop, parse, system)
        label,
        status: 'pending',  // pending | thinking | success | failure | warning
        detail: '',
        round: opts.round || null,
        data: {},           // full SSE event payload
        duration_ms: null,
        candidatesDone: 0,
        candidatesTotal: 0,
        timestamp: null,
        isChild: opts.isChild || false,
        isRound: opts.isRound || false,
    };
}

function findStep(stepId) {
    return state.pipeline.steps.find(s => s.id === stepId) || null;
}

function findCurrentRoundId() {
    // Find the latest round step
    for (let i = state.pipeline.steps.length - 1; i >= 0; i--) {
        if (state.pipeline.steps[i].isRound) {
            return state.pipeline.steps[i].id;
        }
    }
    return null;
}

function mapAgentToStepId(roundId, agent) {
    const map = {
        design: `${roundId}-design`,
        compiler: `${roundId}-compile`,
        safety: `${roundId}-safety`,
        executor: `${roundId}-execute`,
        sensing: `${roundId}-sensing`,
        stop: `${roundId}-stop`,
    };
    return map[agent] || `${roundId}-${agent}`;
}

// ---------------------------------------------------------------------------
// DOM References
// ---------------------------------------------------------------------------

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const mainInput       = $('#mainInput');
const sendBtn         = $('#sendBtn');
const welcomeSection  = $('#welcomeSection');
const parsedPreview   = $('#parsedPreview');
const parsedGrid      = $('#parsedGrid');
const campaignStatus  = $('#campaignStatus');
const campaignBadge   = $('#campaignBadge');
const progressBar     = $('#progressBar');
const statusStats     = $('#statusStats');
const stopBtn         = $('#stopBtn');
const editParamsBtn   = $('#editParamsBtn');
const pipelineContainer = $('#pipelineContainer');
const pipelineEmpty   = $('#pipelineEmpty');
const contextBody     = $('#contextBody');
const panelRight      = $('#panelRight');
const toggleContextBtn = $('#toggleContextBtn');
const closeContextBtn = $('#closeContextBtn');

// ---------------------------------------------------------------------------
// Event Listeners
// ---------------------------------------------------------------------------

sendBtn.addEventListener('click', handleSend);

mainInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
    }
});

// Auto-resize textarea
mainInput.addEventListener('input', () => {
    mainInput.style.height = 'auto';
    mainInput.style.height = Math.min(mainInput.scrollHeight, 150) + 'px';
});

// Example chips
$$('.example-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
        mainInput.value = chip.dataset.example;
        mainInput.style.height = 'auto';
        mainInput.style.height = Math.min(mainInput.scrollHeight, 150) + 'px';
        mainInput.focus();
    });
});

stopBtn.addEventListener('click', handleStop);

if (editParamsBtn) {
    editParamsBtn.addEventListener('click', () => {
        parsedPreview.style.display = 'none';
        welcomeSection.style.display = '';
        mainInput.focus();
        state.phase = 'idle';
    });
}

// Context panel toggle
if (toggleContextBtn) {
    toggleContextBtn.addEventListener('click', () => {
        panelRight.classList.toggle('open');
        state.contextPanelOpen = panelRight.classList.contains('open');
    });
}

if (closeContextBtn) {
    closeContextBtn.addEventListener('click', () => {
        panelRight.classList.remove('open');
        state.contextPanelOpen = false;
    });
}

// Auto-scroll control: disable when user scrolls up
if (pipelineContainer) {
    pipelineContainer.addEventListener('scroll', () => {
        const { scrollTop, scrollHeight, clientHeight } = pipelineContainer;
        state.autoScroll = (scrollHeight - scrollTop - clientHeight) < 60;
    });
}

// ---------------------------------------------------------------------------
// Send Handler
// ---------------------------------------------------------------------------

async function handleSend() {
    const text = mainInput.value.trim();
    if (!text) return;

    if (state.phase === 'idle' || state.phase === 'parsed' || state.phase === 'completed' || state.phase === 'error') {
        await parseAndLaunch(text);
    }
}

async function parseAndLaunch(text) {
    state.phase = 'parsing';
    sendBtn.disabled = true;

    // Initialize the pipeline
    initPipeline();

    // Mark parse step as thinking
    updateStepStatus('parse', 'thinking', 'Analyzing your experiment description...');

    try {
        const parseResp = await fetch(`${API}/nl/parse`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
        });

        if (!parseResp.ok) {
            const err = await parseResp.json().catch(() => ({ detail: 'Parse request failed' }));
            throw new Error(err.detail || `HTTP ${parseResp.status}`);
        }

        const parsed = await parseResp.json();
        state.parsedResult = parsed;
        state.phase = 'parsed';
        state.direction = parsed.direction || 'minimize';

        const extractedCount = parsed.extracted ? parsed.extracted.length : 0;
        updateStepStatus('parse', 'success', `Extracted ${extractedCount} parameters`);

        // Show parsed preview in left panel
        showParsedPreview(parsed);

        // Build and launch campaign
        const orchInput = buildOrchestratorInput(parsed);
        await launchCampaign(orchInput);

    } catch (err) {
        state.phase = 'error';
        updateStepStatus('parse', 'failure', `Failed: ${err.message}`);
        sendBtn.disabled = false;
    }
}

function buildOrchestratorInput(parsed) {
    const rawDims = parsed.dimensions && parsed.dimensions.length > 0
        ? parsed.dimensions
        : [{ name: 'volume', low: 1, high: 50 }];

    const dimensions = rawDims.map((d) => ({
        param_name: d.name || d.param_name || 'unknown',
        param_type: 'number',
        min_value: Math.min(d.low, d.high),
        max_value: Math.max(d.low, d.high),
    }));

    return {
        contract_id: `nlp-${Date.now()}`,
        objective_kpi: parsed.objective_kpi || 'cv',
        direction: parsed.direction || 'minimize',
        max_rounds: parsed.max_rounds || 5,
        batch_size: parsed.batch_size || 4,
        strategy: parsed.strategy || 'lhs',
        target_value: parsed.target_value || null,
        dimensions: dimensions,
        protocol_template: {
            steps: [{ primitive: 'robot.dispense', params: {} }],
            slot_assignments: parsed.slot_assignments || {},
        },
        protocol_pattern_id: parsed.protocol_pattern_id || '',
        policy_snapshot: {},
        dry_run: true,
        plan_only: false,
    };
}

// ---------------------------------------------------------------------------
// Campaign Lifecycle
// ---------------------------------------------------------------------------

async function launchCampaign(input) {
    updateStepStatus('planner', 'thinking', 'Launching campaign...');

    try {
        const resp = await fetch(`${API}/orchestrate/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(input),
        });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: 'Launch failed' }));
            throw new Error(err.detail || `HTTP ${resp.status}`);
        }

        const data = await resp.json();
        state.campaignId = data.campaign_id;
        state.phase = 'running';

        // Show campaign status in left panel
        showCampaignStatus();
        stopBtn.style.display = '';

        // Connect SSE
        connectSSE(data.campaign_id);

    } catch (err) {
        state.phase = 'error';
        updateStepStatus('planner', 'failure', `Launch failed: ${err.message}`);
        sendBtn.disabled = false;
    }
}

async function handleStop() {
    if (!state.campaignId) return;
    try {
        await fetch(`${API}/orchestrate/${state.campaignId}/stop`, { method: 'POST' });
        updateStepStatus('complete', 'warning', 'Campaign stopped by user');
    } catch (err) {
        console.error('Stop failed:', err);
    }
}

// ---------------------------------------------------------------------------
// SSE Connection
// ---------------------------------------------------------------------------

function connectSSE(campaignId) {
    if (state.eventSource) {
        state.eventSource.close();
    }

    const url = `${API}/orchestrate/${campaignId}/events/stream`;
    const es = new EventSource(url);
    state.eventSource = es;

    // Listen for ALL event types (including recovery events and detailed execution events)
    const eventTypes = [
        'campaign_start', 'agent_thinking', 'agent_result',
        'round_start', 'campaign_complete', 'round_complete',
        'strategy_decision', 'stabilize_execution',
        'well_allocator_init', 'well_exhausted',
        'recovery_decision', 'recovery_success', 'recovery_failed',
        'chemical_safety_alert',
        // Detailed execution events
        'agent_decision', 'tool_call', 'hardware_action',
        'protocol_step', 'safety_check', 'thinking', 'log',
    ];

    eventTypes.forEach((type) => {
        es.addEventListener(type, (e) => {
            try {
                const data = JSON.parse(e.data);
                handleSSEEvent(type, data);
            } catch (err) {
                console.warn('SSE parse error:', err);
            }
        });
    });

    // Generic agent_event fallback
    es.addEventListener('agent_event', (e) => {
        try {
            const data = JSON.parse(e.data);
            handleSSEEvent(data.type || 'agent_event', data);
        } catch (err) {
            console.warn('SSE parse error:', err);
        }
    });

    es.onerror = () => {
        console.warn('SSE connection error, falling back to polling');
        if (state.phase === 'running') {
            pollCampaignStatus();
        }
    };

    // BACKUP: Start polling immediately as a safety net
    // This ensures we catch completion even if SSE misses the event
    state.pollTimer = setTimeout(() => {
        if (state.phase === 'running') {
            pollCampaignStatus();
        }
    }, 2000); // Start backup polling after 2 seconds
}

// ---------------------------------------------------------------------------
// SSE Event Router → Pipeline Model
// ---------------------------------------------------------------------------

function handleSSEEvent(type, data) {
    const agent = data.agent || 'system';
    const roundNum = data.round || null;
    const roundId = roundNum ? `round-${roundNum}` : findCurrentRoundId();

    switch (type) {
        case 'campaign_start':
            updateStepStatus('planner', 'thinking', data.message || 'Starting campaign...');
            updateStepData('planner', data);
            break;

        case 'agent_thinking': {
            if (agent === 'planner') {
                updateStepStatus('planner', 'thinking', data.message);
            } else if (roundId) {
                const stepId = mapAgentToStepId(roundId, agent);
                updateStepStatus(stepId, 'thinking', data.message);
            }
            break;
        }

        case 'agent_result': {
            const success = data.success !== false;
            const msg = formatResultMessage(data);

            if (agent === 'planner') {
                updateStepStatus('planner', success ? 'success' : 'failure', msg);
                updateStepData('planner', data);
                if (data.duration_ms) updateStepDuration('planner', data.duration_ms);
            } else if (roundId) {
                const stepId = mapAgentToStepId(roundId, agent);
                updateStepStatus(stepId, success ? 'success' : 'failure', msg);
                updateStepData(stepId, data);
                if (data.duration_ms) updateStepDuration(stepId, data.duration_ms);

                // Track candidate aggregation
                const step = findStep(stepId);
                if (step) {
                    step.candidatesDone++;
                    updateStepCandidates(stepId);
                }
            }

            // Track KPI
            if (data.kpi != null) {
                state.kpiHistory.push({
                    round: roundNum,
                    kpi: data.kpi,
                    params: data.params || {},
                });
                updateBestKpi(data.kpi);
            }
            break;
        }

        case 'round_start':
            state.roundsTotal = data.total_rounds || state.roundsTotal;
            state.roundsDone = (data.round || 1) - 1;
            addRoundToPipeline(data.round, data.total_rounds);
            updateProgress();
            // Re-enable auto-scroll at round start
            state.autoScroll = true;
            break;

        case 'strategy_decision':
            if (roundId) {
                // Dynamically insert a strategy step if not present
                ensureStrategyStep(roundId, data.round);
                updateStepStatus(`${roundId}-strategy`, 'success',
                    `${data.backend} (${data.phase}) — ${data.reason}`);
                updateStepData(`${roundId}-strategy`, data);
            }
            break;

        case 'stabilize_execution':
            if (roundId) {
                updateStepStatus(`${roundId}-design`, 'success',
                    `Stabilize: ${data.total_candidates} runs (${data.n_points} pts × ${data.n_replicates} reps)`);
                updateStepData(`${roundId}-design`, data);
            }
            break;

        case 'well_allocator_init':
            state.wellAllocator = data;
            // Update context panel if showing overview
            if (!state.pipeline.selectedStepId) {
                updateContextPanel();
            }
            break;

        case 'well_exhausted':
            if (roundId) {
                const execStep = findStep(`${roundId}-execute`);
                if (execStep) {
                    execStep.detail += ' [wells exhausted]';
                    updateStepDOM(`${roundId}-execute`);
                }
            }
            break;

        case 'recovery_decision': {
            // Recovery agent made a decision after execution failure
            const roundId = data.round ? `round-${data.round}` : findCurrentRoundId();
            const execStep = state.pipeline.find(s => s.id === `${roundId}-execute`);
            if (execStep) {
                const icon = data.decision === 'retry' ? '🔄' :
                            data.decision === 'abort' ? '⛔' :
                            data.decision === 'skip' ? '⏭️' :
                            data.decision === 'degrade' ? '⚠️' : '🤔';

                const severity = data.error_severity === 'high' ? '🚨' :
                                data.error_severity === 'medium' ? '⚠️' : 'ℹ️';

                let message = `${icon} Recovery: ${data.decision}`;
                if (data.retry_count > 0) {
                    message += ` (attempt ${data.retry_count})`;
                }
                if (data.chemical_safety_event) {
                    message += ' 🛡️ SafetyAgent veto';
                }

                execStep.detail = `${severity} ${data.error_type} → ${message}`;
                execStep.status = data.decision === 'retry' ? 'thinking' :
                                 data.decision === 'abort' ? 'failure' : 'warning';
                updateStepDOM(`${roundId}-execute`);
            }
            break;
        }

        case 'recovery_success': {
            // Execution succeeded after recovery retries
            const roundId = data.round ? `round-${data.round}` : findCurrentRoundId();
            const execStep = state.pipeline.find(s => s.id === `${roundId}-execute`);
            if (execStep) {
                execStep.detail = `✅ Success after ${data.retries} ${data.retries === 1 ? 'retry' : 'retries'}`;
                execStep.status = 'success';
                updateStepDOM(`${roundId}-execute`);
            }
            break;
        }

        case 'recovery_failed': {
            // Max retries exceeded
            const roundId = data.round ? `round-${data.round}` : findCurrentRoundId();
            const execStep = state.pipeline.find(s => s.id === `${roundId}-execute`);
            if (execStep) {
                execStep.detail = `❌ Failed after ${data.retries} retries`;
                execStep.status = 'failure';
                updateStepDOM(`${roundId}-execute`);
            }
            break;
        }

        case 'chemical_safety_alert': {
            // Chemical safety event detected - critical alert
            const roundId = data.round ? `round-${data.round}` : findCurrentRoundId();
            const execStep = state.pipeline.find(s => s.id === `${roundId}-execute`);
            if (execStep) {
                execStep.detail = `🚨 CHEMICAL SAFETY EVENT: ${data.error_type}`;
                execStep.status = 'failure';
                execStep.message = data.message || 'SafetyAgent veto active';
                updateStepDOM(`${roundId}-execute`);
            }
            // Show critical alert in status panel
            updateCampaignStatus({
                status: 'Chemical safety alert',
                message: `🚨 ${data.error_type}`,
                safety_alert: true,
            });
            break;
        }

        case 'round_complete': {
            // Round completed with KPI result
            if (roundId) {
                const roundStep = findStep(roundId);
                if (roundStep) {
                    const eta10 = data.eta10 != null ? data.eta10.toFixed(1) : '?';
                    const improvement = data.improvement_pct != null ? data.improvement_pct.toFixed(1) : '0';
                    roundStep.detail = `η10 = ${eta10} mV (${improvement >= 0 ? '+' : ''}${improvement}%)`;
                    roundStep.status = 'success';
                    updateStepDOM(roundId);
                }
            }
            state.roundsDone = data.round || state.roundsDone;
            updateProgress();
            break;
        }

        case 'campaign_complete':
            state.phase = 'completed';
            state.bestKpi = data.best_kpi;
            state.roundsDone = data.rounds_completed || state.roundsDone;
            updateStepStatus('complete', 'success', data.message || 'Campaign completed');
            updateStepData('complete', data);
            updateProgress();
            onCampaignDone();
            // Auto-select complete step to show results
            selectStep('complete');
            break;

        // Detailed execution events
        case 'agent_decision': {
            // Agent made a decision - add as detail step
            if (roundId) {
                const decision = data.decision || 'Decision made';
                const reasoning = data.reasoning || '';
                addDetailStep(roundId, 'decision', data.agent || 'agent',
                    `💡 ${decision}`, reasoning, data.indent || 0);
            }
            break;
        }

        case 'tool_call': {
            // Tool invocation - add as detail step
            if (roundId) {
                const tool = data.tool || 'tool';
                const operation = data.operation || 'operation';
                const params = data.params ? JSON.stringify(data.params) : '';
                addDetailStep(roundId, 'tool', 'executor',
                    `🔧 ${tool}.${operation}`, params, data.indent || 0);
            }
            break;
        }

        case 'hardware_action': {
            // Hardware operation - add as detail step
            if (roundId) {
                const hardware = data.hardware || 'device';
                const action = data.action || 'action';
                const details = data.details ? JSON.stringify(data.details) : '';
                addDetailStep(roundId, 'hardware', 'executor',
                    `⚙️ ${hardware}: ${action}`, details, data.indent || 0);
            }
            break;
        }

        case 'protocol_step': {
            // Protocol execution step - add as detail step
            if (roundId) {
                const step_num = data.step_num || '?';
                const description = data.description || 'Protocol step';
                addDetailStep(roundId, 'protocol', 'executor',
                    `Step ${step_num}: ${description}`, '', data.indent || 0);
            }
            break;
        }

        case 'safety_check': {
            // Safety validation - add as detail step
            if (roundId) {
                const check_name = data.check_name || 'Safety check';
                const passed = data.passed !== false;
                const icon = passed ? '✅' : '❌';
                const details = data.details || '';
                addDetailStep(roundId, 'safety', 'safety',
                    `${icon} ${check_name}`, details, data.indent || 0);
            }
            break;
        }

        case 'thinking': {
            // Agent thinking process - add as detail step
            if (roundId) {
                const message = data.message || 'Analyzing...';
                addDetailStep(roundId, 'thinking', 'system',
                    `🤔 ${message}`, '', data.indent || 0);
            }
            break;
        }

        case 'log': {
            // General log message - add as detail step
            if (roundId) {
                const level = data.level || 'info';
                const message = data.message || 'Log entry';
                const icon = level === 'error' ? '❌' :
                            level === 'warning' ? '⚠️' :
                            level === 'success' ? '✅' : 'ℹ️';
                addDetailStep(roundId, 'log', 'system',
                    `${icon} ${message}`, '', data.indent || 0);
            }
            break;
        }

        default:
            console.log('Unhandled SSE event:', type, data);
    }
}

function formatResultMessage(data) {
    let msg = data.message || '';
    if (data.kpi != null) {
        msg += ` — KPI: ${data.kpi.toFixed(4)}`;
    }
    if (data.n_candidates != null) {
        msg += ` (${data.n_candidates} candidates)`;
    }
    return msg;
}

function agentLabel(agent) {
    const labels = {
        planner: 'Planner',
        design: 'Design Agent',
        compiler: 'Compiler',
        safety: 'Safety Check',
        executor: 'Executor',
        sensing: 'QC / Sensing',
        stop: 'Stop Agent',
        parse: 'NL Parser',
        system: 'System',
        strategy: 'Strategy',
        recovery: 'Recovery Agent',
    };
    return labels[agent] || agent;
}

// ---------------------------------------------------------------------------
// Pipeline Initialization & Mutation
// ---------------------------------------------------------------------------

function addDetailStep(roundId, type, agent, label, detail, indent) {
    /**
     * Add a detail step as a child under the current round
     * Used for detailed execution tree (tool calls, hardware actions, etc.)
     */
    const roundIdx = state.pipeline.steps.findIndex(s => s.id === roundId);
    if (roundIdx === -1) return;

    // Find the last child of this round to insert after it
    let insertIdx = roundIdx + 1;
    while (insertIdx < state.pipeline.steps.length) {
        const step = state.pipeline.steps[insertIdx];
        if (step.isRound || step.id === 'complete') break;
        if (step.isChild || step.isDetail) insertIdx++;
        else break;
    }

    // Create detail step with unique ID
    const detailId = `${roundId}-detail-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const detailStep = createStep(detailId, type, agent, label, { round: roundId.split('-')[1], isChild: true });
    detailStep.isDetail = true;
    detailStep.detail = detail;
    detailStep.indent = indent;
    detailStep.status = 'success';
    detailStep.timestamp = new Date();

    // Insert into pipeline
    state.pipeline.steps.splice(insertIdx, 0, detailStep);

    // Re-render pipeline to show new step
    renderPipeline();
}

function initPipeline() {
    state.pipeline.steps = [
        createStep('parse', 'parse', 'parse', 'NL Parser'),
        createStep('planner', 'planner', 'planner', 'Campaign Planner'),
        createStep('complete', 'complete', 'system', 'Campaign Complete'),
    ];
    state.pipeline.activeStepId = null;
    state.pipeline.selectedStepId = null;
    state.kpiHistory = [];
    state.bestKpi = null;
    state.roundsDone = 0;
    state.roundsTotal = 0;
    state.wellAllocator = null;
    state.autoScroll = true;
    renderPipeline();
    updateContextPanel();
}

function addRoundToPipeline(roundNum, totalRounds) {
    const roundId = `round-${roundNum}`;

    // Create round header step
    const roundStep = createStep(roundId, 'round', 'system',
        `Round ${roundNum}${totalRounds ? ' / ' + totalRounds : ''}`,
        { round: roundNum, isRound: true });
    roundStep.status = 'thinking';

    // Create child steps for this round
    // Strategy step is NOT added by default — only inserted when strategy_decision fires
    const children = [
        createStep(`${roundId}-design`, 'design', 'design', 'Design Agent', { round: roundNum, isChild: true }),
        createStep(`${roundId}-compile`, 'compile', 'compiler', 'Compiler', { round: roundNum, isChild: true }),
        createStep(`${roundId}-safety`, 'safety', 'safety', 'Safety Check', { round: roundNum, isChild: true }),
        createStep(`${roundId}-execute`, 'execute', 'executor', 'Executor', { round: roundNum, isChild: true }),
        createStep(`${roundId}-sensing`, 'sensing', 'sensing', 'QC / Sensing', { round: roundNum, isChild: true }),
        createStep(`${roundId}-stop`, 'stop', 'stop', 'Stop Check', { round: roundNum, isChild: true }),
    ];

    // Mark previous round as success if exists
    const prevRoundId = `round-${roundNum - 1}`;
    const prevRound = findStep(prevRoundId);
    if (prevRound && prevRound.status === 'thinking') {
        prevRound.status = 'success';
    }

    // Insert before the 'complete' step
    const completeIdx = state.pipeline.steps.findIndex(s => s.id === 'complete');
    state.pipeline.steps.splice(completeIdx, 0, roundStep, ...children);

    renderPipeline();
}

function ensureStrategyStep(roundId, roundNum) {
    // Check if strategy step already exists
    if (findStep(`${roundId}-strategy`)) return;

    // Insert strategy step right after the round header
    const roundIdx = state.pipeline.steps.findIndex(s => s.id === roundId);
    if (roundIdx === -1) return;

    const strategyStep = createStep(`${roundId}-strategy`, 'strategy', 'strategy',
        'Strategy Selection', { round: roundNum, isChild: true });

    state.pipeline.steps.splice(roundIdx + 1, 0, strategyStep);
    renderPipeline();
}

// ---------------------------------------------------------------------------
// Pipeline DOM Rendering
// ---------------------------------------------------------------------------

function renderPipeline() {
    if (pipelineEmpty) {
        pipelineEmpty.style.display = state.pipeline.steps.length > 0 ? 'none' : '';
    }

    // Remove old step elements
    pipelineContainer.querySelectorAll('.pipeline-step, .pipeline-round, .pipeline-complete-banner').forEach(el => el.remove());

    const frag = document.createDocumentFragment();

    for (const step of state.pipeline.steps) {
        if (step.isRound) {
            frag.appendChild(renderRoundNode(step));
        } else {
            frag.appendChild(renderStepNode(step));
        }
    }

    pipelineContainer.appendChild(frag);
    scrollToActive();
}

function renderStepNode(step) {
    const el = document.createElement('div');
    el.className = 'pipeline-step';
    el.dataset.stepId = step.id;

    if (step.isChild) el.classList.add('child');
    if (step.isDetail && step.indent != null) el.dataset.indent = step.indent;
    if (step.id === state.pipeline.selectedStepId) el.classList.add('selected');
    if (step.status === 'thinking') el.classList.add('active');
    if (step === state.pipeline.steps[state.pipeline.steps.length - 1]) el.classList.add('no-connector');

    const agentColor = `var(--agent-${step.agent}, var(--text-bright))`;

    let durationHtml = '';
    if (step.duration_ms != null) {
        durationHtml = `<span class="step-duration">${step.duration_ms}ms</span>`;
    }

    let candidatesHtml = '';
    if (step.candidatesDone > 0) {
        candidatesHtml = `<span class="step-candidates">${step.candidatesDone}${step.candidatesTotal ? '/' + step.candidatesTotal : ''}</span>`;
    }

    let detailHtml = '';
    if (step.detail) {
        detailHtml = `<div class="step-detail">${escapeHtml(step.detail)}</div>`;
    }

    el.innerHTML = `
        <div class="step-status-icon ${step.status}">
            ${statusIcon(step.status)}
        </div>
        <div class="step-content">
            <div class="step-header">
                <span class="step-name" style="color: ${agentColor}">${escapeHtml(step.label)}</span>
                ${candidatesHtml}
                ${durationHtml}
            </div>
            ${detailHtml}
        </div>
    `;

    el.addEventListener('click', () => selectStep(step.id));
    return el;
}

function renderRoundNode(step) {
    const el = document.createElement('div');
    el.className = 'pipeline-round';
    el.dataset.stepId = step.id;

    const statusDot = step.status === 'thinking'
        ? '<span class="spinner" style="margin-right:6px"></span>'
        : step.status === 'success'
        ? '<span style="color:var(--accent-success);margin-right:4px">&#x2713;</span>'
        : '';

    el.innerHTML = `
        <div class="pipeline-round-header">
            <span class="pipeline-round-title">${statusDot}${escapeHtml(step.label)}</span>
            <span class="pipeline-round-badge">${step.detail || ''}</span>
        </div>
    `;

    el.addEventListener('click', () => selectStep(step.id));
    return el;
}

function statusIcon(status) {
    switch (status) {
        case 'pending':  return '<span style="font-size:10px;color:var(--text-muted)">&#x25CB;</span>';
        case 'thinking': return '<span class="spinner"></span>';
        case 'success':  return '&#x2713;';
        case 'failure':  return '&#x2717;';
        case 'warning':  return '&#x26A0;';
        default:         return '';
    }
}

// ---------------------------------------------------------------------------
// Pipeline Step Updates (incremental DOM updates)
// ---------------------------------------------------------------------------

function updateStepStatus(stepId, status, detail) {
    const step = findStep(stepId);
    if (!step) return;

    step.status = status;
    if (detail != null) step.detail = detail;
    step.timestamp = new Date();

    if (status === 'thinking') {
        state.pipeline.activeStepId = stepId;
    }

    updateStepDOM(stepId);

    // If this step is currently selected, refresh context panel
    if (stepId === state.pipeline.selectedStepId) {
        updateContextPanel();
    }
}

function updateStepData(stepId, data) {
    const step = findStep(stepId);
    if (step) {
        step.data = { ...step.data, ...data };
    }
}

function updateStepDuration(stepId, ms) {
    const step = findStep(stepId);
    if (step) {
        step.duration_ms = ms;
        updateStepDOM(stepId);
    }
}

function updateStepCandidates(stepId) {
    const step = findStep(stepId);
    if (!step) return;
    // Update DOM in-place
    const el = pipelineContainer.querySelector(`[data-step-id="${stepId}"]`);
    if (!el) return;
    let badge = el.querySelector('.step-candidates');
    if (!badge) {
        badge = document.createElement('span');
        badge.className = 'step-candidates';
        const header = el.querySelector('.step-header');
        if (header) header.appendChild(badge);
    }
    badge.textContent = `${step.candidatesDone}${step.candidatesTotal ? '/' + step.candidatesTotal : ''}`;
}

function updateStepDOM(stepId) {
    const step = findStep(stepId);
    if (!step) return;

    const el = pipelineContainer.querySelector(`[data-step-id="${stepId}"]`);
    if (!el) return;

    // For round nodes
    if (step.isRound) {
        const titleEl = el.querySelector('.pipeline-round-title');
        if (titleEl) {
            const statusDot = step.status === 'thinking'
                ? '<span class="spinner" style="margin-right:6px"></span>'
                : step.status === 'success'
                ? '<span style="color:var(--accent-success);margin-right:4px">&#x2713;</span>'
                : '';
            titleEl.innerHTML = `${statusDot}${escapeHtml(step.label)}`;
        }
        return;
    }

    // Update status icon
    const iconEl = el.querySelector('.step-status-icon');
    if (iconEl) {
        iconEl.className = `step-status-icon ${step.status}`;
        iconEl.innerHTML = statusIcon(step.status);
    }

    // Update detail
    let detailEl = el.querySelector('.step-detail');
    if (step.detail) {
        if (!detailEl) {
            detailEl = document.createElement('div');
            detailEl.className = 'step-detail';
            const content = el.querySelector('.step-content');
            if (content) content.appendChild(detailEl);
        }
        detailEl.textContent = step.detail;
    }

    // Update duration
    if (step.duration_ms != null) {
        let durEl = el.querySelector('.step-duration');
        if (!durEl) {
            durEl = document.createElement('span');
            durEl.className = 'step-duration';
            const header = el.querySelector('.step-header');
            if (header) header.appendChild(durEl);
        }
        durEl.textContent = `${step.duration_ms}ms`;
    }

    // Toggle active class
    if (step.status === 'thinking') {
        el.classList.add('active');
    } else {
        el.classList.remove('active');
    }

    scrollToActive();
}

function scrollToActive() {
    if (!state.autoScroll) return;
    const activeEl = pipelineContainer.querySelector('.pipeline-step.active');
    if (activeEl) {
        activeEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}

// ---------------------------------------------------------------------------
// Step Selection & Context Panel
// ---------------------------------------------------------------------------

function selectStep(stepId) {
    state.pipeline.selectedStepId = stepId;

    // Update visual selection
    pipelineContainer.querySelectorAll('.pipeline-step, .pipeline-round').forEach(el => {
        el.classList.toggle('selected', el.dataset.stepId === stepId);
    });

    // On narrow screens, open the context panel
    if (window.innerWidth <= 1200) {
        panelRight.classList.add('open');
    }

    updateContextPanel();
}

function updateContextPanel() {
    const stepId = state.pipeline.selectedStepId;

    if (!stepId) {
        contextBody.innerHTML = renderOverviewContext();
        return;
    }

    const step = findStep(stepId);
    if (!step) {
        contextBody.innerHTML = renderOverviewContext();
        return;
    }

    switch (step.type) {
        case 'strategy':
            contextBody.innerHTML = renderStrategyContext(step);
            break;
        case 'complete':
            contextBody.innerHTML = renderCompleteContext(step);
            break;
        case 'round':
            contextBody.innerHTML = renderRoundContext(step);
            break;
        default:
            contextBody.innerHTML = renderGenericContext(step);
    }

    // Attach raw data toggle handler
    const toggle = contextBody.querySelector('.raw-data-toggle');
    const content = contextBody.querySelector('.raw-data-content');
    if (toggle && content) {
        toggle.addEventListener('click', () => {
            content.classList.toggle('open');
            toggle.textContent = content.classList.contains('open') ? '▾ Hide raw data' : '▸ Show raw data';
        });
    }
}

// ---------------------------------------------------------------------------
// Context Panel Renderers
// ---------------------------------------------------------------------------

function renderOverviewContext() {
    let html = '<div class="context-section">';
    html += '<div class="context-section-title">Campaign Overview</div>';
    html += '<div class="diagnostics-grid">';
    html += diagItem('Status', state.phase);
    html += diagItem('Campaign', state.campaignId ? state.campaignId.slice(-8) : '—');
    html += diagItem('Rounds', `${state.roundsDone}/${state.roundsTotal || '?'}`);
    html += diagItem('Best KPI', state.bestKpi != null ? state.bestKpi.toFixed(4) : '—');
    html += diagItem('Direction', state.direction);
    html += diagItem('KPI Points', String(state.kpiHistory.length));
    html += '</div></div>';

    if (state.kpiHistory.length > 0) {
        html += '<div class="context-section">';
        html += '<div class="context-section-title">KPI Trend</div>';
        html += renderKpiTrend();
        html += '</div>';
    }

    if (state.wellAllocator) {
        html += '<div class="context-section">';
        html += '<div class="context-section-title">Well Allocator</div>';
        html += '<div class="diagnostics-grid">';
        html += diagItem('Slots', String(state.wellAllocator.n_slots || '—'));
        html += diagItem('Strategy', state.wellAllocator.strategy || '—');
        html += '</div></div>';
    }

    html += '<p class="text-muted" style="margin-top:var(--gap-md);text-align:center;">Click a pipeline step for details.</p>';
    return html;
}

function renderStrategyContext(step) {
    const d = step.data;
    let html = '';

    // Header
    html += `<div class="context-section">`;
    html += `<div class="context-section-title">Strategy Decision — Round ${step.round || '?'}</div>`;
    html += `<div class="diagnostics-grid">`;
    html += diagItem('Backend', d.backend || '—');
    html += diagItem('Phase', d.phase || '—');
    html += diagItem('Confidence', d.confidence != null ? `${(d.confidence * 100).toFixed(0)}%` : '—');
    html += diagItem('Drift Score', d.drift_score != null ? d.drift_score.toFixed(3) : '—');
    html += `</div></div>`;

    // Reason
    if (d.reason) {
        html += `<div class="context-section">`;
        html += `<div class="context-section-title">Reason</div>`;
        html += `<p style="font-size:12px;color:var(--text-secondary)">${escapeHtml(d.reason)}</p>`;
        html += `</div>`;
    }

    // Phase posterior bars
    if (d.phase_posterior && typeof d.phase_posterior === 'object') {
        html += `<div class="context-section">`;
        html += `<div class="context-section-title">Phase Posterior</div>`;
        html += `<div class="phase-bars">`;
        const phases = ['explore', 'exploit', 'refine', 'stabilize'];
        for (const p of phases) {
            const val = d.phase_posterior[p] || 0;
            const pct = (val * 100).toFixed(0);
            html += `<div class="phase-bar-row">
                <span class="phase-bar-label">${p}</span>
                <div class="phase-bar-track"><div class="phase-bar-fill ${p}" style="width:${pct}%"></div></div>
                <span class="phase-bar-value">${pct}%</span>
            </div>`;
        }
        html += `</div></div>`;
    }

    // Diagnostics
    if (d.diagnostics && typeof d.diagnostics === 'object') {
        html += `<div class="context-section">`;
        html += `<div class="context-section-title">Diagnostics</div>`;
        html += `<div class="diagnostics-grid">`;
        for (const [k, v] of Object.entries(d.diagnostics)) {
            html += diagItem(k, typeof v === 'number' ? v.toFixed(3) : String(v));
        }
        html += `</div></div>`;
    }

    // Explanation
    if (d.explanation) {
        html += `<div class="context-section">`;
        html += `<div class="context-section-title">Explanation</div>`;
        html += `<p style="font-size:12px;color:var(--text-secondary)">${escapeHtml(d.explanation)}</p>`;
        html += `</div>`;
    }

    html += renderRawData(d);
    return html;
}

function renderCompleteContext(step) {
    const d = step.data;
    let html = '';

    html += `<div class="context-section">`;
    html += `<div class="context-section-title">Campaign Results</div>`;
    html += `<div class="diagnostics-grid">`;
    html += diagItem('Rounds', String(d.rounds_completed || '—'));
    html += diagItem('Best KPI', d.best_kpi != null ? d.best_kpi.toFixed(4) : '—');
    html += diagItem('Stop Reason', d.stop_reason || '—');
    html += diagItem('Status', d.status || 'completed');
    html += `</div></div>`;

    // KPI trend
    if (state.kpiHistory.length > 0) {
        html += `<div class="context-section">`;
        html += `<div class="context-section-title">KPI Trend</div>`;
        html += renderKpiTrend();
        html += `</div>`;
    }

    // Top-K recipes
    if (d.top_k_recipes && d.top_k_recipes.length > 0) {
        html += `<div class="context-section">`;
        html += `<div class="context-section-title">Top Recipes</div>`;
        html += `<table class="recipes-table"><thead><tr>`;
        html += `<th>#</th><th>KPI</th><th>Params</th><th>Round</th>`;
        html += `</tr></thead><tbody>`;
        d.top_k_recipes.forEach((r, i) => {
            const params = r.params ? Object.entries(r.params).map(([k, v]) =>
                `${k}=${typeof v === 'number' ? v.toFixed(2) : v}`).join(', ') : '—';
            html += `<tr>
                <td>${i + 1}</td>
                <td>${r.kpi != null ? r.kpi.toFixed(4) : '—'}</td>
                <td>${escapeHtml(params)}</td>
                <td>${r.round || '—'}</td>
            </tr>`;
        });
        html += `</tbody></table></div>`;
    }

    html += renderRawData(d);
    return html;
}

function renderRoundContext(step) {
    let html = '';

    html += `<div class="context-section">`;
    html += `<div class="context-section-title">${escapeHtml(step.label)}</div>`;
    html += `<div class="diagnostics-grid">`;
    html += diagItem('Status', step.status);
    html += diagItem('Round', String(step.round || '?'));
    html += `</div></div>`;

    // Show KPI values for this round
    const roundKpis = state.kpiHistory.filter(k => k.round === step.round);
    if (roundKpis.length > 0) {
        html += `<div class="context-section">`;
        html += `<div class="context-section-title">Round KPIs (${roundKpis.length} points)</div>`;
        html += `<div class="diagnostics-grid">`;
        roundKpis.forEach((k, i) => {
            html += diagItem(`#${i + 1}`, k.kpi.toFixed(4));
        });
        html += `</div></div>`;
    }

    return html;
}

function renderGenericContext(step) {
    let html = '';

    html += `<div class="context-section">`;
    html += `<div class="context-section-title">${escapeHtml(step.label)}</div>`;
    html += `<div class="diagnostics-grid">`;
    html += diagItem('Agent', agentLabel(step.agent));
    html += diagItem('Status', step.status);
    if (step.duration_ms) html += diagItem('Duration', `${step.duration_ms}ms`);
    if (step.round) html += diagItem('Round', String(step.round));
    if (step.candidatesDone > 0) html += diagItem('Candidates', `${step.candidatesDone}`);
    html += `</div></div>`;

    if (step.detail) {
        html += `<div class="context-section">`;
        html += `<div class="context-section-title">Detail</div>`;
        html += `<p style="font-size:12px;color:var(--text-secondary);line-height:1.5">${escapeHtml(step.detail)}</p>`;
        html += `</div>`;
    }

    // Show agent-result-specific data
    const d = step.data;
    if (d.kpi != null) {
        html += `<div class="context-section">`;
        html += `<div class="context-section-title">KPI</div>`;
        html += `<div class="diagnostics-grid">`;
        html += diagItem('Value', d.kpi.toFixed(4));
        html += diagItem('Best', state.bestKpi != null ? state.bestKpi.toFixed(4) : '—');
        html += `</div></div>`;
    }

    html += renderRawData(d);
    return html;
}

// ---------------------------------------------------------------------------
// Context Helpers
// ---------------------------------------------------------------------------

function diagItem(label, value) {
    return `<div class="diagnostic-item">
        <div class="label">${escapeHtml(label)}</div>
        <div class="value">${escapeHtml(value)}</div>
    </div>`;
}

function renderKpiTrend() {
    if (state.kpiHistory.length === 0) return '';

    const values = state.kpiHistory.map(k => k.kpi);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;

    // For minimize direction, lower = better, so we invert the bar height
    const isMinimize = state.direction === 'minimize';

    let html = '<div class="kpi-trend">';
    values.forEach((v, i) => {
        let pct;
        if (isMinimize) {
            // Lower is better → taller bar for lower values
            pct = ((max - v) / range) * 100;
        } else {
            pct = ((v - min) / range) * 100;
        }
        pct = Math.max(pct, 5); // min visible height

        const isBest = (isMinimize && v === Math.min(...values)) ||
                       (!isMinimize && v === Math.max(...values));
        const isLast = i === values.length - 1;

        html += `<div class="kpi-trend-bar${isBest ? ' best' : ''}${isLast ? ' current' : ''}" ` +
                `style="height:${pct}%" title="${v.toFixed(4)}"></div>`;
    });
    html += '</div>';
    return html;
}

function renderRawData(data) {
    if (!data || Object.keys(data).length === 0) return '';
    let html = '<div class="context-section" style="margin-top:var(--gap-md)">';
    html += '<div class="raw-data-toggle">&#x25B8; Show raw data</div>';
    html += '<div class="raw-data-content">';
    html += `<pre>${escapeHtml(JSON.stringify(data, null, 2))}</pre>`;
    html += '</div></div>';
    return html;
}

// ---------------------------------------------------------------------------
// Best KPI Tracking
// ---------------------------------------------------------------------------

function updateBestKpi(kpi) {
    if (state.bestKpi === null) {
        state.bestKpi = kpi;
    } else if (state.direction === 'minimize') {
        state.bestKpi = Math.min(state.bestKpi, kpi);
    } else {
        state.bestKpi = Math.max(state.bestKpi, kpi);
    }
}

// ---------------------------------------------------------------------------
// Poll fallback (when SSE drops)
// ---------------------------------------------------------------------------

async function pollCampaignStatus() {
    if (!state.campaignId || state.phase !== 'running') return;

    try {
        const resp = await fetch(`${API}/orchestrate/${state.campaignId}/status`);
        if (resp.ok) {
            const data = await resp.json();

            // Check if campaign is finished
            if (data.status === 'completed' || data.status === 'failed') {
                state.phase = data.status;

                // Extract detailed results
                const result = data.result || {};
                const bestKpi = result.best_kpi ?? data.best_kpi ?? 'N/A';
                const roundsCompleted = result.rounds_completed ?? 0;
                const stopReason = result.stop_reason ?? 'unknown';

                // Update UI with detailed message
                let message = `Campaign ${data.status}`;
                if (result.rounds_completed) {
                    message += ` • ${roundsCompleted} rounds`;
                }
                message += ` • Best KPI: ${bestKpi}`;
                if (stopReason && stopReason !== 'unknown') {
                    message += ` • ${stopReason.replace('stop_', '').replace('_', ' ')}`;
                }

                updateStepStatus('complete',
                    data.status === 'completed' ? 'success' : 'failure',
                    message);

                // Store result data for context panel
                if (result.top_k_recipes) {
                    updateStepData('complete', result);
                }

                state.bestKpi = bestKpi;
                state.roundsDone = roundsCompleted;
                updateProgress();

                onCampaignDone();

                // Auto-select complete step to show results
                selectStep('complete');

                console.log('Campaign completed via polling:', data);
            } else {
                // Still running, poll again
                state.pollTimer = setTimeout(pollCampaignStatus, 2000); // Poll every 2 seconds
            }
        } else {
            // Retry on error
            state.pollTimer = setTimeout(pollCampaignStatus, 3000);
        }
    } catch (err) {
        console.warn('Poll error:', err);
        // Retry on exception
        state.pollTimer = setTimeout(pollCampaignStatus, 5000);
    }
}

function onCampaignDone() {
    // Close SSE connection
    if (state.eventSource) {
        state.eventSource.close();
        state.eventSource = null;
    }

    // Clear polling timer
    if (state.pollTimer) {
        clearTimeout(state.pollTimer);
        state.pollTimer = null;
    }

    stopBtn.style.display = 'none';
    sendBtn.disabled = false;

    campaignBadge.textContent = state.phase;
    campaignBadge.className = `status-badge ${state.phase}`;
    progressBar.style.width = '100%';

    console.log('Campaign finished, all connections closed');
}

// ---------------------------------------------------------------------------
// Left Panel UI Rendering (kept from v1)
// ---------------------------------------------------------------------------

function showParsedPreview(parsed) {
    welcomeSection.style.display = 'none';
    parsedPreview.style.display = '';

    parsedGrid.innerHTML = '';

    const fields = [
        ['KPI', parsed.objective_kpi],
        ['Direction', parsed.direction],
        ['Max Rounds', parsed.max_rounds],
        ['Target', parsed.target_value],
        ['Batch Size', parsed.batch_size],
        ['Strategy', parsed.strategy],
        ['Protocol', parsed.protocol_pattern_id],
    ];

    fields.forEach(([label, value]) => {
        if (value != null) {
            const item = document.createElement('div');
            item.className = 'parsed-item';
            item.innerHTML = `
                <div class="label">${escapeHtml(label)}</div>
                <div class="value">${escapeHtml(String(value))}</div>
            `;
            parsedGrid.appendChild(item);
        }
    });

    // Dimensions
    if (parsed.dimensions && parsed.dimensions.length > 0) {
        parsed.dimensions.forEach((dim) => {
            const item = document.createElement('div');
            item.className = 'parsed-item wide';
            item.innerHTML = `
                <div class="label">Dim: ${escapeHtml(dim.name)}</div>
                <div class="value">[${dim.low} — ${dim.high}]${dim.unit ? ' ' + escapeHtml(dim.unit) : ''}</div>
            `;
            parsedGrid.appendChild(item);
        });
    }

    // Slots
    if (parsed.slot_assignments && Object.keys(parsed.slot_assignments).length > 0) {
        Object.entries(parsed.slot_assignments).forEach(([slot, labware]) => {
            const item = document.createElement('div');
            item.className = 'parsed-item';
            item.innerHTML = `
                <div class="label">${escapeHtml(slot)}</div>
                <div class="value">${escapeHtml(labware)}</div>
            `;
            parsedGrid.appendChild(item);
        });
    }

    // Detected instruments
    if (parsed.detected_instruments && parsed.detected_instruments.length > 0) {
        const item = document.createElement('div');
        item.className = 'parsed-item wide';
        item.innerHTML = `
            <div class="label">Instruments</div>
            <div class="value">${escapeHtml(parsed.detected_instruments.join(', '))}</div>
        `;
        parsedGrid.appendChild(item);
    }

    // Unknown instruments (onboarding suggestion)
    if (parsed.onboarding_suggested && parsed.unknown_instruments && parsed.unknown_instruments.length > 0) {
        const item = document.createElement('div');
        item.className = 'parsed-item wide';
        item.style.borderColor = 'var(--accent-warning)';
        item.innerHTML = `
            <div class="label" style="color:var(--accent-warning)">Unknown Instruments</div>
            <div class="value" style="color:var(--accent-warning)">${escapeHtml(parsed.unknown_instruments.join(', '))} — onboarding suggested</div>
        `;
        parsedGrid.appendChild(item);
    }
}

function showCampaignStatus() {
    campaignStatus.style.display = '';
    campaignBadge.textContent = 'running';
    campaignBadge.className = 'status-badge running';
    updateProgress();
}

function updateProgress() {
    const pct = state.roundsTotal > 0
        ? Math.round((state.roundsDone / state.roundsTotal) * 100)
        : 0;
    progressBar.style.width = pct + '%';

    statusStats.innerHTML = `
        <div class="stat-item">
            <div class="stat-value">${state.roundsDone}/${state.roundsTotal || '?'}</div>
            <div class="stat-label">Rounds</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">${state.bestKpi != null ? state.bestKpi.toFixed(4) : '—'}</div>
            <div class="stat-label">Best KPI</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">${state.campaignId ? state.campaignId.slice(-8) : '—'}</div>
            <div class="stat-label">Campaign</div>
        </div>
    `;
}

// ---------------------------------------------------------------------------
// Utils
// ---------------------------------------------------------------------------

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
