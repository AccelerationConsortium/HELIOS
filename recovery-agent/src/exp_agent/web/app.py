"""
Exp-Agent Web Dashboard

A modern web-based dashboard for monitoring the recovery agent.
Uses FastAPI + WebSocket for real-time updates.

Usage:
    python -m exp_agent.web.app
    # Then open http://localhost:8000
"""
import asyncio
import json
import time
import uuid
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# ============================================================================
# State Management
# ============================================================================

@dataclass
class DeviceState:
    name: str
    device_type: str
    status: str = "idle"
    telemetry: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class PipelineState:
    stage: int = 0
    stage_name: str = "IDLE"
    device: Optional[str] = None
    decision: Optional[str] = None


@dataclass
class DashboardState:
    devices: Dict[str, DeviceState] = field(default_factory=dict)
    pipeline: PipelineState = field(default_factory=PipelineState)
    events: List[Dict] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=lambda: {
        "abort": 0, "degrade": 0, "retry": 0, "skip": 0, "total": 0
    })

    def to_dict(self):
        return {
            "devices": {k: asdict(v) for k, v in self.devices.items()},
            "pipeline": asdict(self.pipeline),
            "events": self.events[-20:],  # Last 20 events
            "stats": self.stats,
            "timestamp": datetime.now().isoformat()
        }


# Global state
state = DashboardState()
state.devices = {
    "heater_1": DeviceState("heater_1", "Heater", telemetry={"temperature": 25.0, "target": 25.0}),
    "pump_1": DeviceState("pump_1", "Pump", telemetry={"flow_rate": 0.0, "pressure": 1.0}),
    "stage_1": DeviceState("stage_1", "Positioner", telemetry={"x": 0.0, "y": 0.0, "z": 0.0}),
    "spec_1": DeviceState("spec_1", "Spectrometer", telemetry={"signal": 0, "integration_ms": 100}),
}

# Connected WebSocket clients
clients: List[WebSocket] = []


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(title="Exp-Agent Dashboard")


# ============================================================================
# HTML Template (Single-file for simplicity)
# ============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Exp-Agent Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { font-family: 'JetBrains Mono', monospace; }

        /* Color palette from user's design */
        :root {
            --pink: #E8A4C4;
            --gold: #E4B44C;
            --orange: #E8924C;
            --brown: #8B5A3C;
            --purple: #7B5AA8;
            --blue: #7B8DC4;
            --magenta: #C45A8C;
            --hotpink: #E45A8C;
            --navy: #1A2460;
            --amber: #E8A44C;
            --slate: #7B84A8;
            --lavender: #9CA4C8;
        }

        @keyframes pulse-ring {
            0% { transform: scale(0.8); opacity: 1; }
            100% { transform: scale(1.4); opacity: 0; }
        }
        .pulse-ring::before {
            content: '';
            position: absolute;
            inset: -4px;
            border-radius: 50%;
            border: 2px solid currentColor;
            animation: pulse-ring 1.5s ease-out infinite;
        }
        .stage-active {
            background: linear-gradient(135deg, var(--purple) 0%, var(--magenta) 100%);
            box-shadow: 0 0 20px rgba(123, 90, 168, 0.5);
        }
        .stage-done { background: linear-gradient(135deg, var(--slate) 0%, var(--lavender) 100%); }
        .stage-pending { background: linear-gradient(135deg, #f5f5f5 0%, #e8e8e8 100%); color: #666; }
        .card-glow:hover { box-shadow: 0 0 30px rgba(123, 141, 196, 0.2); }
        .status-idle { color: var(--slate); }
        .status-running { color: var(--purple); }
        .status-error { color: var(--hotpink); }
        .decision-abort { background: linear-gradient(135deg, var(--hotpink) 0%, var(--magenta) 100%); }
        .decision-degrade { background: linear-gradient(135deg, var(--purple) 0%, var(--blue) 100%); }
        .decision-retry { background: linear-gradient(135deg, var(--gold) 0%, var(--orange) 100%); }
        .decision-skip { background: linear-gradient(135deg, var(--blue) 0%, var(--lavender) 100%); }

        /* Device card gradients */
        .device-heater { background: linear-gradient(135deg, var(--orange) 0%, var(--gold) 100%); }
        .device-pump { background: linear-gradient(135deg, var(--blue) 0%, var(--lavender) 100%); }
        .device-positioner { background: linear-gradient(135deg, var(--purple) 0%, var(--magenta) 100%); }
        .device-spectrometer { background: linear-gradient(135deg, var(--pink) 0%, var(--magenta) 100%); }

        /* Button styles */
        .btn-heater { background: linear-gradient(135deg, var(--orange) 0%, var(--brown) 100%); }
        .btn-heater:hover { background: linear-gradient(135deg, var(--gold) 0%, var(--orange) 100%); }
        .btn-pump { background: linear-gradient(135deg, var(--blue) 0%, var(--navy) 100%); }
        .btn-pump:hover { background: linear-gradient(135deg, var(--lavender) 0%, var(--blue) 100%); }
        .btn-positioner { background: linear-gradient(135deg, var(--purple) 0%, var(--navy) 100%); }
        .btn-positioner:hover { background: linear-gradient(135deg, var(--lavender) 0%, var(--purple) 100%); }
        .btn-spectrometer { background: linear-gradient(135deg, var(--magenta) 0%, var(--purple) 100%); }
        .btn-spectrometer:hover { background: linear-gradient(135deg, var(--pink) 0%, var(--magenta) 100%); }

        /* Progress bars */
        .bar-heater { background: linear-gradient(90deg, var(--gold) 0%, var(--orange) 50%, var(--hotpink) 100%); }
        .bar-spectrometer { background: linear-gradient(90deg, var(--purple) 0%, var(--magenta) 100%); }
    </style>
</head>
<body class="bg-white text-gray-800 min-h-screen">
    <!-- Header -->
    <header class="bg-white/90 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
            <div class="flex items-center gap-3">
                <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center text-white" style="background: linear-gradient(135deg, var(--purple) 0%, var(--magenta) 100%);">
                    <i data-lucide="flask-conical" class="w-5 h-5"></i>
                </div>
                <div>
                    <h1 class="text-xl font-bold text-gray-800">Exp-Agent</h1>
                    <p class="text-xs text-gray-500">Recovery Dashboard</p>
                </div>
            </div>
            <div class="flex items-center gap-4">
                <div id="connection-status" class="flex items-center gap-2 text-sm">
                    <span class="w-2 h-2 rounded-full" style="background: var(--purple);"></span>
                    <span class="text-gray-500">Connected</span>
                </div>
                <span id="uptime" class="text-sm text-gray-400">00:00:00</span>
            </div>
        </div>
    </header>

    <main class="max-w-7xl mx-auto px-4 py-6 space-y-6">
        <!-- Device Cards Grid -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <!-- Heater Card -->
            <div id="card-heater_1" class="bg-white rounded-2xl p-5 border border-gray-100 shadow-sm card-glow transition-all">
                <div class="flex items-center justify-between mb-4">
                    <div class="flex items-center gap-3">
                        <div class="w-10 h-10 rounded-xl device-heater flex items-center justify-center text-white">
                            <i data-lucide="flame" class="w-5 h-5"></i>
                        </div>
                        <div>
                            <h3 class="font-semibold text-gray-800">Heater</h3>
                            <p class="text-xs text-gray-400">heater_1</p>
                        </div>
                    </div>
                    <span id="status-heater_1" class="text-xs font-medium px-2 py-1 rounded-full bg-gray-100 status-idle">IDLE</span>
                </div>
                <div class="space-y-3">
                    <div class="flex justify-between items-center">
                        <span class="text-gray-400 text-sm">Temperature</span>
                        <span id="temp-heater_1" class="text-lg text-gray-700">25.0°C</span>
                    </div>
                    <div class="h-2 bg-gray-100 rounded-full overflow-hidden">
                        <div id="tempbar-heater_1" class="h-full bar-heater transition-all" style="width: 20%"></div>
                    </div>
                    <div id="error-heater_1" class="text-xs hidden" style="color: var(--hotpink);"></div>
                </div>
            </div>

            <!-- Pump Card -->
            <div id="card-pump_1" class="bg-white rounded-2xl p-5 border border-gray-100 shadow-sm card-glow transition-all">
                <div class="flex items-center justify-between mb-4">
                    <div class="flex items-center gap-3">
                        <div class="w-10 h-10 rounded-xl device-pump flex items-center justify-center text-white">
                            <i data-lucide="droplets" class="w-5 h-5"></i>
                        </div>
                        <div>
                            <h3 class="font-semibold text-gray-800">Pump</h3>
                            <p class="text-xs text-gray-400">pump_1</p>
                        </div>
                    </div>
                    <span id="status-pump_1" class="text-xs font-medium px-2 py-1 rounded-full bg-gray-100 status-idle">IDLE</span>
                </div>
                <div class="space-y-3">
                    <div class="flex justify-between items-center">
                        <span class="text-gray-400 text-sm">Flow Rate</span>
                        <span id="flow-pump_1" class="text-lg text-gray-700">0.0 mL/min</span>
                    </div>
                    <div class="flex justify-between items-center">
                        <span class="text-gray-400 text-sm">Pressure</span>
                        <span id="pressure-pump_1" class="text-gray-700">1.0 bar</span>
                    </div>
                    <div id="error-pump_1" class="text-xs hidden" style="color: var(--hotpink);"></div>
                </div>
            </div>

            <!-- Positioner Card -->
            <div id="card-stage_1" class="bg-white rounded-2xl p-5 border border-gray-100 shadow-sm card-glow transition-all">
                <div class="flex items-center justify-between mb-4">
                    <div class="flex items-center gap-3">
                        <div class="w-10 h-10 rounded-xl device-positioner flex items-center justify-center text-white">
                            <i data-lucide="move-3d" class="w-5 h-5"></i>
                        </div>
                        <div>
                            <h3 class="font-semibold text-gray-800">Positioner</h3>
                            <p class="text-xs text-gray-400">stage_1</p>
                        </div>
                    </div>
                    <span id="status-stage_1" class="text-xs font-medium px-2 py-1 rounded-full bg-gray-100 status-idle">IDLE</span>
                </div>
                <div class="space-y-2">
                    <div class="grid grid-cols-3 gap-2 text-center">
                        <div>
                            <span class="text-gray-400 text-xs">X</span>
                            <p id="x-stage_1" class="text-gray-700">0.0</p>
                        </div>
                        <div>
                            <span class="text-gray-400 text-xs">Y</span>
                            <p id="y-stage_1" class="text-gray-700">0.0</p>
                        </div>
                        <div>
                            <span class="text-gray-400 text-xs">Z</span>
                            <p id="z-stage_1" class="text-gray-700">0.0</p>
                        </div>
                    </div>
                    <div id="error-stage_1" class="text-xs hidden" style="color: var(--hotpink);"></div>
                </div>
            </div>

            <!-- Spectrometer Card -->
            <div id="card-spec_1" class="bg-white rounded-2xl p-5 border border-gray-100 shadow-sm card-glow transition-all">
                <div class="flex items-center justify-between mb-4">
                    <div class="flex items-center gap-3">
                        <div class="w-10 h-10 rounded-xl device-spectrometer flex items-center justify-center text-white">
                            <i data-lucide="scan-line" class="w-5 h-5"></i>
                        </div>
                        <div>
                            <h3 class="font-semibold text-gray-800">Spectrometer</h3>
                            <p class="text-xs text-gray-400">spec_1</p>
                        </div>
                    </div>
                    <span id="status-spec_1" class="text-xs font-medium px-2 py-1 rounded-full bg-gray-100 status-idle">IDLE</span>
                </div>
                <div class="space-y-3">
                    <div class="flex justify-between items-center">
                        <span class="text-gray-400 text-sm">Signal</span>
                        <span id="signal-spec_1" class="text-lg text-gray-700">0</span>
                    </div>
                    <div class="h-2 bg-gray-100 rounded-full overflow-hidden">
                        <div id="signalbar-spec_1" class="h-full bar-spectrometer transition-all" style="width: 0%"></div>
                    </div>
                    <div id="error-spec_1" class="text-xs hidden" style="color: var(--hotpink);"></div>
                </div>
            </div>
        </div>

        <!-- Pipeline & Stats Row -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <!-- Recovery Pipeline -->
            <div class="lg:col-span-2 bg-white rounded-2xl p-5 border border-gray-100 shadow-sm">
                <h3 class="font-semibold mb-4 flex items-center gap-2 text-gray-800">
                    <i data-lucide="git-branch" class="w-4 h-4" style="color: var(--blue);"></i>
                    Recovery Pipeline
                </h3>
                <div class="flex items-center justify-between gap-2 mb-4">
                    <div id="stage-0" class="stage-active flex-1 h-12 rounded-xl flex items-center justify-center text-xs font-medium transition-all text-white">IDLE</div>
                    <div id="stage-1" class="stage-pending flex-1 h-12 rounded-xl flex items-center justify-center text-xs font-medium transition-all">SENSE</div>
                    <div id="stage-2" class="stage-pending flex-1 h-12 rounded-xl flex items-center justify-center text-xs font-medium transition-all">CLASSIFY</div>
                    <div id="stage-3" class="stage-pending flex-1 h-12 rounded-xl flex items-center justify-center text-xs font-medium transition-all">ANALYZE</div>
                    <div id="stage-4" class="stage-pending flex-1 h-12 rounded-xl flex items-center justify-center text-xs font-medium transition-all">DECIDE</div>
                    <div id="stage-5" class="stage-pending flex-1 h-12 rounded-xl flex items-center justify-center text-xs font-medium transition-all">EXECUTE</div>
                    <div id="stage-6" class="stage-pending flex-1 h-12 rounded-xl flex items-center justify-center text-xs font-medium transition-all">VERIFY</div>
                    <div id="stage-7" class="stage-pending flex-1 h-12 rounded-xl flex items-center justify-center text-xs font-medium transition-all">MEMORY</div>
                </div>
                <div class="flex items-center gap-4">
                    <div id="decision-badge" class="hidden px-4 py-2 rounded-xl text-sm font-bold text-white"></div>
                    <span id="pipeline-device" class="text-sm text-gray-400"></span>
                </div>
            </div>

            <!-- Stats -->
            <div class="bg-white rounded-2xl p-5 border border-gray-100 shadow-sm">
                <h3 class="font-semibold mb-4 flex items-center gap-2 text-gray-800">
                    <i data-lucide="bar-chart-3" class="w-4 h-4" style="color: var(--gold);"></i>
                    Decision Statistics
                </h3>
                <div class="grid grid-cols-2 gap-3">
                    <div class="rounded-xl p-3 text-center" style="background: rgba(228, 90, 140, 0.1);">
                        <p class="text-2xl font-bold" style="color: var(--hotpink);" id="stat-abort">0</p>
                        <p class="text-xs text-gray-400">ABORT</p>
                    </div>
                    <div class="rounded-xl p-3 text-center" style="background: rgba(123, 90, 168, 0.1);">
                        <p class="text-2xl font-bold" style="color: var(--purple);" id="stat-degrade">0</p>
                        <p class="text-xs text-gray-400">DEGRADE</p>
                    </div>
                    <div class="rounded-xl p-3 text-center" style="background: rgba(228, 180, 76, 0.1);">
                        <p class="text-2xl font-bold" style="color: var(--gold);" id="stat-retry">0</p>
                        <p class="text-xs text-gray-400">RETRY</p>
                    </div>
                    <div class="rounded-xl p-3 text-center" style="background: rgba(123, 141, 196, 0.1);">
                        <p class="text-2xl font-bold" style="color: var(--blue);" id="stat-skip">0</p>
                        <p class="text-xs text-gray-400">SKIP</p>
                    </div>
                </div>
                <div class="mt-4 text-center">
                    <p class="text-3xl font-bold text-gray-800" id="stat-total">0</p>
                    <p class="text-xs text-gray-400">Total Decisions</p>
                </div>
            </div>
        </div>

        <!-- Controls & Event Log -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <!-- Controls -->
            <div class="bg-white rounded-2xl p-5 border border-gray-100 shadow-sm">
                <h3 class="font-semibold mb-4 flex items-center gap-2 text-gray-800">
                    <i data-lucide="zap" class="w-4 h-4" style="color: var(--orange);"></i>
                    Fault Injection
                </h3>
                <div class="grid grid-cols-2 gap-2">
                    <button onclick="injectFault('heater')" class="btn-heater px-4 py-3 rounded-xl text-sm font-medium transition-all flex items-center justify-center gap-2 text-white">
                        <i data-lucide="flame" class="w-4 h-4"></i> Heater
                    </button>
                    <button onclick="injectFault('pump')" class="btn-pump px-4 py-3 rounded-xl text-sm font-medium transition-all flex items-center justify-center gap-2 text-white">
                        <i data-lucide="droplets" class="w-4 h-4"></i> Pump
                    </button>
                    <button onclick="injectFault('positioner')" class="btn-positioner px-4 py-3 rounded-xl text-sm font-medium transition-all flex items-center justify-center gap-2 text-white">
                        <i data-lucide="move-3d" class="w-4 h-4"></i> Positioner
                    </button>
                    <button onclick="injectFault('spectrometer')" class="btn-spectrometer px-4 py-3 rounded-xl text-sm font-medium transition-all flex items-center justify-center gap-2 text-white">
                        <i data-lucide="scan-line" class="w-4 h-4"></i> Spectrometer
                    </button>
                </div>
                <button onclick="runAllScenarios()" class="mt-3 w-full px-4 py-3 rounded-xl text-sm font-medium transition-all flex items-center justify-center gap-2 text-white" style="background: linear-gradient(135deg, var(--navy) 0%, var(--slate) 100%);">
                    <i data-lucide="play" class="w-4 h-4"></i> Run All Scenarios
                </button>
                <button onclick="resetDashboard()" class="mt-2 w-full bg-gray-100 hover:bg-gray-200 px-4 py-2 rounded-xl text-sm text-gray-500 transition-all">
                    Reset
                </button>
            </div>

            <!-- Event Log -->
            <div class="lg:col-span-2 bg-white rounded-2xl p-5 border border-gray-100 shadow-sm">
                <h3 class="font-semibold mb-4 flex items-center gap-2 text-gray-800">
                    <i data-lucide="scroll-text" class="w-4 h-4" style="color: var(--purple);"></i>
                    Event Log
                </h3>
                <div id="event-log" class="h-48 overflow-y-auto space-y-1 text-xs bg-gray-50 rounded-xl p-3">
                    <div class="text-gray-400">Waiting for events...</div>
                </div>
            </div>
        </div>
    </main>

    <script>
        // Initialize Lucide icons
        lucide.createIcons();

        // WebSocket connection
        let ws;
        let startTime = Date.now();

        function connect() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);

            ws.onopen = () => {
                document.getElementById('connection-status').innerHTML =
                    '<span class="w-2 h-2 rounded-full" style="background: var(--purple);"></span><span class="text-gray-500">Connected</span>';
            };

            ws.onclose = () => {
                document.getElementById('connection-status').innerHTML =
                    '<span class="w-2 h-2 rounded-full" style="background: var(--hotpink);"></span><span class="text-gray-500">Disconnected</span>';
                setTimeout(connect, 2000);
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                updateDashboard(data);
            };
        }

        function updateDashboard(data) {
            // Update devices
            for (const [name, device] of Object.entries(data.devices)) {
                updateDevice(name, device);
            }

            // Update pipeline
            updatePipeline(data.pipeline);

            // Update stats
            document.getElementById('stat-abort').textContent = data.stats.abort;
            document.getElementById('stat-degrade').textContent = data.stats.degrade;
            document.getElementById('stat-retry').textContent = data.stats.retry;
            document.getElementById('stat-skip').textContent = data.stats.skip;
            document.getElementById('stat-total').textContent = data.stats.total;

            // Update events
            updateEvents(data.events);
        }

        function updateDevice(name, device) {
            const statusEl = document.getElementById(`status-${name}`);
            if (statusEl) {
                statusEl.textContent = device.status.toUpperCase();
                statusEl.className = `text-xs font-medium px-2 py-1 rounded-full bg-gray-100 status-${device.status}`;
            }

            // Device-specific telemetry
            if (name === 'heater_1') {
                const temp = device.telemetry.temperature || 25;
                document.getElementById('temp-heater_1').textContent = `${temp.toFixed(1)}°C`;
                document.getElementById('tempbar-heater_1').style.width = `${Math.min(temp / 150 * 100, 100)}%`;
            } else if (name === 'pump_1') {
                document.getElementById('flow-pump_1').textContent = `${(device.telemetry.flow_rate || 0).toFixed(1)} mL/min`;
                document.getElementById('pressure-pump_1').textContent = `${(device.telemetry.pressure || 1).toFixed(1)} bar`;
            } else if (name === 'stage_1') {
                document.getElementById('x-stage_1').textContent = (device.telemetry.x || 0).toFixed(1);
                document.getElementById('y-stage_1').textContent = (device.telemetry.y || 0).toFixed(1);
                document.getElementById('z-stage_1').textContent = (device.telemetry.z || 0).toFixed(1);
            } else if (name === 'spec_1') {
                const signal = device.telemetry.signal || 0;
                document.getElementById('signal-spec_1').textContent = signal.toFixed(0);
                document.getElementById('signalbar-spec_1').style.width = `${Math.min(signal / 65000 * 100, 100)}%`;
            }

            // Error display
            const errorEl = document.getElementById(`error-${name}`);
            if (errorEl) {
                if (device.error) {
                    errorEl.textContent = device.error;
                    errorEl.classList.remove('hidden');
                } else {
                    errorEl.classList.add('hidden');
                }
            }
        }

        function updatePipeline(pipeline) {
            const stages = ['IDLE', 'SENSE', 'CLASSIFY', 'ANALYZE', 'DECIDE', 'EXECUTE', 'VERIFY', 'MEMORY'];

            for (let i = 0; i < 8; i++) {
                const el = document.getElementById(`stage-${i}`);
                el.className = 'flex-1 h-12 rounded-xl flex items-center justify-center text-xs font-medium transition-all ';

                if (i === pipeline.stage) {
                    el.className += 'stage-active text-white';
                } else if (i < pipeline.stage) {
                    el.className += 'stage-done text-white';
                } else {
                    el.className += 'stage-pending';
                }
            }

            const badge = document.getElementById('decision-badge');
            const deviceSpan = document.getElementById('pipeline-device');

            if (pipeline.decision) {
                badge.textContent = pipeline.decision.toUpperCase();
                badge.className = `px-4 py-2 rounded-xl text-sm font-bold decision-${pipeline.decision}`;
                badge.classList.remove('hidden');
            } else {
                badge.classList.add('hidden');
            }

            deviceSpan.textContent = pipeline.device ? `Device: ${pipeline.device}` : '';
        }

        function updateEvents(events) {
            const log = document.getElementById('event-log');
            log.innerHTML = events.map(e => {
                const levelColors = {
                    'INFO': 'color: var(--purple)',
                    'WARN': 'color: var(--gold)',
                    'ERROR': 'color: var(--hotpink)'
                };
                const color = levelColors[e.level] || 'color: var(--slate)';
                return `<div><span class="text-gray-400">${e.time}</span> <span style="${color}">[${e.level}]</span> <span class="text-gray-600">${e.message}</span></div>`;
            }).join('') || '<div class="text-gray-400">Waiting for events...</div>';
            log.scrollTop = log.scrollHeight;
        }

        function injectFault(device) {
            ws.send(JSON.stringify({ action: 'inject', device }));
        }

        function runAllScenarios() {
            ws.send(JSON.stringify({ action: 'run_all' }));
        }

        function resetDashboard() {
            ws.send(JSON.stringify({ action: 'reset' }));
        }

        // Update uptime
        setInterval(() => {
            const elapsed = Math.floor((Date.now() - startTime) / 1000);
            const h = Math.floor(elapsed / 3600).toString().padStart(2, '0');
            const m = Math.floor((elapsed % 3600) / 60).toString().padStart(2, '0');
            const s = (elapsed % 60).toString().padStart(2, '0');
            document.getElementById('uptime').textContent = `${h}:${m}:${s}`;
        }, 1000);

        // Start connection
        connect();
    </script>
</body>
</html>
"""


# ============================================================================
# Routes
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    return HTML_TEMPLATE


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)

    try:
        # Send initial state
        await websocket.send_json(state.to_dict())

        while True:
            # Receive commands
            data = await websocket.receive_json()
            action = data.get("action")

            if action == "inject":
                device = data.get("device")
                await inject_fault(device)
            elif action == "run_all":
                await run_all_scenarios()
            elif action == "reset":
                await reset_state()

    except WebSocketDisconnect:
        clients.remove(websocket)


async def broadcast(data: dict):
    """Send state to all connected clients."""
    for client in clients:
        try:
            await client.send_json(data)
        except:
            pass


def log_event(msg: str, level: str = "INFO"):
    """Add event to log."""
    state.events.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "level": level,
        "message": msg
    })


async def update_and_broadcast():
    """Update all clients with current state."""
    await broadcast(state.to_dict())


# ============================================================================
# Fault Scenarios
# ============================================================================

async def inject_fault(device_type: str):
    """Inject a fault for the specified device type."""

    if device_type == "heater":
        await run_heater_scenario()
    elif device_type == "pump":
        await run_pump_scenario()
    elif device_type == "positioner":
        await run_positioner_scenario()
    elif device_type == "spectrometer":
        await run_spectrometer_scenario()


async def run_heater_scenario():
    """Heater overshoot → DEGRADE"""
    device = state.devices["heater_1"]
    log_event("━━━ Heater Overshoot Scenario ━━━", "WARN")
    await update_and_broadcast()
    await asyncio.sleep(0.5)

    # Temperature rising
    device.status = "running"
    log_event("Heater heating to target 120°C...", "INFO")
    await update_and_broadcast()

    for temp in [60, 80, 100, 115, 125, 132, 138]:
        device.telemetry["temperature"] = temp
        if temp > 130:
            device.error = "⚠ OVERSHOOT: 138°C > 130°C limit"
            device.status = "error"
            log_event(f"FAULT DETECTED: Temperature {temp}°C exceeds safety limit", "ERROR")
        await update_and_broadcast()
        await asyncio.sleep(0.4)

    await asyncio.sleep(0.3)

    # Stage 1: SENSE
    state.pipeline.stage = 1
    state.pipeline.device = "heater_1"
    log_event("SENSE: Reading device telemetry...", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.6)
    log_event("  → temperature=138°C, status=error", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.4)

    # Stage 2: CLASSIFY
    state.pipeline.stage = 2
    log_event("CLASSIFY: Analyzing error type...", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.6)
    log_event("  → error_type: overshoot", "INFO")
    log_event("  → unsafe: True, recoverable: True", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.4)

    # Stage 3: ANALYZE
    state.pipeline.stage = 3
    log_event("ANALYZE: Checking telemetry signature...", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.6)
    log_event("  → pattern: drift (continuous rise)", "INFO")
    log_event("  → delta: +18°C over target", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.4)

    # Stage 4: DECIDE
    state.pipeline.stage = 4
    log_event("DECIDE: Evaluating recovery options...", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.5)
    log_event("  → Option 1: ABORT - lose sample", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.3)
    log_event("  → Option 2: DEGRADE - reduce to 110°C ✓", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.5)

    state.pipeline.decision = "degrade"
    state.stats["degrade"] += 1
    state.stats["total"] += 1
    log_event("DECISION: DEGRADE → set_temperature(110°C)", "WARN")
    await update_and_broadcast()
    await asyncio.sleep(0.6)

    # Stage 5: EXECUTE
    state.pipeline.stage = 5
    log_event("EXECUTE: Applying recovery action...", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.5)
    log_event("  → cool_down() initiated", "INFO")
    device.telemetry["temperature"] = 125
    await update_and_broadcast()
    await asyncio.sleep(0.4)
    device.telemetry["temperature"] = 118
    await update_and_broadcast()
    await asyncio.sleep(0.4)
    device.telemetry["temperature"] = 110
    device.status = "running"
    device.error = None
    log_event("  → target reached: 110°C", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.4)

    # Stage 6: VERIFY
    state.pipeline.stage = 6
    log_event("VERIFY: Checking postconditions...", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.6)
    log_event("  → temperature stable at 110°C ✓", "INFO")
    log_event("  → status=running ✓", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.4)

    # Stage 7: MEMORY
    state.pipeline.stage = 7
    log_event("MEMORY: Recording to experience DB...", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.5)
    log_event("  → pattern: overshoot → degrade", "INFO")
    log_event("  → outcome: SUCCESS", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.6)

    # Reset pipeline
    log_event("━━━ Scenario Complete ━━━", "INFO")
    state.pipeline.stage = 0
    state.pipeline.decision = None
    device.status = "idle"
    await update_and_broadcast()


async def run_pump_scenario():
    """Pump flow blocked → DEGRADE"""
    device = state.devices["pump_1"]
    log_event("━━━ Pump Blockage Scenario ━━━", "WARN")
    await update_and_broadcast()
    await asyncio.sleep(0.5)

    device.status = "running"
    log_event("Pump running at 50 mL/min...", "INFO")
    await update_and_broadcast()

    for flow in [50, 40, 25, 10, 3, 0]:
        device.telemetry["flow_rate"] = flow
        device.telemetry["pressure"] = 1 + (50 - flow) / 8
        if flow < 5:
            device.error = f"⚠ FLOW BLOCKED: {flow} mL/min"
            device.status = "error"
            log_event(f"FAULT DETECTED: Flow dropped to {flow} mL/min", "ERROR")
        await update_and_broadcast()
        await asyncio.sleep(0.4)

    await asyncio.sleep(0.3)

    # Stage 1: SENSE
    state.pipeline.stage = 1
    state.pipeline.device = "pump_1"
    log_event("SENSE: Reading pump telemetry...", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.6)
    log_event("  → flow_rate=0 mL/min, pressure=7.3 bar", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.4)

    # Stage 2: CLASSIFY
    state.pipeline.stage = 2
    log_event("CLASSIFY: Analyzing error type...", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.6)
    log_event("  → error_type: flow_blocked", "INFO")
    log_event("  → unsafe: True, recoverable: True", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.4)

    # Stage 3: ANALYZE
    state.pipeline.stage = 3
    log_event("ANALYZE: Checking flow signature...", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.6)
    log_event("  → pattern: drift (continuous drop)", "INFO")
    log_event("  → pressure rising: possible blockage", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.4)

    # Stage 4: DECIDE
    state.pipeline.stage = 4
    log_event("DECIDE: Evaluating recovery options...", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.5)
    log_event("  → Option 1: RETRY - prime pump", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.3)
    log_event("  → Option 2: DEGRADE - stop + check ✓", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.5)

    state.pipeline.decision = "degrade"
    state.stats["degrade"] += 1
    state.stats["total"] += 1
    log_event("DECISION: DEGRADE → stop_pump() + prime_pump()", "WARN")
    await update_and_broadcast()
    await asyncio.sleep(0.6)

    # Stage 5: EXECUTE
    state.pipeline.stage = 5
    log_event("EXECUTE: Applying recovery action...", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.5)
    log_event("  → stop_pump() executed", "INFO")
    device.telemetry["pressure"] = 4.0
    await update_and_broadcast()
    await asyncio.sleep(0.4)
    log_event("  → prime_pump() at low flow...", "INFO")
    device.telemetry["pressure"] = 2.0
    await update_and_broadcast()
    await asyncio.sleep(0.4)
    device.telemetry["flow_rate"] = 0
    device.telemetry["pressure"] = 1.0
    device.status = "idle"
    device.error = None
    log_event("  → pump stopped safely", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.4)

    # Stage 6: VERIFY
    state.pipeline.stage = 6
    log_event("VERIFY: Checking postconditions...", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.6)
    log_event("  → flow_rate=0 (stopped) ✓", "INFO")
    log_event("  → pressure=1.0 bar (normal) ✓", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.4)

    # Stage 7: MEMORY
    state.pipeline.stage = 7
    log_event("MEMORY: Recording to experience DB...", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.5)
    log_event("  → pattern: flow_blocked → degrade", "INFO")
    log_event("  → outcome: SUCCESS (manual check needed)", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.6)

    log_event("━━━ Scenario Complete ━━━", "INFO")
    state.pipeline.stage = 0
    state.pipeline.decision = None
    await update_and_broadcast()


async def run_positioner_scenario():
    """Positioner collision → ABORT"""
    device = state.devices["stage_1"]
    log_event("━━━ Positioner Collision Scenario ━━━", "ERROR")
    await update_and_broadcast()
    await asyncio.sleep(0.5)

    device.status = "running"
    log_event("Positioner moving to target (25, 0, 0)...", "INFO")
    await update_and_broadcast()

    for x in [3, 7, 11, 14, 15]:
        device.telemetry["x"] = x
        if x == 15:
            device.error = "⚠ COLLISION at x=15mm"
            device.status = "error"
            log_event(f"FAULT DETECTED: Collision at x={x}mm!", "ERROR")
        await update_and_broadcast()
        await asyncio.sleep(0.4)

    await asyncio.sleep(0.3)

    # Stage 1: SENSE
    state.pipeline.stage = 1
    state.pipeline.device = "stage_1"
    log_event("SENSE: Reading positioner state...", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.6)
    log_event("  → position=(15, 0, 0), status=error", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.4)

    # Stage 2: CLASSIFY
    state.pipeline.stage = 2
    log_event("CLASSIFY: Analyzing error type...", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.6)
    log_event("  → error_type: collision", "ERROR")
    log_event("  → unsafe: True, recoverable: FALSE", "ERROR")
    await update_and_broadcast()
    await asyncio.sleep(0.4)

    # Stage 3: ANALYZE
    state.pipeline.stage = 3
    log_event("ANALYZE: Checking motion signature...", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.6)
    log_event("  → pattern: stall (motion stopped)", "INFO")
    log_event("  → motor current spike detected", "WARN")
    await update_and_broadcast()
    await asyncio.sleep(0.4)

    # Stage 4: DECIDE
    state.pipeline.stage = 4
    log_event("DECIDE: Evaluating recovery options...", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.5)
    log_event("  → DEGRADE: NOT SAFE (position unknown)", "ERROR")
    await update_and_broadcast()
    await asyncio.sleep(0.3)
    log_event("  → RETRY: NOT SAFE (may cause damage)", "ERROR")
    await update_and_broadcast()
    await asyncio.sleep(0.3)
    log_event("  → ABORT: Only safe option ✓", "ERROR")
    await update_and_broadcast()
    await asyncio.sleep(0.5)

    state.pipeline.decision = "abort"
    state.stats["abort"] += 1
    state.stats["total"] += 1
    log_event("DECISION: ABORT → experiment terminated", "ERROR")
    await update_and_broadcast()
    await asyncio.sleep(0.6)

    # Stage 5: EXECUTE
    state.pipeline.stage = 5
    log_event("EXECUTE: Emergency shutdown...", "ERROR")
    await update_and_broadcast()
    await asyncio.sleep(0.5)
    log_event("  → stop() - all motion halted", "WARN")
    await update_and_broadcast()
    await asyncio.sleep(0.4)
    log_event("  → sample marked as COMPROMISED", "ERROR")
    await update_and_broadcast()
    await asyncio.sleep(0.4)

    # Stage 6: VERIFY
    state.pipeline.stage = 6
    log_event("VERIFY: Checking shutdown state...", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.6)
    log_event("  → motors disabled ✓", "INFO")
    log_event("  → position locked ✓", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.4)

    # Stage 7: MEMORY
    state.pipeline.stage = 7
    log_event("MEMORY: Recording to experience DB...", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.5)
    log_event("  → pattern: collision → abort", "ERROR")
    log_event("  → outcome: FAIL (manual intervention req)", "ERROR")
    await update_and_broadcast()
    await asyncio.sleep(0.6)

    log_event("━━━ Scenario Complete (ABORTED) ━━━", "ERROR")
    state.pipeline.stage = 0
    state.pipeline.decision = None
    await update_and_broadcast()


async def run_spectrometer_scenario():
    """Spectrometer saturation → DEGRADE"""
    device = state.devices["spec_1"]
    log_event("━━━ Spectrometer Saturation Scenario ━━━", "WARN")
    await update_and_broadcast()
    await asyncio.sleep(0.5)

    device.status = "running"
    log_event("Spectrometer acquiring at 100ms integration...", "INFO")
    await update_and_broadcast()

    for sig in [25000, 38000, 48000, 56000, 62000, 65000]:
        device.telemetry["signal"] = sig
        if sig > 60000:
            device.error = f"⚠ SATURATED: {sig} > 60000"
            device.status = "error"
            log_event(f"FAULT DETECTED: Signal {sig} exceeds max (65535)", "ERROR")
        await update_and_broadcast()
        await asyncio.sleep(0.4)

    await asyncio.sleep(0.3)

    # Stage 1: SENSE
    state.pipeline.stage = 1
    state.pipeline.device = "spec_1"
    log_event("SENSE: Reading spectrometer data...", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.6)
    log_event("  → signal=65000, integration=100ms", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.4)

    # Stage 2: CLASSIFY
    state.pipeline.stage = 2
    log_event("CLASSIFY: Analyzing error type...", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.6)
    log_event("  → error_type: signal_saturated", "INFO")
    log_event("  → unsafe: False, recoverable: True", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.4)

    # Stage 3: ANALYZE
    state.pipeline.stage = 3
    log_event("ANALYZE: Checking signal signature...", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.6)
    log_event("  → pattern: drift (continuous rise)", "INFO")
    log_event("  → peak at 65000 (16-bit max)", "WARN")
    await update_and_broadcast()
    await asyncio.sleep(0.4)

    # Stage 4: DECIDE
    state.pipeline.stage = 4
    log_event("DECIDE: Evaluating recovery options...", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.5)
    log_event("  → Option 1: SKIP - lose data point", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.3)
    log_event("  → Option 2: DEGRADE - reduce integration ✓", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.5)

    state.pipeline.decision = "degrade"
    state.stats["degrade"] += 1
    state.stats["total"] += 1
    log_event("DECISION: DEGRADE → reduce_integration(0.5)", "WARN")
    await update_and_broadcast()
    await asyncio.sleep(0.6)

    # Stage 5: EXECUTE
    state.pipeline.stage = 5
    log_event("EXECUTE: Applying recovery action...", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.5)
    log_event("  → reduce_integration(factor=0.5)", "INFO")
    device.telemetry["signal"] = 55000
    await update_and_broadcast()
    await asyncio.sleep(0.4)
    log_event("  → re-acquiring spectrum...", "INFO")
    device.telemetry["signal"] = 48000
    await update_and_broadcast()
    await asyncio.sleep(0.4)
    device.telemetry["signal"] = 45000
    device.status = "running"
    device.error = None
    log_event("  → signal now in valid range", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.4)

    # Stage 6: VERIFY
    state.pipeline.stage = 6
    log_event("VERIFY: Checking postconditions...", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.6)
    log_event("  → signal=45000 (< 60000) ✓", "INFO")
    log_event("  → integration=50ms (reduced) ✓", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.4)

    # Stage 7: MEMORY
    state.pipeline.stage = 7
    log_event("MEMORY: Recording to experience DB...", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.5)
    log_event("  → pattern: saturated → degrade", "INFO")
    log_event("  → outcome: SUCCESS", "INFO")
    await update_and_broadcast()
    await asyncio.sleep(0.6)

    log_event("━━━ Scenario Complete ━━━", "INFO")
    state.pipeline.stage = 0
    state.pipeline.decision = None
    device.status = "idle"
    await update_and_broadcast()


async def run_all_scenarios():
    """Run all 4 scenarios in sequence."""
    log_event("Starting all scenarios", "INFO")
    await update_and_broadcast()

    await run_heater_scenario()
    await asyncio.sleep(0.5)
    await run_pump_scenario()
    await asyncio.sleep(0.5)
    await run_positioner_scenario()
    await asyncio.sleep(0.5)
    await run_spectrometer_scenario()

    log_event("All scenarios complete", "INFO")
    await update_and_broadcast()


async def reset_state():
    """Reset all state."""
    global state
    state = DashboardState()
    state.devices = {
        "heater_1": DeviceState("heater_1", "Heater", telemetry={"temperature": 25.0, "target": 25.0}),
        "pump_1": DeviceState("pump_1", "Pump", telemetry={"flow_rate": 0.0, "pressure": 1.0}),
        "stage_1": DeviceState("stage_1", "Positioner", telemetry={"x": 0.0, "y": 0.0, "z": 0.0}),
        "spec_1": DeviceState("spec_1", "Spectrometer", telemetry={"signal": 0, "integration_ms": 100}),
    }
    log_event("Dashboard reset", "INFO")
    await update_and_broadcast()


# ============================================================================
# Main
# ============================================================================

def main():
    import uvicorn
    print("\n" + "=" * 60)
    print("  🔬 Exp-Agent Web Dashboard")
    print("=" * 60)
    print("\n  Open in browser: http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
