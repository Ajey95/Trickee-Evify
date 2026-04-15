"""
Trickee x Evify — LIVE PREDICTIVE AI DASHBOARD
==============================================
Instead of just playing back historical CSV data (reactive), this dashboard
loads the actual trained PyTorch LSTM model (V4.1) and makes live inferences
on a rolling 100-minute context window.

Run: streamlit run evify_predictive_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import os
import torch
import torch.nn as nn
import joblib

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG & THEMING
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Trickee Predictive AI", page_icon="🧠", layout="wide")

TEAL, GREEN, ORANGE, RED, GOLD, MAGENTA = "#00d4ff", "#00c853", "#ff6b35", "#ff3b30", "#ffd600", "#ff00ff"
BG, CARD, BORDER, DIM = "#0d1117", "#13203a", "#1f3a5f", "#7d9bbd"

st.markdown(f"""
<style>
  .stApp {{ background-color: {BG} !important; color: #fff; }}
  [data-testid="stSidebar"] {{ background: {CARD} !important; border-right: 1px solid {BORDER}; }}
  .t-header {{ font-size: 28px; font-weight: 900; background: linear-gradient(90deg, {MAGENTA}, {TEAL}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  .t-section {{ font-size: 16px; font-weight: 700; color: {MAGENTA}; border-left: 3px solid {MAGENTA}; padding-left: 10px; margin: 20px 0 10px 0; }}
  .kpi-card {{ background: {BORDER}; border-radius: 10px; padding: 15px; text-align: center; border: 1px solid {MAGENTA}40; }}
  .kpi-val {{ font-size: 28px; font-weight: bold; color: #fff; text-shadow: 0 0 10px {MAGENTA}80; }}
  .kpi-lbl {{ font-size: 11px; color: {DIM}; text-transform: uppercase; }}
  .chip-ai {{ background: {MAGENTA}; color: #fff; padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 12px; }}
</style>
""", unsafe_allow_html=True)

DARK_LAYOUT = dict(paper_bgcolor=BG, plot_bgcolor=CARD, font=dict(color="#c9d1d9"), margin=dict(l=10, r=10, t=40, b=10))

# ─────────────────────────────────────────────────────────────────────────────
#  AI MODEL DEFINITION (Must match V4.1 exactly)
# ─────────────────────────────────────────────────────────────────────────────
class BatteryRangeModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm      = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=lstm_dropout, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=4, dropout=dropout)
        self.bn        = nn.BatchNorm1d(hidden_size * 2)
        self.fc1       = nn.Linear(hidden_size * 2, 128)
        self.fc2       = nn.Linear(128, 64)
        self.fc3       = nn.Linear(64, 1)
        self.drop      = nn.Dropout(dropout)
        self.relu      = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)
        out, _ = self.attention(out, out, out)
        out    = out[:, -1, :]
        out    = self.bn(out)
        out    = self.relu(self.fc1(out)); out = self.drop(out)
        out    = self.relu(self.fc2(out)); out = self.drop(out)
        return self.fc3(out)

FEATURE_COLS = [
    'soc', 'current', 'battery_voltage', 'soh', 'power', 'speed', 
    'ignstatus', 'allow_charging', 'regen_status', 'throttle_status', 
    'cell_temperature_01', 'temp_rise_rate', 'cycle_count', 'cell_imbalance_mv', 
    'wh_throughput', 'r_internal_mohm', 'voltage_sag_v', 'power_density', 
    'minute_of_day', 'day_of_week'
]
SEQ_LEN = 20

# ─────────────────────────────────────────────────────────────────────────────
#  DATA & MODEL LOADING CACHE
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(__file__)
DATA_PATH  = os.path.join(ROOT_DIR, "evify_data_2.0", "evify_training_data.csv")
MODEL_DIR  = os.path.join(ROOT_DIR, "aicodeold", "model_training", "v4")

@st.cache_resource
def load_ai_engine():
    model_path  = os.path.join(MODEL_DIR, "battery_model_v4_1.pth")
    scaler_path = os.path.join(MODEL_DIR, "scaler_v4_1.joblib")
    ysc_path    = os.path.join(MODEL_DIR, "y_scaler_v4_1.joblib")
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(ysc_path)):
        return None, None, None, False
        
    scaler   = joblib.load(scaler_path)
    y_scaler = joblib.load(ysc_path)
    
    model = BatteryRangeModel(len(FEATURE_COLS), hidden_size=64)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False)['model_state_dict'])
    model.eval()
    
    return model, scaler, y_scaler, True

@st.cache_data
def load_evify_data():
    if not os.path.exists(DATA_PATH): return pd.DataFrame()
    df = pd.read_csv(DATA_PATH)
    df['time'] = pd.to_datetime(df['event_time'], errors='coerce')
    df = df.dropna(subset=['time']).sort_values(['vehicle_id', 'time']).reset_index(drop=True)
    
    # Ensure numerics
    for c in ['soc','battery_voltage','current','soh','speed','temp_max','cycle_count','cell_imbalance_mv']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
            
    # Physics Engineering
    df['power_w'] = df['battery_voltage'] * df['current']
    df['power'] = df['power_w']
    df['ignstatus'] = df.get('ignition_on', pd.Series([1]*len(df))).astype(int)
    df['allow_charging'] = df.get('charge_plug', pd.Series([0]*len(df))).astype(int)
    df['cell_temperature_01'] = df['temp_max']
    
    r_age = 0.055 * (1.0 + 0.003 * df['cycle_count'])
    r_soh = r_age * (1.0 + (100.0 - df['soh']) / 200.0)
    df['r_internal_mohm'] = (r_soh * 1000).round(2)
    
    # OCV
    s = df['soc'].clip(0.0, 100.0)
    oc = pd.Series(0.0, index=s.index)
    oc[s <= 10] = 42.0 + s[s <= 10] * 0.20
    m2 = (s > 10) & (s <= 20); oc[m2] = 44.0 + (s[m2] - 10) * 0.40
    m3 = (s > 20) & (s <= 90); oc[m3] = 48.0 + (s[m3] - 20) * 0.08571
    oc[s > 90] = 54.0 + (s[s > 90] - 90) * 0.44
    df['voltage_sag_v'] = (oc - df['battery_voltage']).round(3)

    df['temp_rise_rate'] = df.groupby('vehicle_id')['temp_max'].diff().fillna(0.0).clip(-5, 5)
    df['power_density'] = (df['power_w'] / 1824.0).round(4)
    df['minute_of_day'] = df['time'].dt.hour * 60 + df['time'].dt.minute
    df['day_of_week'] = df['time'].dt.dayofweek
    
    # Delta Target
    df['delta_soc'] = df.groupby('vehicle_id')['soc'].shift(-1) - df['soc']
    df = df.dropna(subset=['delta_soc']).reset_index(drop=True)
    return df

df = load_evify_data()
model, scaler, y_scaler, ai_ready = load_ai_engine()

if df.empty:
    st.error("Data missing. Run generator first.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
VEHICLES = sorted(df['vehicle_id'].unique())
time_steps = sorted(df['time'].unique())

if 'p_step_idx' not in st.session_state: st.session_state.p_step_idx = SEQ_LEN
if 'p_playing'  not in st.session_state: st.session_state.p_playing = False
if 'sel_veh'    not in st.session_state: st.session_state.sel_veh = VEHICLES[0]

with st.sidebar:
    st.markdown('<div class="t-header">🧠 Trickee AI</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:{DIM};font-size:12px;">Live Inference Dashboard</div>', unsafe_allow_html=True)
    st.divider()

    if not ai_ready:
        st.error(f"⚠️ **AI Model Missing**\nPlease upload `battery_model_v4_1.pth` and scalers to the `v4` folder to enable Live AI.")
        st.stop()
    else:
        st.success("✅ **V4.1 AI Engine Loaded**")

    st.markdown("**1. Select Vehicle Tracker**")
    st.session_state.sel_veh = st.selectbox("", VEHICLES, label_visibility="collapsed")
    
    st.markdown("**2. Telemetry Timeline**")
    step_val = st.slider("Timeline", SEQ_LEN, len(time_steps) - 1, st.session_state.p_step_idx, label_visibility="collapsed")
    st.session_state.p_step_idx = step_val
    
    cur_time = pd.to_datetime(time_steps[st.session_state.p_step_idx])
    st.caption(f"🕒 **Current Time**: {cur_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    c1, c2 = st.columns(2)
    if c1.button("⏸ Pause" if st.session_state.p_playing else "▶ Play", use_container_width=True):
        st.session_state.p_playing = not st.session_state.p_playing
        st.rerun()
    if c2.button("⏮ Reset", use_container_width=True):
        st.session_state.p_step_idx = SEQ_LEN
        st.session_state.p_playing = False
        st.rerun()

# Auto-play loop
if st.session_state.p_playing:
    time.sleep(0.5)
    if st.session_state.p_step_idx < len(time_steps) - 1:
        st.session_state.p_step_idx += 1
    else:
        st.session_state.p_playing = False
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
#  LIVE AI INFERENCE CORE
# ─────────────────────────────────────────────────────────────────────────────
veh_df = df[df['vehicle_id'] == st.session_state.sel_veh].reset_index(drop=True)

# Find the row index closest to the current global timeline step
current_row_idx = veh_df[veh_df['time'] <= cur_time].index.max()

if pd.isna(current_row_idx) or current_row_idx < SEQ_LEN:
    st.info("Insufficient historical context (Need 100 minutes of prior driving for the LSTM). Advance the slider.")
    st.stop()

# Extract the rolling 20-step window ending at the current time
window_df = veh_df.iloc[current_row_idx - SEQ_LEN + 1 : current_row_idx + 1]

# Do AI Inference
with torch.no_grad():
    X_raw = window_df[FEATURE_COLS].values
    X_scaled = scaler.transform(X_raw).astype(np.float32)
    X_tensor = torch.tensor(X_scaled).unsqueeze(0)  # Shape: (1, 20, 20)
    
    # Model returns scaled delta_soc
    pred_delta_scaled = model(X_tensor).numpy().reshape(-1, 1)
    
    # Unscale delta
    pred_delta = y_scaler.inverse_transform(pred_delta_scaled)[0][0]

cur_row = window_df.iloc[-1]
actual_soc = cur_row['soc']
predicted_next_soc = actual_soc + pred_delta

# If we have the ground truth "future" available, grab it to show how accurate we were
future_row_idx = current_row_idx + 1
true_next_soc = veh_df.iloc[future_row_idx]['soc'] if future_row_idx < len(veh_df) else None
true_delta = cur_row['delta_soc']

# ─────────────────────────────────────────────────────────────────────────────
#  UI RENDERING
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f'<div class="t-header">🧠 Predictive Engine — {st.session_state.sel_veh}</div>', unsafe_allow_html=True)
st.markdown(f'<div style="color:{DIM};">LSTM evaluating 100 minutes of thermodynamic history to predict the next 5 minutes.</div>', unsafe_allow_html=True)
st.divider()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="kpi-card"><div class="kpi-lbl">Current SOC (Reality)</div><div class="kpi-val" style="color:#fff;">{actual_soc:.1f}%</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="kpi-card"><div class="kpi-lbl">AI Predicted Shift (5m)</div><div class="kpi-val">{pred_delta:+.2f}%</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="kpi-card"><div class="kpi-lbl">AI Predicted Next SOC</div><div class="kpi-val">{predicted_next_soc:.1f}%</div></div>', unsafe_allow_html=True)
with col4:
    if true_next_soc is not None:
        error = abs(predicted_next_soc - true_next_soc)
        st.markdown(f'<div class="kpi-card"><div class="kpi-lbl">True Next SOC (Spoilers)</div><div class="kpi-val" style="color:{GREEN}">{true_next_soc:.1f}%</div><div style="font-size:11px;color:{DIM}">AI Error: {error:.2f}%</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="kpi-card"><div class="kpi-lbl">Future True SOC</div><div class="kpi-val" style="color:{DIM}">Unknown</div></div>', unsafe_allow_html=True)

st.divider()

# ── 100-MINUTE CONTEXT GRAPH ──
st.markdown('<div class="t-section">📈 The Predictive Horizon</div>', unsafe_allow_html=True)
st.caption("Solid line is known history. Dotted magenta line is what the AI predicts will happen in the future.")

fig = go.Figure()

# Plot historical window (real SOC)
fig.add_trace(go.Scatter(
    x=window_df['time'], y=window_df['soc'], 
    mode='lines+markers', name='Actual History (100 min)', 
    line=dict(color=TEAL, width=3)
))

# Plot future prediction point
next_time = cur_time + pd.Timedelta(minutes=5)
fig.add_trace(go.Scatter(
    x=[cur_time, next_time], y=[actual_soc, predicted_next_soc],
    mode='lines+markers', name='AI Prediction (Future)',
    line=dict(color=MAGENTA, width=4, dash='dashdot'),
    marker=dict(size=10, symbol='star')
))

# If we have the ground truth, plot it as a hollow circle for comparison
if true_next_soc is not None:
    fig.add_trace(go.Scatter(
        x=[next_time], y=[true_next_soc],
        mode='markers', name='Actual Future (Verification)',
        marker=dict(color=GREEN, size=12, line=dict(color=GREEN, width=2), symbol='circle-open')
    ))

fig.update_layout(
    **DARK_LAYOUT, height=450,
    yaxis=dict(title="SOC (%)", range=[max(0, actual_soc - 5), min(100, actual_soc + 5)]),
    xaxis=dict(title="Time"),
    hovermode='x unified'
)
st.plotly_chart(fig, use_container_width=True)

# ── CONTEXT FEATURES ──
st.markdown('<div class="t-section">🔬 What the AI is internally looking at to make this guess:</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
c1.metric("Thermal Momentum", f"{cur_row['temp_rise_rate']:+.2f} °C/m", "If heating fast, voltage drops sooner", delta_color="inverse")
c2.metric("Motor Load Stress", f"{cur_row['power_density']:.3f} kW/kWh", "Aggressive accelerating saps range")
c3.metric("Pack Resistance", f"{cur_row['r_internal_mohm']:.1f} mΩ", "Old packs drop SOC faster (SOH proxy)")
