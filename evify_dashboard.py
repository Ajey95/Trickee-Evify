"""
Trickee x Evify — AI Intelligence Dashboard (Reactive)
======================================================
Premium Fleet & Driver Dashboard dynamically loading the synthetic V4.1 Evify Data.

Run: streamlit run evify_dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import os

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG & THEMING
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Trickee x Evify API", page_icon="⚡", layout="wide")

TEAL, GREEN, ORANGE, RED, GOLD = "#00d4ff", "#00c853", "#ff6b35", "#ff3b30", "#ffd600"
BG, CARD, BORDER, DIM = "#0d1117", "#13203a", "#1f3a5f", "#7d9bbd"

st.markdown(f"""
<style>
  .stApp {{ background-color: {BG} !important; color: #fff; }}
  [data-testid="stSidebar"] {{ background: {CARD} !important; border-right: 1px solid {BORDER}; }}
  .t-header {{ font-size: 28px; font-weight: 900; background: linear-gradient(90deg, {TEAL}, {GREEN}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  .t-section {{ font-size: 16px; font-weight: 700; color: {TEAL}; border-left: 3px solid {TEAL}; padding-left: 10px; margin: 20px 0 10px 0; }}
  
  /* KPI Cards */
  .kpi-card {{ background: linear-gradient(135deg, {CARD} 0%, #1a2d4a 100%); border: 1px solid {BORDER}; border-radius: 14px; padding: 16px 20px; box-shadow: 0 4px 16px rgba(0,212,255,0.06); text-align: center; }}
  .kpi-lbl {{ font-size: 11px; color: {DIM}; text-transform: uppercase; letter-spacing: 0.5px; }}
  .kpi-val {{ font-size: 34px; font-weight: 900; margin: 6px 0; color: #fff; }}
  .kpi-sub {{ font-size: 11px; color: #aaa; margin-top: 4px; }}
  
  /* Chips */
  .chip {{ display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 700; }}
  .chip-charging {{ background: #00c853; color: #000; }}
  .chip-discharging {{ background: #ff6b35; color: #fff; }}
  .chip-regen {{ background: #ffd600; color: #000; }}
  .chip-idling {{ background: #455a64; color: #fff; }}
  
  /* Sidebar status rows */
  .snap-row {{ display: flex; justify-content: space-between; margin: 3px 0; }}
  .snap-name {{ font-size: 12px; color: #c9d1d9; font-weight: 600; }}
</style>
""", unsafe_allow_html=True)

DARK_LAYOUT = dict(paper_bgcolor=BG, plot_bgcolor=CARD, font=dict(color="#c9d1d9"), margin=dict(l=10, r=10, t=40, b=10))

def soc_color(soc):
    if soc >= 60: return GREEN
    if soc >= 25: return GOLD
    return RED

# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "evify data 2.0", "evify_training_data.csv")

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
if df.empty:
    st.error("Data missing. Run generator first.")
    st.stop()

VEHICLES = sorted(df['vehicle_id'].unique())
time_steps = sorted(df['time'].unique())
N_STEPS = len(time_steps)

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR / REPLAY CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
if 'step_idx' not in st.session_state: st.session_state.step_idx = 0
if 'playing'  not in st.session_state: st.session_state.playing = False
if 'role'     not in st.session_state: st.session_state.role = "fleet"
if 'sel_veh'  not in st.session_state: st.session_state.sel_veh = VEHICLES[0]

with st.sidebar:
    st.markdown('<div class="t-header">⚡ Trickee x Evify</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:{DIM};font-size:12px;margin-bottom:18px;">EV Intelligence Platform</div>', unsafe_allow_html=True)
    st.divider()
    
    # Role selector
    st.markdown("**👤 Dashboard Mode**")
    role_pick = st.radio("", ["🏢 Fleet Manager", "🚗 Driver"], index=0 if st.session_state.role == "fleet" else 1, label_visibility="collapsed")
    st.session_state.role = "fleet" if "Fleet" in role_pick else "driver"

    if st.session_state.role == "driver":
        st.markdown("**🚗 Select Your Vehicle**")
        st.session_state.sel_veh = st.selectbox("", VEHICLES, index=VEHICLES.index(st.session_state.sel_veh), label_visibility="collapsed")
    st.divider()

    st.markdown("**📡 Telemetry Timeline**")
    step_val = st.slider("Timeline", 0, N_STEPS - 1, st.session_state.step_idx, label_visibility="collapsed")
    st.session_state.step_idx = step_val
    
    cur_time = pd.to_datetime(time_steps[st.session_state.step_idx])
    st.caption(f"🕒 **Current Time**: {cur_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    c1, c2 = st.columns(2)
    if c1.button("⏸ Pause" if st.session_state.playing else "▶ Play", use_container_width=True):
        st.session_state.playing = not st.session_state.playing
        st.rerun()
    if c2.button("⏮ Reset", use_container_width=True):
        st.session_state.step_idx = 0
        st.session_state.playing = False
        st.rerun()

    st.divider()
    
    # Fleet snapshot
    cur_state = df[df['time'] == time_steps[st.session_state.step_idx]]
    if not cur_state.empty and st.session_state.role == "fleet":
        st.markdown("**📊 All Vehicles**")
        for _, r in cur_state.iterrows():
            sc = soc_color(r.soc)
            icon = "🔌" if r.current < -0.5 and r.speed < 1 else ("♻️" if r.current < -0.5 else ("🏃" if r.speed >= 1 else "🛑"))
            st.markdown(
                f'<div class="snap-row">'
                f'<span class="snap-name">{r.vehicle_id}</span>'
                f'<span style="font-size:12px;color:{sc};font-weight:700;">{r.soc:.0f}% {icon}</span>'
                f'</div>', unsafe_allow_html=True)

# Auto-play loop
if st.session_state.playing:
    time.sleep(0.5)
    if st.session_state.step_idx < N_STEPS - 1:
        st.session_state.step_idx += 1
    else:
        st.session_state.playing = False
    st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
#  ███████  FLEET MANAGER VIEW
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.role == "fleet":
    st.markdown('<div class="t-header">🏢 Fleet Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:{DIM}; margin-bottom: 20px;">Live representation of Evify 2.0 telemetry with physics validation</div>', unsafe_allow_html=True)

    # ── 5 KPI Cards ──
    if not cur_state.empty:
        cols = st.columns(5)
        for i, (_, r) in enumerate(cur_state.iterrows()):
            sc  = soc_color(r.soc)
            if r.current < -0.5 and r.speed < 1:
                chp, chp_cls = "Charging 🔌", "chip-charging"
            elif r.current < -0.5 and r.speed >= 1:
                chp, chp_cls = "Regen ♻️", "chip-regen"
            elif r.current >= -0.5 and r.speed >= 1:
                chp, chp_cls = "Driving 🏃", "chip-discharging"
            else:
                chp, chp_cls = "Idling 🛑", "chip-idling"
                
            cols[i].markdown(f"""
            <div class="kpi-card">
              <div class="kpi-lbl">{r.vehicle_id}</div>
              <div class="kpi-val" style="color:{sc};">{r.soc:.1f}%</div>
              <div style="margin-top:6px;"><span class="chip {chp_cls}">{chp}</span></div>
              <div class="kpi-sub">{r.speed:.1f} km/h &nbsp;·&nbsp; {r.battery_voltage:.1f} V</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tA, tB, tC, tD = st.tabs(["🎯 Predictive Target", "⚡ Battery Analytics", "🌡️ Thermal Monitor", "🔬 Asset Health"])

    # ── TAB A: TARGET ───────────────────────────────────────────────────────
    with tA:
        st.markdown('<div class="t-section">🎯 The Target: Predicting Delta SOC</div>', unsafe_allow_html=True)
        st.info("Trickee V4.1 predicts **Delta SOC** (the shift in battery % over the next 5 mins) to an accuracy of 0.41%.")
        fig_delta = px.bar(
            cur_state, x="vehicle_id", y="delta_soc", 
            color="delta_soc", color_continuous_scale=[[0, RED], [0.5, "#455a64"], [1, GREEN]],
            text_auto=".2f", title="Next 5-Minute SOC Shift (Target Variable)"
        )
        fig_delta.add_hline(y=0, line_color="#ffffff")
        fig_delta.update_layout(**DARK_LAYOUT, yaxis_title="Delta SOC (%)", xaxis_title="")
        st.plotly_chart(fig_delta, use_container_width=True)

    # ── TAB B: BATTERY ──────────────────────────────────────────────────────
    with tB:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="t-section">State of Charge</div>', unsafe_allow_html=True)
            fig_soc = go.Figure()
            for _, r in cur_state.sort_values("soc").iterrows():
                fig_soc.add_trace(go.Bar(
                    x=[r.soc], y=[r.vehicle_id], orientation="h",
                    marker_color=soc_color(r.soc), text=f"{r.soc}%", textposition="auto", showlegend=False,
                ))
            fig_soc.add_vline(x=20, line_dash="dash", line_color=RED, annotation_text="Low Bat", annotation_font_color=RED)
            fig_soc.update_layout(**DARK_LAYOUT, height=280, xaxis=dict(range=[0, 108], title="SOC (%)"), yaxis=dict(title=""))
            st.plotly_chart(fig_soc, use_container_width=True)

        with c2:
            st.markdown('<div class="t-section">Battery Age — Charge Cycles</div>', unsafe_allow_html=True)
            fig_age = go.Figure()
            for _, r in cur_state.iterrows():
                hc = GREEN if r.soh >= 95 else (GOLD if r.soh >= 85 else RED)
                fig_age.add_trace(go.Bar(
                    x=[r.vehicle_id], y=[r.cycle_count], marker_color=hc, showlegend=False, text=f"{r.cycle_count}c", textposition="auto",
                ))
            fig_age.update_layout(**DARK_LAYOUT, height=280, yaxis=dict(title="Cycles Used"), xaxis=dict(title=""), title="Charge Cycles (Color = SOH)")
            st.plotly_chart(fig_age, use_container_width=True)

        st.markdown('<div class="t-section">⚡ Instantaneous Power Flow</div>', unsafe_allow_html=True)
        fig_pwr = go.Figure()
        for _, r in cur_state.iterrows():
            pc = GREEN if r.power_w >= 0 else ORANGE
            fig_pwr.add_trace(go.Bar(
                x=[r.vehicle_id], y=[r.power_w], marker_color=pc, showlegend=False, text=f"{r.power_w:+.0f}W", textposition="auto"
            ))
        fig_pwr.add_hline(y=0, line_color="#ffffff")
        fig_pwr.update_layout(**DARK_LAYOUT, height=300, yaxis=dict(title="Power (W) [+ Charging | − Discharging]"), title="Power Flow")
        st.plotly_chart(fig_pwr, use_container_width=True)

    # ── TAB C: THERMAL ──────────────────────────────────────────────────────
    with tC:
        st.markdown('<div class="t-section">🔥 Thermal Abuse Prevention</div>', unsafe_allow_html=True)
        st.caption("High current at high temperature = battery damage. Trickee flags this before the BMS catches it.")
        
        fig_abuse = px.scatter(
            cur_state, x="temp_max", y="current", color="soc",
            color_continuous_scale=[[0, RED], [0.5, GOLD], [1, GREEN]],
            size="cycle_count", size_max=50, text="vehicle_id",
            labels={"temp_max": "Max Cell Temperature (°C)", "current": "Battery Current (A)", "soc": "SOC%"},
            title="Thermal vs Load Matrix — All Vehicles", height=400,
        )
        fig_abuse.update_traces(textposition="top center")
        fig_abuse.add_hrect(
            y0=-25, y1=-70, x0=35, x1=50, fillcolor=RED, opacity=0.08,
            annotation_text="⚠️ High Discharge / Temp Risk", annotation_font_color=RED,
        )
        fig_abuse.update_layout(**DARK_LAYOUT)
        st.plotly_chart(fig_abuse, use_container_width=True)
        
        c1, c2 = st.columns(2)
        with c1:
            fig_temp = px.bar(
                cur_state, x="vehicle_id", y="temp_rise_rate", color="temp_max",
                color_continuous_scale="Inferno", title="Temperature Rise Rate (Color = Max Temp °C)", text="temp_rise_rate"
            )
            fig_temp.update_traces(texttemplate="%{text:+.1f}°C/m", textposition="outside")
            fig_temp.update_layout(**DARK_LAYOUT)
            st.plotly_chart(fig_temp, use_container_width=True)
            
        with c2:
            fig_sag = px.bar(
                cur_state, x="vehicle_id", y="voltage_sag_v", color="voltage_sag_v",
                color_continuous_scale=[[0, GREEN], [1, RED]], title="Voltage Sag (OCV - Terminal)"
            )
            fig_sag.update_layout(**DARK_LAYOUT)
            st.plotly_chart(fig_sag, use_container_width=True)

    # ── TAB D: HEALTH ───────────────────────────────────────────────────────
    with tD:
        st.markdown('<div class="t-section">🧬 Long-term Capital Health</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        
        with c1:
            imbal = cur_state[["vehicle_id", "cell_imbalance_mv"]].copy()
            imbal["Status"] = imbal["cell_imbalance_mv"].apply(lambda x: "Critical" if x > 50 else ("Monitor" if x > 20 else "Healthy"))
            color_map = {"Critical": RED, "Monitor": GOLD, "Healthy": GREEN}

            fig_imbal = px.bar(
                imbal.sort_values("cell_imbalance_mv"), x="cell_imbalance_mv", y="vehicle_id", orientation="h",
                color="Status", color_discrete_map=color_map, text="cell_imbalance_mv",
                title="Cell Voltage Imbalance (mV)", height=280
            )
            fig_imbal.update_traces(texttemplate="%{text:.1f} mV", textposition="auto")
            fig_imbal.add_vline(x=20, line_dash="dash", line_color=GOLD, annotation_text="Monitor")
            fig_imbal.add_vline(x=50, line_dash="dash", line_color=RED, annotation_text="Action!")
            fig_imbal.update_layout(**DARK_LAYOUT)
            st.plotly_chart(fig_imbal, use_container_width=True)

        with c2:
            fig_r0 = px.bar(
                cur_state, x="vehicle_id", y="r_internal_mohm", color="soh",
                color_continuous_scale=["red", "yellow", "green"], title="Internal Resistance (mΩ) by SOH%"
            )
            fig_r0.update_layout(**DARK_LAYOUT)
            st.plotly_chart(fig_r0, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  ████████  DRIVER VIEW
# ═════════════════════════════════════════════════════════════════════════════
else:
    veh_rows = cur_state[cur_state.vehicle_id == st.session_state.sel_veh]
    if veh_rows.empty:
        st.info(f"No data for **{st.session_state.sel_veh}** at this step. Advance slider ▶")
        st.stop()
        
    r = veh_rows.iloc[0]
    chp_cls = "chip-charging" if r.current > 0 else "chip-discharging"
    chp_lbl = "Charging 🔌" if r.current > 0 else "Driving 🏃"
    
    st.markdown(
        f'<div class="t-header">🚗 My Dashboard — {r.vehicle_id}</div>'
        f'<div class="t-sub"><span class="chip {chp_cls}">{chp_lbl}</span> &nbsp;·&nbsp; '
        f'{r.cycle_count} charge cycles &nbsp;·&nbsp; SOH {r.soh}%</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ── FOUR GAUGES ──────────────────────────────────────────────────────────
    g1, g2, g3, g4 = st.columns(4)

    with g1:
        fig_g_soc = go.Figure(go.Indicator(
            mode="gauge+number", value=r.soc,
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": soc_color(r.soc)}, "bgcolor": CARD,
                   "steps": [{"range": [0, 20], "color": "#2a1010"}]},
            title={"text": "STATE OF CHARGE", "font": {"color": DIM, "size": 12}}
        ))
        fig_g_soc.update_layout(**DARK_LAYOUT, height=240)
        st.plotly_chart(fig_g_soc, use_container_width=True)

    with g2:
        fig_g_v = go.Figure(go.Indicator(
            mode="gauge+number", value=r.battery_voltage,
            gauge={"axis": {"range": [44, 58]}, "bar": {"color": TEAL}, "bgcolor": CARD},
            title={"text": "PACK VOLTAGE (V)", "font": {"color": DIM, "size": 12}}
        ))
        fig_g_v.update_layout(**DARK_LAYOUT, height=240)
        st.plotly_chart(fig_g_v, use_container_width=True)

    with g3:
        cur_col = GREEN if r.current > 0 else ORANGE
        cur_lbl = "CHARGING (A)" if r.current > 0 else "DISCHARGE (A)"
        fig_g_c = go.Figure(go.Indicator(
            mode="gauge+number", value=abs(r.current),
            gauge={"axis": {"range": [0, 60]}, "bar": {"color": cur_col}, "bgcolor": CARD},
            title={"text": cur_lbl, "font": {"color": DIM, "size": 12}}
        ))
        fig_g_c.update_layout(**DARK_LAYOUT, height=240)
        st.plotly_chart(fig_g_c, use_container_width=True)

    with g4:
        t_col = GREEN if r.temp_max < 38 else (GOLD if r.temp_max < 45 else RED)
        fig_g_t = go.Figure(go.Indicator(
            mode="gauge+number", value=r.temp_max,
            gauge={"axis": {"range": [20, 55]}, "bar": {"color": t_col}, "bgcolor": CARD,
                   "steps": [{"range": [45, 55], "color": "#2a1010"}]},
            title={"text": "BATTERY TEMP (°C)", "font": {"color": DIM, "size": 12}}
        ))
        fig_g_t.update_layout(**DARK_LAYOUT, height=240)
        st.plotly_chart(fig_g_t, use_container_width=True)
