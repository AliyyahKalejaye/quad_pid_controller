import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import traceback

# ==============================
# STREAMLIT PAGE SETUP
# ==============================
st.set_page_config(
    page_title="PID Controller",
    layout="wide"
)

st.title("PID Controller")
st.write("Interactive tuning of an existing PID controller")

# ==============================
# SAFE IMPORT OF PID CONTROLLER
# ==============================
try:
    from pid_core import PID
except Exception as e:
    st.error("❌ Failed to import PID controller from `droneControl.py`")
    st.code(traceback.format_exc())
    st.stop()

# ==============================
# SIDEBAR CONTROLS
# ==============================
st.sidebar.header("PID Parameters")

kp = st.sidebar.slider("Kp (Proportional)", 0.0, 10.0, 1.0, 0.1)
ki = st.sidebar.slider("Ki (Integral)", 0.0, 5.0, 0.0, 0.05)
kd = st.sidebar.slider("Kd (Derivative)", 0.0, 5.0, 0.0, 0.05)

st.sidebar.divider()

setpoint = st.sidebar.slider("Setpoint", 0.0, 10.0, 5.0, 0.1)
simulation_time = st.sidebar.slider("Simulation Time (s)", 2.0, 20.0, 10.0, 1.0)

st.sidebar.divider()

reset_pid = st.sidebar.button("🔄 Reset PID State")

# ==============================
# INITIALIZE / RESET PID
# ==============================
if "pid" not in st.session_state or reset_pid:
    st.session_state.pid = PID(kp, ki, kd)

pid = st.session_state.pid

# Update gains dynamically (do NOT reset state)
pid.kp = kp
pid.ki = ki
pid.kd = kd

# ==============================
# SIMULATION PARAMETERS
# ==============================
dt = 0.01
time = np.arange(0, simulation_time, dt)

x = 0.0  # system output

output_history = []
control_history = []
error_history = []

# ==============================
# RUN SIMULATION (WITH SAFETY)
# ==============================
try:
    for _ in time:
        error = setpoint - x
        u = pid.update(setpoint, x, dt)

        # Optional safety clamp (anti-explosion)
        u = np.clip(u, -100.0, 100.0)

        # Simple first-order plant
        x += (-x + u) * dt

        output_history.append(x)
        control_history.append(u)
        error_history.append(error)

except Exception as e:
    st.error("❌ Error occurred during PID simulation")
    st.code(traceback.format_exc())
    st.stop()

# ==============================
# PLOTS
# ==============================
col1, col2 = st.columns(2)

with col1:
    st.subheader("System Output")

    fig1, ax1 = plt.subplots()
    ax1.plot(time, output_history, label="Output")
    ax1.plot(time, [setpoint] * len(time), "--", label="Setpoint")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True)

    st.pyplot(fig1)

with col2:
    st.subheader("Control Signal")

    fig2, ax2 = plt.subplots()
    ax2.plot(time, control_history, label="Control Output (u)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Control Effort")
    ax2.grid(True)

    st.pyplot(fig2)

# ==============================
# ERROR PLOT (IMPORTANT FOR PID)
# ==============================
st.subheader("Tracking Error")

fig3, ax3 = plt.subplots()
ax3.plot(time, error_history, label="Error (Setpoint − Output)")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Error")
ax3.grid(True)

st.pyplot(fig3)

# ==============================
# DEBUG / INTERNAL STATE VIEW
# ==============================
with st.expander("🛠 Debug / Internal PID State"):
    st.write("PID Gains")
    st.json({
        "Kp": pid.kp,
        "Ki": pid.ki,
        "Kd": pid.kd
    })

    if hasattr(pid, "integral"):
        st.write(f"Integral Term: `{pid.integral}`")

    if hasattr(pid, "prev_error"):
        st.write(f"Previous Error: `{pid.prev_error}`")

    st.write(f"Final Output Value: `{output_history[-1]}`")
    st.write(f"Final Error: `{error_history[-1]}`")

# ==============================
# TUNING HELP
# ==============================
st.markdown("""
### PID Tuning Notes
- **Kp**: Increases responsiveness, too high → oscillations
- **Ki**: Eliminates steady-state error, too high → windup
- **Kd**: Dampens oscillations, sensitive to noise


st.markdown(
    """
    <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #0e1117;
            color: #fafafa;
            text-align: center;
            padding: 10px;
            font-size: 0.9rem;
            z-index: 100;
        }
    </style>

    <div class="footer">
        Built by <strong>Aliyyah Kalejaye</strong>
    </div>
    """,
    unsafe_allow_html=True
)


💡 Use the **error plot** to judge tuning quality.
""")


