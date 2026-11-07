import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Backpropagation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BLUE = "#4285F4"
RED = "#DB4437"
YELLOW = "#F4B400"
GREEN = "#0F9D58"
PURPLE = "#6600FF"
GRAY = "#94a3b8"
BG = "#0b1220"

STEPS = [
    {
        "title": "Backpropagation algorithm",
        "body": "Backpropagation trains large networks quickly by sending the error backward through the same weights used in the forward pass. Walk the steps the way the Google lesson uses scroll.",
        "focus": "none",
        "show": [],
    },
    {
        "title": "Simple neural network",
        "body": "One input, one output, and two hidden layers of two nodes. Neighboring layers are linked by weights \(w_{ij}\). Those weights are the parameters the trainer updates.",
        "focus": "net",
        "show": ["weights"],
    },
    {
        "title": "Activation function",
        "body": "Each node has a total input \(x\), an activation \(f(x)\), and an output \(y = f(x)\). \(f\) must be nonlinear or the whole net can only fit a line. This demo uses sigmoid: \(f(x) = 1 / (1 + e^{-x})\).",
        "focus": "activation",
        "show": ["y"],
    },
    {
        "title": "Error function",
        "body": "The predicted output \(y_{out}\) should match the target \(y_{target}\). Distance is measured with \(E = \\frac{1}{2}(y_{out} - y_{target})^2\).",
        "focus": "output",
        "show": ["y", "error"],
    },
    {
        "title": "Forward pass · input",
        "body": "Take one example \((x_{input}, y_{target})\) and write it on the input node. The input has no activation, so \(y_1 = x_{input}\).",
        "focus": "forward_input",
        "show": ["y"],
    },
    {
        "title": "Forward pass · hidden input",
        "body": "For each node \(j\) in the next layer, \(x_j = \\sum_i w_{ij} y_i + b_j\). The first hidden layer now has a total input from the example.",
        "focus": "forward_h1_x",
        "show": ["x", "weights"],
    },
    {
        "title": "Forward pass · hidden output",
        "body": "Apply the activation: \(y = f(x)\). For sigmoid, \(y = 1 / (1 + e^{-x})\). Those \(y\) values become the inputs to the next layer.",
        "focus": "forward_h1_y",
        "show": ["x", "y"],
    },
    {
        "title": "Forward pass · rest of the net",
        "body": "Repeat the two formulas through the second hidden layer and the output node. The network now has a prediction \(y_{out}\).",
        "focus": "forward_all",
        "show": ["x", "y", "error"],
    },
    {
        "title": "Error derivative",
        "body": "After comparing \(y_{out}\) with \(y_{target}\), backprop needs \(\\partial E / \\partial w_{ij}\) for every weight. The update is \(w_{ij} \\leftarrow w_{ij} - \\alpha \\, \\partial E / \\partial w_{ij}\), where \(\\alpha\) is the learning rate.",
        "focus": "update_rule",
        "show": ["error"],
    },
    {
        "title": "Extra derivatives",
        "body": "To get \(\\partial E / \\partial w_{ij}\), store two extra values on every node: how error changes with total input \(\\partial E / \\partial x\), and with output \(\\partial E / \\partial y\).",
        "focus": "extra_deriv",
        "show": ["dE_dx", "dE_dy"],
    },
    {
        "title": "Backward pass · output \(y\)",
        "body": "Start at the prediction. For \(E = \\frac{1}{2}(y_{out} - y_{target})^2\), \(\\partial E / \\partial y_{out} = y_{out} - y_{target}\).",
        "focus": "back_dy_out",
        "show": ["y", "dE_dy", "error"],
    },
    {
        "title": "Backward pass · output \(x\)",
        "body": "Chain rule: \(\\partial E / \\partial x = f'(x)\, \\partial E / \\partial y\). For sigmoid, \(f'(x) = f(x)(1 - f(x)) = y(1 - y)\).",
        "focus": "back_dx_out",
        "show": ["dE_dx", "dE_dy"],
    },
    {
        "title": "Backward pass · weights",
        "body": "The incoming weight derivative is \(\\partial E / \\partial w_{ij} = y_i \, \\partial E / \\partial x_j\). Purple edges show those gradients.",
        "focus": "back_dw",
        "show": ["dE_dx", "dE_dw", "weights"],
    },
    {
        "title": "Backward pass · previous layer",
        "body": "Send the error to the previous outputs: \(\\partial E / \\partial y_i = \\sum_j w_{ij}\, \\partial E / \\partial x_j\). The circle is complete: \(y \\rightarrow x \\rightarrow w \\rightarrow y\).",
        "focus": "back_prev",
        "show": ["dE_dx", "dE_dy", "weights"],
    },
    {
        "title": "Backward pass · remaining layers",
        "body": "Repeat the three formulas until every \(\\partial E / \\partial x\), \(\\partial E / \\partial y\), and \(\\partial E / \\partial w\) is filled. Then the update rule can move every weight.",
        "focus": "back_all",
        "show": ["dE_dx", "dE_dy", "dE_dw"],
    },
    {
        "title": "Weight update",
        "body": "Apply \(w \\leftarrow w - \\alpha \\, \\partial E / \\partial w\) once. Error after the update is shown under the network. Step through again with a new input or target from the sidebar.",
        "focus": "done",
        "show": ["y", "error", "weights"],
    },
]


def sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def run_network(x_in, y_target, w, b, alpha):
    y0 = float(x_in)
    x1 = w["w01"] * y0 + b["b1"]
    x2 = w["w02"] * y0 + b["b2"]
    y1, y2 = sigmoid(x1), sigmoid(x2)
    x3 = w["w13"] * y1 + w["w23"] * y2 + b["b3"]
    x4 = w["w14"] * y1 + w["w24"] * y2 + b["b4"]
    y3, y4 = sigmoid(x3), sigmoid(x4)
    x5 = w["w35"] * y3 + w["w45"] * y4 + b["b5"]
    y5 = sigmoid(x5)
    error = 0.5 * (y5 - y_target) ** 2

    dE_dy5 = y5 - y_target
    dE_dx5 = y5 * (1 - y5) * dE_dy5
    dE_dw35 = y3 * dE_dx5
    dE_dw45 = y4 * dE_dx5
    dE_dy3 = w["w35"] * dE_dx5
    dE_dy4 = w["w45"] * dE_dx5
    dE_dx3 = y3 * (1 - y3) * dE_dy3
    dE_dx4 = y4 * (1 - y4) * dE_dy4
    dE_dw13 = y1 * dE_dx3
    dE_dw23 = y2 * dE_dx3
    dE_dw14 = y1 * dE_dx4
    dE_dw24 = y2 * dE_dx4
    dE_dy1 = w["w13"] * dE_dx3 + w["w14"] * dE_dx4
    dE_dy2 = w["w23"] * dE_dx3 + w["w24"] * dE_dx4
    dE_dx1 = y1 * (1 - y1) * dE_dy1
    dE_dx2 = y2 * (1 - y2) * dE_dy2
    dE_dw01 = y0 * dE_dx1
    dE_dw02 = y0 * dE_dx2

    w_new = {k: w[k] - alpha * d for k, d in {
        "w01": dE_dw01, "w02": dE_dw02,
        "w13": dE_dw13, "w23": dE_dw23, "w14": dE_dw14, "w24": dE_dw24,
        "w35": dE_dw35, "w45": dE_dw45,
    }.items()}

    y1n, y2n = sigmoid(w_new["w01"] * y0 + b["b1"]), sigmoid(w_new["w02"] * y0 + b["b2"])
    y3n = sigmoid(w_new["w13"] * y1n + w_new["w23"] * y2n + b["b3"])
    y4n = sigmoid(w_new["w14"] * y1n + w_new["w24"] * y2n + b["b4"])
    y5n = sigmoid(w_new["w35"] * y3n + w_new["w45"] * y4n + b["b5"])
    error_new = 0.5 * (y5n - y_target) ** 2

    return {
        "x": [x_in, x1, x2, x3, x4, x5],
        "y": [y0, y1, y2, y3, y4, y5],
        "dE_dx": [0.0, dE_dx1, dE_dx2, dE_dx3, dE_dx4, dE_dx5],
        "dE_dy": [w["w01"] * dE_dx1 + w["w02"] * dE_dx2, dE_dy1, dE_dy2, dE_dy3, dE_dy4, dE_dy5],
        "dE_dw": {
            "w01": dE_dw01, "w02": dE_dw02,
            "w13": dE_dw13, "w23": dE_dw23, "w14": dE_dw14, "w24": dE_dw24,
            "w35": dE_dw35, "w45": dE_dw45,
        },
        "error": error,
        "error_new": error_new,
        "y_target": y_target,
    }


NODES = [
    {"id": 0, "name": "in", "layer": 0, "px": 0.0, "py": 0.0},
    {"id": 1, "name": "h1a", "layer": 1, "px": 1.0, "py": 0.85},
    {"id": 2, "name": "h1b", "layer": 1, "px": 1.0, "py": -0.85},
    {"id": 3, "name": "h2a", "layer": 2, "px": 2.0, "py": 0.85},
    {"id": 4, "name": "h2b", "layer": 2, "px": 2.0, "py": -0.85},
    {"id": 5, "name": "out", "layer": 3, "px": 3.0, "py": 0.0},
]
EDGES = [
    (0, 1, "w01"), (0, 2, "w02"),
    (1, 3, "w13"), (2, 3, "w23"), (1, 4, "w14"), (2, 4, "w24"),
    (3, 5, "w35"), (4, 5, "w45"),
]


def node_active(focus, node_id):
    if focus in ("none",):
        return False
    if focus in ("net", "activation", "update_rule", "extra_deriv", "forward_all", "back_all", "done"):
        return True
    if focus == "output" and node_id == 5:
        return True
    if focus == "forward_input" and node_id == 0:
        return True
    if focus in ("forward_h1_x", "forward_h1_y") and node_id in (1, 2):
        return True
    if focus in ("back_dy_out", "back_dx_out") and node_id == 5:
        return True
    if focus == "back_dw" and node_id in (3, 4, 5):
        return True
    if focus == "back_prev" and node_id in (3, 4, 5):
        return True
    return False


def fmt(v):
    return f"{v:+.3f}"


def draw_network(state, focus, show):
    fig = go.Figure()
    for src, dst, key in EDGES:
        x0, y0 = NODES[src]["px"], NODES[src]["py"]
        x1, y1 = NODES[dst]["px"], NODES[dst]["py"]
        show_dw = "dE_dw" in show
        color = PURPLE if show_dw else ("#64748b" if "weights" in show or focus != "none" else "#334155")
        width = 3 if show_dw else 2
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color=color, width=width),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        if "dE_dw" in show:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            fig.add_annotation(
                x=mx, y=my, text=f"∂E/∂w={fmt(state['dE_dw'][key])}",
                showarrow=False, font=dict(size=10, color=PURPLE), bgcolor=BG,
            )

    xs, ys, colors, texts, sizes = [], [], [], [], []
    for n in NODES:
        active = node_active(focus, n["id"])
        xs.append(n["px"])
        ys.append(n["py"])
        if focus.startswith("back") or focus in ("extra_deriv", "done"):
            colors.append(GREEN if active else "#1e293b")
        elif focus.startswith("forward") or focus in ("activation", "output"):
            colors.append(RED if active else "#1e293b")
        else:
            colors.append(BLUE if active else "#1e293b")
        sizes.append(42 if active else 32)
        bits = [n["name"]]
        i = n["id"]
        if "x" in show and i != 0:
            bits.append(f"x={fmt(state['x'][i])}")
        if "y" in show:
            bits.append(f"y={fmt(state['y'][i])}")
        if "dE_dx" in show and i != 0:
            bits.append(f"∂E/∂x={fmt(state['dE_dx'][i])}")
        if "dE_dy" in show:
            bits.append(f"∂E/∂y={fmt(state['dE_dy'][i])}")
        texts.append("<br>".join(bits))

    fig.add_trace(
        go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(size=sizes, color=colors, line=dict(width=2, color="#e2e8f0")),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    for x, y, t in zip(xs, ys, texts):
        fig.add_annotation(
            x=x, y=y + 0.38, text=t, showarrow=False,
            font=dict(size=11, color="#e2e8f0"), align="center",
        )

    fig.add_annotation(x=0, y=-1.35, text="input", showarrow=False, font=dict(color=GRAY, size=12))
    fig.add_annotation(x=1, y=-1.35, text="hidden 1", showarrow=False, font=dict(color=GRAY, size=12))
    fig.add_annotation(x=2, y=-1.35, text="hidden 2", showarrow=False, font=dict(color=GRAY, size=12))
    fig.add_annotation(x=3, y=-1.35, text="output", showarrow=False, font=dict(color=GRAY, size=12))

    fig.update_layout(
        height=560,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        xaxis=dict(visible=False, range=[-0.45, 3.45]),
        yaxis=dict(visible=False, range=[-1.7, 1.55], scaleanchor="x", scaleratio=1),
    )
    return fig


DEFAULT_W = {
    "w01": 0.8, "w02": -0.6,
    "w13": 0.5, "w23": 0.9, "w14": -0.4, "w24": 0.7,
    "w35": 1.1, "w45": -0.8,
}
DEFAULT_B = {"b1": 0.1, "b2": -0.2, "b3": 0.05, "b4": 0.15, "b5": -0.1}

st.markdown(
    """
    <style>
    .stApp { background: #070b14; color: #e2e8f0; }
    .lesson h2 { color: #f8fafc; margin-bottom: 0.4rem; }
    .kicker { color: #93c5fd; letter-spacing: 0.12em; text-transform: uppercase; font-size: 0.78rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "step" not in st.session_state:
    st.session_state.step = 0

st.sidebar.header("Example")
x_in = st.sidebar.slider("x input", 0.0, 1.0, 0.8, 0.05)
y_target = st.sidebar.slider("y target", 0.0, 1.0, 0.2, 0.05)
alpha = st.sidebar.slider("learning rate α", 0.05, 2.0, 0.8, 0.05)
st.sidebar.caption("Blue x / red y / yellow ∂E/∂x / green ∂E/∂y / purple ∂E/∂w")

state = run_network(x_in, y_target, DEFAULT_W, DEFAULT_B, alpha)
n_steps = len(STEPS)
step = st.session_state.step
step = max(0, min(n_steps - 1, step))
st.session_state.step = step
lesson = STEPS[step]

left, right = st.columns([0.92, 1.18], gap="large")

with left:
    st.markdown('<div class="kicker">Google-style backprop walkthrough</div>', unsafe_allow_html=True)
    st.title("Backpropagation")
    c1, c2, c3 = st.columns(3)
    if c1.button("Previous", use_container_width=True) and st.session_state.step > 0:
        st.session_state.step -= 1
        st.rerun()
    if c2.button("Next", use_container_width=True) and st.session_state.step < n_steps - 1:
        st.session_state.step += 1
        st.rerun()
    if c3.button("Reset", use_container_width=True):
        st.session_state.step = 0
        st.rerun()

    new_step = st.slider("Lesson step", 1, n_steps, step + 1, label_visibility="collapsed")
    if new_step - 1 != step:
        st.session_state.step = new_step - 1
        st.rerun()

    st.subheader(lesson["title"])
    st.markdown(lesson["body"])
    if "error" in lesson["show"]:
        st.metric("Error E", f"{state['error']:.4f}", delta=None)
        st.caption(f"y_out = {state['y'][5]:.3f} · y_target = {y_target:.3f}")
    if lesson["focus"] == "done":
        st.metric("Error after one update", f"{state['error_new']:.4f}")

with right:
    st.plotly_chart(
        draw_network(state, lesson["focus"], lesson["show"]),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.caption("Sticky diagram analogue: the same net stays on the right while the step (scroll position) changes.")
