# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

st.set_page_config(layout="wide")
st.title("Personalized Tumor Progression Simulator")

st.markdown("""
This simulator allows users to explore different tumor trajectories based on predefined combinations of treatment sensitivity, cellular plasticity, and therapeutic strategy. Select one or more scenarios and compare tumor evolution and its cellular subpopulations.

**Legend of cellular subpopulations:**
- **Sensitive:** cells that respond well to treatment.
- **Partially resistant:** cells that partially resist therapeutic effects.
- **Resistant:** cells that do not respond to standard therapy and tend to proliferate.

**Note:** The dashed line represents total tumor volume. The clinical threshold indicates the volume beyond which intensive therapeutic intervention is recommended.
""")

with st.expander("ℹ️ FAQ / Help"):
    st.markdown("""
    **What does each subpopulation represent?**  
    The curves represent the evolution of different cellular subpopulations within the tumor, with varying sensitivity to treatment.

    **What is the clinical threshold?**  
    An arbitrary value indicating when the tumor volume may require a change in therapeutic strategy.

    **Can I change treatments?**  
    This version includes predefined scenarios and standard treatments. For customization, please contact the development team.

    **What is cellular plasticity?**  
    It refers to the ability of tumor cells to switch states (sensitive ↔ resistant), impacting evolution and treatment response.

    **Can I export results?**  
    Options to download graphs and simulations in PDF or CSV will be available soon.
    """)

scenarios = {
    "Scenario 1: Sensitive + High therapy": {
        "mu": [0.9, 0.5, 0.1], "epsilon": 0.8, "transition": 1.0
    },
    "Scenario 2: Partially resistant + Medium therapy": {
        "mu": [0.7, 0.4, 0.2], "epsilon": 0.5, "transition": 1.0
    },
    "Scenario 3: High resistance + Low therapy": {
        "mu": [0.4, 0.3, 0.1], "epsilon": 0.3, "transition": 1.0
    },
    "Scenario 4: Adaptive mix + High plasticity": {
        "mu": [0.8, 0.5, 0.25], "epsilon": 0.6, "transition": 2.0
    },
    "Scenario 5: Experimental control": {
        "mu": [0.6, 0.6, 0.6], "epsilon": 0.0, "transition": 0.0
    },
    "Scenario 6: Heterogeneous tumor + Sequential therapy": {
        "mu": [0.85, 0.6, 0.3], "epsilon": 0.9, "transition": 0.8
    },
    "Scenario 7: Slow adaptive tumor + Reduced plasticity": {
        "mu": [0.75, 0.45, 0.2], "epsilon": 0.4, "transition": 0.5
    }
}

selected_scenarios = st.multiselect(
    "Select one or more scenarios to compare:",
    list(scenarios.keys()),
    default=["Scenario 1: Sensitive + High therapy"]
)

t_max = 300
t_span = (0, t_max)
t_eval = np.linspace(*t_span, 300)
x0 = [0.03, 0.015, 0.005]

base_transition = np.array([
    [0.0,  0.001, 0.0005],
    [0.001, 0.0,  0.001],
    [0.0002, 0.001, 0.0]
])

def tumor_dynamics(t, x, epsilon, mu_profile, transition_matrix):
    r = np.array([0.03, 0.02, 0.015])
    K = 1.0
    total_tumor = np.sum(x)
    treatment_effect = -epsilon * mu_profile * x
    growth = r * x * (1 - total_tumor / K)
    transition = transition_matrix @ x - np.sum(transition_matrix, axis=1) * x
    return growth + treatment_effect + transition

results = {}
for key in selected_scenarios:
    config = scenarios[key]
    transition_matrix = base_transition * config["transition"]
    mu_profile = np.array(config["mu"])
    epsilon = config["epsilon"]
    sol = solve_ivp(
        lambda t, x: tumor_dynamics(t, x, epsilon, mu_profile, transition_matrix),
        t_span, x0, t_eval=t_eval
    )
    results[key] = sol

st.subheader("Comparison of Total Tumor Evolution")
fig1, ax1 = plt.subplots(figsize=(10, 5))
for key in selected_scenarios:
    total = np.sum(results[key].y, axis=0)
    ax1.plot(results[key].t, total, label=key)
ax1.axhline(0.2, color='gray', linestyle='--', label='Clinical threshold')
ax1.set_xlabel("Time (days)")
ax1.set_ylabel("Total tumor volume")
ax1.set_title("Scenario Comparison")
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

st.subheader("Relative Proportion of Subpopulations per Scenario")
fig2, ax2 = plt.subplots(figsize=(10, 5))
for key in selected_scenarios:
    proportions = results[key].y / np.sum(results[key].y, axis=0)
    ax2.plot(results[key].t, proportions[0], label=f"{key} - Sensitive", linestyle='-')
    ax2.plot(results[key].t, proportions[1], label=f"{key} - Partial", linestyle='--')
    ax2.plot(results[key].t, proportions[2], label=f"{key} - Resistant", linestyle=':')
ax2.set_xlabel("Time (days)")
ax2.set_ylabel("Relative proportion")
ax2.set_title("Comparative Clonal Composition")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)
