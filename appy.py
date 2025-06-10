code = """# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

st.set_page_config(layout="wide")
st.title("Simulador de Progresión Tumoral Personalizada")

st.markdown(\"\"\"
Este simulador permite explorar distintas trayectorias tumorales según combinaciones predefinidas de sensibilidad al tratamiento, plasticidad celular y estrategia terapéutica. Seleccione uno o varios escenarios y compare la evolución del tumor y sus subpoblaciones celulares.

**Leyenda de subpoblaciones celulares:**
- **Sensible:** células que responden bien al tratamiento.
- **Parcialmente resistente:** células que resisten parcialmente los efectos terapéuticos.
- **Resistente:** células que no responden a la terapia estándar y tienden a proliferar.

**Nota:** La línea discontinua representa el volumen tumoral total. El umbral clínico indica el volumen a partir del cual se recomienda intervención terapéutica intensiva.
\"\"\")

with st.expander("ℹ️ Preguntas frecuentes / Ayuda"):
    st.markdown(\"\"\"
    **¿Qué representa cada subpoblación?**  
    Las curvas representan la evolución de subpoblaciones celulares dentro del tumor con distinta sensibilidad al tratamiento. 

    **¿Qué significa el umbral clínico?**  
    Es un valor arbitrario que representa el punto en el que el volumen tumoral puede requerir un cambio en la estrategia terapéutica.

    **¿Puedo cambiar los tratamientos?**  
    Esta versión está diseñada con escenarios preconfigurados y tratamientos estándar. Para personalizaciones, contactar con el equipo desarrollador.

    **¿Qué significa la plasticidad celular?**  
    Es la capacidad de las células tumorales de cambiar entre estados (sensibles ↔ resistentes), afectando la evolución y respuesta al tratamiento.

    **¿Puedo exportar los resultados?**  
    Próximamente se habilitará una opción para descargar los gráficos y simulaciones en PDF o CSV.
    \"\"\")

scenarios = {
    "Escenario 1: Sensible + Terapia alta": {
        "mu": [0.9, 0.5, 0.1], "epsilon": 0.8, "transition": 1.0
    },
    "Escenario 2: Parcialmente resistente + Terapia media": {
        "mu": [0.7, 0.4, 0.2], "epsilon": 0.5, "transition": 1.0
    },
    "Escenario 3: Alta resistencia + Terapia baja": {
        "mu": [0.4, 0.3, 0.1], "epsilon": 0.3, "transition": 1.0
    },
    "Escenario 4: Mixto adaptado + plasticidad alta": {
        "mu": [0.8, 0.5, 0.25], "epsilon": 0.6, "transition": 2.0
    },
    "Escenario 5: Control experimental": {
        "mu": [0.6, 0.6, 0.6], "epsilon": 0.0, "transition": 0.0
    },
    "Escenario 6: Tumor heterogéneo + Terapia secuencial": {
        "mu": [0.85, 0.6, 0.3], "epsilon": 0.9, "transition": 0.8
    },
    "Escenario 7: Tumor adaptativo lento + Plasticidad reducida": {
        "mu": [0.75, 0.45, 0.2], "epsilon": 0.4, "transition": 0.5
    }
}

selected_scenarios = st.multiselect(
    "Seleccione uno o más escenarios para comparar:",
    list(scenarios.keys()),
    default=["Escenario 1: Sensible + Terapia alta"]
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

st.subheader("Comparación de la evolución del tumor total")
fig1, ax1 = plt.subplots(figsize=(10, 5))
for key in selected_scenarios:
    total = np.sum(results[key].y, axis=0)
    ax1.plot(results[key].t, total, label=key)
ax1.axhline(0.2, color='gray', linestyle='--', label='Umbral clínico')
ax1.set_xlabel("Tiempo (días)")
ax1.set_ylabel("Volumen tumoral total")
ax1.set_title("Comparación entre escenarios")
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

st.subheader("Proporción relativa de subpoblaciones en cada escenario")
fig2, ax2 = plt.subplots(figsize=(10, 5))
for key in selected_scenarios:
    proportions = results[key].y / np.sum(results[key].y, axis=0)
    ax2.plot(results[key].t, proportions[0], label=f"{key} - Sensible", linestyle='-')
    ax2.plot(results[key].t, proportions[1], label=f"{key} - Parcial", linestyle='--')
    ax2.plot(results[key].t, proportions[2], label=f"{key} - Resistente", linestyle=':')
ax2.set_xlabel("Tiempo (días)")
ax2.set_ylabel("Proporción relativa")
ax2.set_title("Composición clonal comparativa")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)
"""

output_path.write_text(code)
output_path
