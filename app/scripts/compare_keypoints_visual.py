import matplotlib.pyplot as plt
import streamlit as st

def plot_difference_graph(difference_lists):
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, valores in enumerate(difference_lists):
        ax.plot(valores, label=f"Parte {i+1}")

    ax.set_title("Diferença de movimento por parte do corpo")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Diferença (score)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

