import matplotlib.pyplot as plt
import io
from PIL import Image
import streamlit as st

def plot_difference_graph(differences):
    fig, ax = plt.subplots()
    ax.plot(differences, label='Diferença por frame')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Diferença')
    ax.set_title('Evolução da Diferença')
    ax.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image = Image.open(buf)
    st.image(image)
    return image
