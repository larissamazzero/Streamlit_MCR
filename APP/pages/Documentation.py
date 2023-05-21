import streamlit as st
st.set_page_config(
    page_title="MCR-ALS",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)
header = st.container()
content = st.container()

with header:
    st.title("Documentation")

    st.markdown('* Documentação completa e como utilizar o código')
