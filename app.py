import streamlit as st

st.set_page_config(page_title="Test App", layout="wide")

st.title("Test App")
st.write("If you see this, the app is running correctly.")

col1, col2 = st.columns(2)
with col1:
    st.selectbox("Choose", ["A", "B", "C"], key="test")
with col2:
    st.write("Hello")

if st.button("Click me"):
    st.success("Button clicked!")
