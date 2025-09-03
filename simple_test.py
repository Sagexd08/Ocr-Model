import streamlit as st

st.title("🔍 Simple OCR Test")
st.write("Testing Streamlit functionality...")

if st.button("Click me!"):
    st.success("✅ Streamlit is working!")
    
st.write("Current working directory:", st.session_state.get('cwd', 'Not set'))

# Test file upload
uploaded_file = st.file_uploader("Test file upload", type=['pdf', 'png', 'jpg'])
if uploaded_file:
    st.write(f"File uploaded: {uploaded_file.name}")
    st.write(f"File size: {uploaded_file.size} bytes")
