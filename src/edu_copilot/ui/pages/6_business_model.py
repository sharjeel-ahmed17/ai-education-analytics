import os
import streamlit as st

st.set_page_config(layout="wide")

# Locate target markdown file in the project workspace
current_dir = os.path.dirname(os.path.abspath(__file__))
docs_path = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "..", "docs", "business_model.md"))

if os.path.exists(docs_path):
    with open(docs_path, "r", encoding="utf-8") as f:
        markdown_content = f.read()
    st.markdown(markdown_content)
else:
    st.error(f"Error: Business model document could not be resolved at {docs_path}")
