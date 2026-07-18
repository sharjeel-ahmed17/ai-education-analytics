import os
import streamlit as st
import streamlit.components.v1 as components

st.title("Interactive Custom UI Dashboard")
st.markdown("This page renders the custom HTML/CSS/JS frontend mockup of **EduCopilot** inside a Streamlit component container.")

# Paths to the workspace root static files
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", "..", ".."))

html_path = os.path.join(project_root, "index.html")
css_path = os.path.join(project_root, "style.css")
js_path = os.path.join(project_root, "app.js")

# Double check if files exist
if os.path.exists(html_path) and os.path.exists(css_path) and os.path.exists(js_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    with open(css_path, "r", encoding="utf-8") as f:
        css_content = f.read()
    with open(js_path, "r", encoding="utf-8") as f:
        js_content = f.read()
        
    # Inject CSS inline
    html_content = html_content.replace(
        '<link rel="stylesheet" href="style.css">',
        f'<style>{css_content}</style>'
    )
    
    # Inject JS inline
    html_content = html_content.replace(
        '<script src="app.js"></script>',
        f'<script>{js_content}</script>'
    )
    
    # Render component
    components.html(html_content, height=850, scrolling=True)
else:
    st.error("Error: Could not locate `index.html`, `style.css`, or `app.js` in the project root folder.")
    st.info(f"Searched in: `{project_root}`")
