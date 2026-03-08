# ─────────────────────────────────────────────
# app.py — Root level entry point
# Required by Hugging Face Spaces
# This file launches the Streamlit app
# ─────────────────────────────────────────────

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Launch the Hugging Face optimised app
exec(open("frontend/app_hf.py").read())