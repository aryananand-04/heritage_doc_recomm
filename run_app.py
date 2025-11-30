#!/usr/bin/env python3
"""
Streamlit App Launcher
Run this file to start the Heritage Recommender dashboard.
"""

import subprocess
import sys
import os
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()
APP_FILE = PROJECT_ROOT / 'src' / '8_streamlit_app' / 'streamlit_app.py'

if __name__ == '__main__':
    if not APP_FILE.exists():
        print(f"❌ Error: App file not found at {APP_FILE}")
        sys.exit(1)
    
    print(f"🚀 Starting Heritage Recommender Dashboard...")
    print(f"📁 Project root: {PROJECT_ROOT}")
    print(f"📄 App file: {APP_FILE}")
    print()
    
    # Change to project root and run streamlit
    os.chdir(PROJECT_ROOT)
    
    # Try to use venv streamlit if available
    venv_streamlit = PROJECT_ROOT / 'venv' / 'bin' / 'streamlit'
    if venv_streamlit.exists():
        streamlit_cmd = str(venv_streamlit)
        print("✅ Using virtual environment's Streamlit")
    else:
        streamlit_cmd = [sys.executable, '-m', 'streamlit']
        print("⚠️  Using system Streamlit (venv not found)")
    
    # Run streamlit with explicit path
    if isinstance(streamlit_cmd, str):
        cmd = [
            streamlit_cmd, 'run',
            str(APP_FILE.relative_to(PROJECT_ROOT)),
            '--server.port=8501',
            '--server.address=localhost'
        ]
    else:
        cmd = streamlit_cmd + [
            'run',
            str(APP_FILE.relative_to(PROJECT_ROOT)),
            '--server.port=8501',
            '--server.address=localhost'
        ]
    
    print(f"🚀 Starting app at http://localhost:8501")
    print()
    
    subprocess.run(cmd)

