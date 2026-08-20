"""
PhishGuard :: run the web app
--------------------------------
Usage:
    python3 run_app.py
Then open http://localhost:5000 in a browser.
"""
from src.app import app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
