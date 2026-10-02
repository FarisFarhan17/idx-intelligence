"""Entry point: run the IDX Intelligence backend + frontend server.

    python run.py

Then open the printed URL (default http://127.0.0.1:5000).
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_BACKEND", os.getenv("KERAS_BACKEND", "tensorflow"))

from backend.app import main  # noqa: E402

if __name__ == "__main__":
    main()
