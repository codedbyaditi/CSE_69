import uvicorn
import os
import sys
import traceback

if __name__ == "__main__":
    # Ensure the backend directory is in the python path
    backend_path = os.path.join(os.path.dirname(__file__), "backend")
    if backend_path not in sys.path:
        sys.path.append(backend_path)

    print("Starting Substation Bot Server...")
    print("Attempting import of application module ...")
    try:
        # Test import early to surface errors
        from backend.app.main import app  # noqa: F401
        print("Import successful. Launching Uvicorn.")
    except Exception:
        print("ERROR: Failed to import FastAPI app. Traceback:")
        traceback.print_exc()
        sys.exit(1)

    # Use 0.0.0.0 to allow access from other devices if needed
    print("Open http://127.0.0.1:8080 (or http://<your-ip>:8080) in your browser")
    try:
        # Run without reload to avoid continuous restart/shutdown loop in this environment
        uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8080, reload=False)
    except Exception:
        print("ERROR: Uvicorn failed to start. Traceback:")
        traceback.print_exc()
        sys.exit(1)
