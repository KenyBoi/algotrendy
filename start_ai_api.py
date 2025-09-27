#!/usr/bin/env python3
"""
AI Orchestrator API Server Startup Script

This script starts the FastAPI server for the AI Orchestrator Module.
Run this from the project root directory.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Start the AI Orchestrator API server"""

    # Change to src directory
    src_dir = Path(__file__).parent / "src"
    os.chdir(src_dir)

    print("🤖 Starting AI Orchestrator API Server...")
    print("=" * 50)
    print(f"📁 Working directory: {src_dir}")
    print("🌐 API will be available at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")
    print("🔄 Press Ctrl+C to stop the server")
    print("=" * 50)

    try:
        # Start the server
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "ai_orchestrator_api:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--log-level", "info"
        ], check=True)

    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed to start: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()