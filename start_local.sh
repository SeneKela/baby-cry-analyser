#!/bin/bash

# Function to open a new terminal tab and run a command
open_tab() {
    local cmd="$1"
    local title="$2"
    osascript -e "tell application \"Terminal\" to do script \"$cmd\""
}

echo "Starting CrySense AI locally..."

# Get absolute path
ROOT_DIR=$(pwd)

# Start Backend
echo "Starting Backend..."
BACKEND_CMD="cd '$ROOT_DIR/code/web/backend' && python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"
open_tab "$BACKEND_CMD" "CrySense Backend"

# Start Frontend
echo "Starting Frontend..."
FRONTEND_CMD="cd '$ROOT_DIR/code/web/frontend' && npm run dev"
open_tab "$FRONTEND_CMD" "CrySense Frontend"

echo "Both services are starting in new Terminal windows."
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:5173"
