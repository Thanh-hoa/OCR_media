#!/bin/bash
case "$1" in
  dev)
    uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
    ;;
  install)
    pip install -r requirements.txt
    ;;
  health)
    curl -s http://127.0.0.1:8000/health
    ;;
  *)
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
    ;;
esac
