web: gunicorn AI_backend.AI_backend.wsgi --bind 0.0.0.0:$PORT
worker: python AI_backend/AI_Logic/daemon_sched.py
