web: sh setup.sh && streamlit run frontend.py
worker: gunicorn -w 4 -k uvicorn.workers.UvicornWorker backend:app 