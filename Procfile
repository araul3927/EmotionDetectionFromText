web: sh setup.sh && streamlit run frontend.py
api: uvicorn backend:app --host=0.0.0.0 --port=${PORT:-8000} 