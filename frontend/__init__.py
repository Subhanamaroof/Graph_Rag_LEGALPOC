"""Streamlit presentation layer for the Graph RAG backend (Story 1.10).

The frontend only talks to the FastAPI backend over HTTP. It never opens ChromaDB, Neo4j or
Groq and never generates answers, citations, scores or graph data itself.
"""
