from fastapi import FastAPI

app = FastAPI(
    title="Daedalus",
    version="0.1.0",
)

@app.get("/")
def root():
    return {"message": "Daedalus is running"}

@app.get("/")
def health():
    return {"status":"ok"}

#Logger
from daedalus.core import configure_logging
import logging

configure_logging()