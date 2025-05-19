from fastapi import FastAPI, Query
import model

app = FastAPI()

@app.get("/")
def root():
    return {"message": "API is live"}

@app.get("/ask")
def ask(query: str = Query(...)):
    return {"answer": model.ask(query)}
