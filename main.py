### 3. app/main.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ask")
def ask(query: str = Query(..., description="Câu hỏi tuyển sinh")):
    reply = model.ask(query)
    return {"answer": reply}