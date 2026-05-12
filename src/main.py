from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "HQC Backend Running Successfully"}

@app.get("/health")
def health():
    return {"status": "online"}