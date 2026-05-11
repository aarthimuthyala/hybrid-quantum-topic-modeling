from fastapi import APIRouter

router = APIRouter()

@router.post("/load-dataset")
async def load_dataset(payload: dict):

    return {
        "status": "success",
        "message": "Dataset loaded successfully",
        "dataset_id": payload.get("corpus_id"),
        "topics": [
            ["ai", "machine", "learning"],
            ["quantum", "optimization", "qaoa"]
        ],
        "metrics": {
            "coherence": 0.82,
            "silhouette_score": 0.76
        }
    }