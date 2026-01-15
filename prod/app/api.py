import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from cleaning import clean_text
import nltk

MAX_LEN = 30
MODEL_PATH = "model_w2v_03.keras"
TOKENIZER_PATH = "tokenizer.pickle"

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

app = FastAPI(
    title="API d'Analyse de Sentiment",
    description="API pour analyser le sentiment de tweets avec un modèle LSTM + Word2Vec",
    version="1.0.0"
)

model = None
tokenizer = None

@app.on_event("startup")
async def load_model_and_tokenizer():
    """Charger le modèle et le tokenizer au démarrage de l'application"""
    global model, tokenizer

    try:
        print("Chargement du modèle TensorFlow...")
        # Charger le modèle sans compiler pour éviter les problèmes de compatibilité
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)

        # Recompiler le modèle avec les mêmes paramètres que l'entraînement
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        print(f"✅ Modèle chargé et recompilé depuis {MODEL_PATH}")

        print("Chargement du tokenizer...")
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        print(f"✅ Tokenizer chargé depuis {TOKENIZER_PATH}")

    except Exception as e:
        print(f"❌ Erreur lors du chargement : {str(e)}")
        raise

class PredictRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "I love this amazing product! :)"
            }
        }

class PredictResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    score: float

    class Config:
        json_schema_extra = {
            "example": {
                "text": "I love this amazing product! :)",
                "sentiment": "positif",
                "confidence": 0.85,
                "score": 0.85
            }
        }

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

@app.get("/", tags=["Root"])
async def root():
    """Endpoint racine pour vérifier que l'API fonctionne"""
    return {
        "message": "API d'Analyse de Sentiment - Utilisez /docs pour voir la documentation",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Vérifier l'état de santé de l'API"""
    return {
        "status": "healthy",
        "model_loaded": model is not None and tokenizer is not None
    }

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_sentiment(request: PredictRequest):
    """
    Analyser le sentiment d'un texte

    - **text**: Le texte à analyser (tweet ou phrase courte)

    Retourne le sentiment (positif/négatif), le score brut et la confiance
    """

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Le modèle ou le tokenizer n'est pas chargé. Veuillez réessayer plus tard."
        )

    if not request.text or request.text.strip() == "":
        raise HTTPException(
            status_code=400,
            detail="Le texte ne peut pas être vide"
        )

    try:
        tokens = clean_text(request.text, processing="lemmatizer")

        if not tokens or len(tokens) == 0:
            raise HTTPException(
                status_code=400,
                detail="Le texte ne contient aucun mot valide après nettoyage"
            )

        text_cleaned = " ".join(tokens)

        sequences = tokenizer.texts_to_sequences([text_cleaned])

        padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

        prediction = model.predict(padded, verbose=0)
        score = float(prediction[0][0])

        if score >= 0.5:
            sentiment = "positif"
            confidence = score
        else:
            sentiment = "négatif"
            confidence = 1 - score

        return PredictResponse(
            text=request.text,
            sentiment=sentiment,
            confidence=round(confidence, 4),
            score=round(score, 4)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction : {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
