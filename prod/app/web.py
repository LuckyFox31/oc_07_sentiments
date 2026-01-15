import streamlit as st
import requests
import json

# Configuration de la page
st.set_page_config(
    page_title="Analyse de Sentiment",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("🎭 Analyse de Sentiment de Tweets")
st.markdown("---")

# Sidebar pour la configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_url = st.text_input(
        "URL de l'API",
        value="http://localhost:8000",
        help="Adresse de l'API FastAPI"
    )

    st.markdown("---")
    st.subheader("ℹ️ À propos")
    st.info(
        """
        Cette application utilise un modèle de Deep Learning
        (LSTM + Word2Vec) pour analyser le sentiment de tweets.

        **Modèle**: Bidirectional LSTM
        **Embedding**: Word2Vec (100d)
        **Précision**: ~80% sur le jeu de test
        """
    )

    st.markdown("---")
    st.subheader("📊 Interprétation")
    st.markdown(
        """
        - **😊 Positif**: Score ≥ 0.5
        - **😞 Négatif**: Score < 0.5
        - **Confiance**: Certitude du modèle (0-100%)
        """
    )

# Exemples de phrases pour tester rapidement
st.subheader("💡 Exemples de phrases")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("😊 Phrase Positive", use_container_width=True):
        st.session_state.text_input = "I love this amazing product! It's wonderful! :)"

with col2:
    if st.button("😞 Phrase Négative", use_container_width=True):
        st.session_state.text_input = "This is terrible and I hate it :("

with col3:
    if st.button("😐 Phrase Neutre", use_container_width=True):
        st.session_state.text_input = "I went to the store today"

st.markdown("---")

# Zone de saisie de texte
st.subheader("✍️ Saisissez votre texte")
text_input = st.text_area(
    "Entrez votre phrase ou tweet à analyser :",
    value=st.session_state.get('text_input', ''),
    height=150,
    placeholder="Exemple : I love this amazing product! :)",
    help="Le texte sera nettoyé automatiquement (suppression des URLs, mentions, hashtags...)"
)

# Bouton d'analyse
analyze_button = st.button("🔍 Analyser le sentiment", type="primary", use_container_width=True)

# Traitement de la requête
if analyze_button:
    if not text_input or text_input.strip() == "":
        st.error("⚠️ Veuillez saisir un texte avant d'analyser.")
    else:
        with st.spinner("🔄 Analyse en cours..."):
            try:
                # Envoyer la requête à l'API
                response = requests.post(
                    f"{api_url}/predict",
                    json={"text": text_input},
                    timeout=10
                )

                if response.status_code == 200:
                    result = response.json()

                    st.markdown("---")
                    st.subheader("📊 Résultats de l'analyse")

                    # Affichage du sentiment avec un grand emoji
                    sentiment = result['sentiment']
                    confidence = result['confidence']
                    score = result['score']

                    if sentiment == "positif":
                        emoji = "😊"
                        color = "green"
                        sentiment_text = "POSITIF"
                    else:
                        emoji = "😞"
                        color = "red"
                        sentiment_text = "NÉGATIF"

                    # Colonnes pour l'affichage
                    col1, col2, col3 = st.columns([1, 2, 1])

                    with col1:
                        st.markdown(f"<h1 style='text-align: center; font-size: 100px;'>{emoji}</h1>", unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"<h2 style='text-align: center; color: {color};'>{sentiment_text}</h2>", unsafe_allow_html=True)
                        st.metric(
                            label="Confiance du modèle",
                            value=f"{confidence * 100:.2f}%",
                            delta=None
                        )

                    with col3:
                        st.metric(
                            label="Score brut",
                            value=f"{score:.4f}",
                            delta=None
                        )

                    # Barre de progression pour visualiser le score
                    st.markdown("### 📈 Visualisation du score")
                    st.progress(score, text=f"Score : {score:.4f} (0 = Négatif, 1 = Positif)")

                    # Affichage du texte original et des détails
                    with st.expander("📝 Détails de l'analyse"):
                        st.markdown(f"**Texte original :**")
                        st.text(result['text'])

                        st.markdown(f"**Interprétation :**")
                        if sentiment == "positif":
                            st.success(f"Le modèle est confiant à {confidence * 100:.2f}% que ce texte exprime un sentiment positif.")
                        else:
                            st.error(f"Le modèle est confiant à {confidence * 100:.2f}% que ce texte exprime un sentiment négatif.")

                        st.markdown(f"**Note technique :**")
                        st.info(
                            f"""
                            - Le texte a été nettoyé (suppression URLs, mentions, hashtags)
                            - Lemmatisation effectuée avec NLTK
                            - Score brut du modèle : {score:.4f}
                            - Seuil de classification : 0.5
                            """
                        )

                else:
                    st.error(f"❌ Erreur {response.status_code} : {response.text}")

            except requests.exceptions.ConnectionError:
                st.error(
                    """
                    ❌ **Impossible de se connecter à l'API**

                    Vérifiez que l'API FastAPI est bien démarrée :
                    ```
                    uvicorn api:app --reload --host 0.0.0.0 --port 8000
                    ```
                    """
                )
            except requests.exceptions.Timeout:
                st.error("❌ La requête a expiré. L'API met trop de temps à répondre.")
            except Exception as e:
                st.error(f"❌ Erreur inattendue : {str(e)}")

# Pied de page
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>🤖 Développé avec FastAPI, Streamlit et TensorFlow | Modèle : LSTM + Word2Vec</p>
    </div>
    """,
    unsafe_allow_html=True
)
