import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import plotly.graph_objects as go
import plotly.express as px

from nltk.corpus import stopwords

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

st.set_page_config(
    page_title="MindWell AI",
    layout="wide"
)

with open("style.css") as f:

    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

st.markdown(
"""
<div class='main-title'>
MINDWELL AI
</div>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<div class='sub-title'>
AI-Based Mental Health Sentiment Monitoring System
</div>
""",
unsafe_allow_html=True
)

@st.cache_resource

def train_model():

    nltk.download('stopwords')

    df = pd.read_csv(
        "Combined Data.csv"
    )

    df = df[['statement','status']]

    df.dropna(inplace=True)

    stop_words = set(
        stopwords.words('english')
    )

    cleaned_text = []

    for text in df['statement']:

        text = text.lower()

        text = re.sub(
            r'[^a-zA-Z\s]',
            '',
            text
        )

        words = text.split()

        filtered = []

        for word in words:

            if word not in stop_words:

                filtered.append(word)

        cleaned = " ".join(filtered)

        cleaned_text.append(cleaned)

    df['cleaned_text'] = cleaned_text

    tokenizer = Tokenizer(
        num_words=10000
    )

    tokenizer.fit_on_texts(
        df['cleaned_text']
    )

    sequences = tokenizer.texts_to_sequences(
        df['cleaned_text']
    )

    X = pad_sequences(
        sequences,
        maxlen=100,
        padding='post'
    )

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(
        df['status']
    )

    X_train,X_test,y_train,y_test = train_test_split(

        X,
        y,

        test_size=0.2,

        random_state=42
    )

    model = Sequential()

    model.add(

        Embedding(

            input_dim=10000,

            output_dim=128,

            input_length=100
        )
    )

    model.add(

        LSTM(
            128
        )
    )

    model.add(

        Dropout(0.3)
    )

    model.add(

        Dense(
            64,
            activation='relu'
        )
    )

    model.add(

        Dropout(0.3)
    )

    model.add(

        Dense(
            len(label_encoder.classes_),
            activation='softmax'
        )
    )

    model.compile(

        optimizer='adam',

        loss='sparse_categorical_crossentropy',

        metrics=['accuracy']
    )

    model.fit(

        X_train,
        y_train,

        validation_split=0.2,

        epochs=5,

        batch_size=64,

        verbose=0
    )

    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = train_model()

left,right = st.columns([2,1])

with left:

    st.markdown(
    """
    <div class='glass-card'>

    <h2>About the Project</h2>

    <p>
    This AI-powered Mental Health Monitoring System analyzes emotional sentiment patterns using Natural Language Processing and LSTM-based Recurrent Neural Networks.
    </p>

    <ul>
    <li>Detect emotional stress patterns</li>
    <li>Monitor depression and anxiety indicators</li>
    <li>Generate AI-based emotional analysis</li>
    <li>Provide emotional wellness guidance</li>
    </ul>

    </div>
    """,
    unsafe_allow_html=True
    )

with right:

    st.markdown(
    """
    <div class='glass-card'>

    <h2>AI Dashboard</h2>

    <h1 style='color:#00ffcc'>
    LIVE ANALYSIS
    </h1>

    <p>
    LSTM + NLP + Deep Learning
    </p>

    </div>
    """,
    unsafe_allow_html=True
    )

st.markdown(
"## Emotion Analysis"
)

sample_sentences = [

    "I feel depressed and lonely",

    "I am happy today",

    "I feel anxious and stressed",

    "Nobody understands my pain",

    "I am excited for my future"

]

selected = st.selectbox(

    "Sample Emotional Sentences",

    sample_sentences
)

user_input = st.text_area(

    "Enter your thoughts or feelings here...",

    value=selected,

    height=180
)

if st.button("Analyze Emotion"):

    processed = user_input.lower()

    processed = re.sub(
        r'[^a-zA-Z\s]',
        '',
        processed
    )

    sequence = tokenizer.texts_to_sequences(
        [processed]
    )

    padded = pad_sequences(

        sequence,

        maxlen=100,

        padding='post'
    )

    prediction = model.predict(
        padded,
        verbose=0
    )

    probs = prediction[0]

    predicted_class = np.argmax(
        probs
    )

    emotion = label_encoder.inverse_transform(
        [predicted_class]
    )[0]

    confidence = float(
        np.max(probs) * 100
    )

    emotion_lower = emotion.lower()

    if emotion_lower in [

        "depression",
        "sad",
        "anxiety",
        "stress"

    ]:

        risk = "HIGH RISK"

        emoji = "😟"

        color = "#ff4b4b"

        guidance = "Take a short break and avoid isolation."

        activity = "Go for a walk and listen to calming music."

        tip = "Talk with trusted friends or family."

    elif emotion_lower in [

        "normal",
        "neutral"

    ]:

        risk = "STABLE"

        emoji = "🙂"

        color = "#00c6ff"

        guidance = "Your emotional state appears balanced."

        activity = "Maintain healthy daily activities."

        tip = "Practice mindfulness and gratitude."

    else:

        risk = "POSITIVE"

        emoji = "😊"

        color = "#00ff99"

        guidance = "Positive emotional patterns detected."

        activity = "Continue activities you enjoy."

        tip = "Maintain positive social interaction."

    c1,c2,c3 = st.columns(3)

    with c1:

        st.markdown(
        f"""
        <div class='metric-card'>

        <div class='metric-title'>
        CURRENT EMOTION
        </div>

        <div style='font-size:60px'>
        {emoji}
        </div>

        <div class='metric-value'
        style='color:{color};'>
        {emotion.upper()}
        </div>

        <div class='metric-sub'>
        Emotional Category
        </div>

        </div>
        """,
        unsafe_allow_html=True
        )

    with c2:

        st.markdown(
        f"""
        <div class='metric-card'>

        <div class='metric-title'>
        CONFIDENCE SCORE
        </div>

        <div class='metric-value'>
        {confidence:.2f}%
        </div>

        <div class='metric-sub'>
        AI Confidence Level
        </div>

        </div>
        """,
        unsafe_allow_html=True
        )

    with c3:

        st.markdown(
        f"""
        <div class='metric-card'>

        <div class='metric-title'>
        RISK ASSESSMENT
        </div>

        <div class='metric-value'
        style='color:{color};'>
        {risk}
        </div>

        <div class='metric-sub'>
        Emotional Monitoring
        </div>

        </div>
        """,
        unsafe_allow_html=True
        )

    left,right = st.columns([2,1])

    with left:

        fig = go.Figure(

            data=[

                go.Bar(

                    x=label_encoder.classes_,

                    y=probs
                )

            ]
        )

        fig.update_layout(

            title="Emotion Probability Distribution",

            template="plotly_dark",

            height=400
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        pie = px.pie(

            names=label_encoder.classes_,

            values=probs,

            title="Sentiment Contribution"
        )

        pie.update_layout(

            template="plotly_dark",

            height=400
        )

        st.plotly_chart(
            pie,
            use_container_width=True
        )

    g1,g2 = st.columns(2)

    with g1:

        st.markdown(
        f"""
        <div class='glass-card'>

        <h2>Emotional Guidance</h2>

        <p>{guidance}</p>

        </div>
        """,
        unsafe_allow_html=True
        )

    with g2:

        st.markdown(
        f"""
        <div class='glass-card'>

        <h2>Wellness Recommendations</h2>

        <p><b>Positive Activity:</b><br>{activity}</p>

        <p><b>Wellness Tip:</b><br>{tip}</p>

        </div>
        """,
        unsafe_allow_html=True
        )