import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

# 1. Page Config
st.set_page_config(page_title="AI Generation Detector", page_icon="🧠", layout="wide")

st.title("🧠 OKCupid: Advanced Generation Predictor")

col_img1, col_img2, col_img3 = st.columns([1, 2, 1]) 

with col_img2:
    st.image("https://blog.photofeeler.com/wp-content/uploads/2018/12/okcupid-search.jpg", width=400)


# 2. Asset Loading
@st.cache_resource
def load_full_model():
    # Modelin içinde Normalization katmanı olduğunu varsayıyoruz (önceki önerimdeki gibi)
    return tf.keras.models.load_model('okcupid_model2.keras')

try:
    model = load_full_model()
except Exception as e:
    st.error(f"Model Load Error: {e}")

# 3. User Interface (Feature Collection)
with st.expander("📊 Personal & Lifestyle Metrics", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 18, 100, 25)
        sex = st.selectbox("Gender", ["m", "f"])
        job = st.selectbox("Job Sector", ["computer", "banking", "clerical", "construction", "education", "artistic", "other"])
    with c2:
        drinks = st.selectbox("Drinking", ["socially", "rarely", "often", "never"])
        smokes = st.selectbox("Smoking", ["never", "sometimes", "no", "when drinking"])
        status = st.selectbox("Relationship", ["single", "available", "seeing someone"])
    with c3:
        body_type = st.selectbox("Body Type", ["average", "fit", "athletic", "thin", "curvy"])
        diet = st.selectbox("Diet", ["anything", "mostly anything", "vegetarian", "vegan"])
        essay_len = st.number_input("Total Essay Length", 0, 10000, 500)

with st.expander("📝 Interest & Keywords (Text Features)"):
    st.write("Does your profile mention these?")
    c4, c5, c6 = st.columns(3)
    yoga = c4.checkbox("Yoga")
    writing = c5.checkbox("Writing / Creative")
    banking = c6.checkbox("Finance / Banking")
    working = c4.checkbox("Career Focused (Working)")
    world = c5.checkbox("Travel / World")

# 4. Data Processing & Alignment
if st.button("🚀 Predict My Generation"):
    try:
        # ÖNEMLİ: Model 10219 sütun beklediği için boş bir şablon oluşturuyoruz
        total_features = 10219
        input_vector = np.zeros((1, total_features))

        # Elimizdeki verileri ilgili indekslere yerleştiriyoruz
        # NOT: Bu indeksler eğitim setindeki (X.columns) sıralama ile birebir aynı olmalı!
        # Örnek manuel eşleştirme:
        input_vector[0, 0] = age
        input_vector[0, 1] = 1 if sex == "f" else 0
        input_vector[0, 2] = essay_len
        
        # Kelime ve Job özellikleri (Örnek indeksler - Eğitim setine göre güncellemelisin)
        if yoga: input_vector[0, 100] = 1
        if writing: input_vector[0, 101] = 1
        if working: input_vector[0, 102] = 1
        if job == "computer": input_vector[0, 50] = 1
        if job == "banking": input_vector[0, 51] = 1

        # 5. Prediction
        prediction = model.predict(input_vector)
        class_idx = np.argmax(prediction)
        
        # Sonuç görselleştirme
        gens = {0: "Gen X (Classic & Experienced)", 1: "Millennial (The Digital Bridge)", 2: "Gen Z (Digital Native)"}
        colors = {0: "orange", 1: "blue", 2: "green"}
        
        st.markdown(f"### Result: :{colors[class_idx]}[{gens[class_idx]}]")
        st.progress(float(np.max(prediction)))

    except Exception as e:
        st.error(f"Prediction Failed: {e}")
        st.info("Tip: Make sure the input shape matches 10219.")