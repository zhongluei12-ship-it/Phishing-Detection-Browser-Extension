import streamlit as st
import machine_learning as ml
import feature_extraction as fe
from bs4 import BeautifulSoup
import requests as re
import matplotlib.pyplot as plt


# ---------------------------
# PAGE SETTINGS
# ---------------------------
st.set_page_config(
    page_title="Phishing Website Detector",
    layout="wide",
)

st.title("🔍 Phishing Website Detection (Content-Based ML)")
st.write(
    "This educational project detects phishing **using only webpage content (HTML)**, "
    "without relying on URL-based features."
)


# ---------------------------
# PROJECT DETAILS SECTION
# ---------------------------
with st.expander("📘 PROJECT DETAILS"):
    st.subheader("Approach")
    st.write(
        """
        - Supervised learning using 7 scikit-learn classifiers  
        - Dataset built using **Tranco (legit)** and **PhishTank (phishing)**  
        - Features extracted with BeautifulSoup from HTML content  
        - Content-based features only — *no URL heuristics used*  
        """
    )

    st.subheader("Dataset Summary")
    phishing_count = ml.phishing_df.shape[0]
    legit_count = ml.legitimate_df.shape[0]
    total = phishing_count + legit_count

    labels = ["Phishing", "Legitimate"]
    sizes = [
        phishing_count / total * 100,
        legit_count / total * 100,
    ]

    fig, ax = plt.subplots()
    ax.pie(
        sizes,
        labels=labels,
        explode=(0.1, 0),
        startangle=90,
        autopct="%1.1f%%",
        shadow=True,
    )
    ax.axis("equal")
    st.pyplot(fig)

    st.write("### Sample Data (Legitimate)")
    row_limit = st.slider("Show first n rows:", 0, 100)
    st.dataframe(ml.legitimate_df.head(row_limit))

    st.write("---")
    st.subheader("Download Full Combined Dataset")

    @st.cache_data
    def convert_df(df):
        return df.to_csv().encode("utf-8")

    csv = convert_df(ml.df)

    st.download_button(
        "Download CSV",
        csv,
        file_name="phishing_legitimate_dataset.csv",
        mime="text/csv",
    )

    st.subheader("ML Results")
    st.table(ml.df_results)

    st.caption("Models: NB, SVM, DT, RF, AdaBoost, Neural Network, KNN")


# ---------------------------
# MODEL SELECTION
# ---------------------------
st.write("## 🧠 Choose Machine Learning Model")

model_options = {
    "Gaussian Naive Bayes": ml.nb_model,
    "Support Vector Machine": ml.svm_model,
    "Decision Tree": ml.dt_model,
    "Random Forest": ml.rf_model,
    "AdaBoost": ml.ab_model,
    "Neural Network": ml.nn_model,
    "K-Neighbours": ml.kn_model,
}

choice = st.selectbox("Select a model:", list(model_options.keys()))
model = model_options[choice]
st.success(f"✔️ Selected Model: **{choice}**")


# ---------------------------
# URL INPUT + PREDICTION
# ---------------------------
st.write("## 🌐 Check a Website")

url = st.text_input("Enter a URL (with or without https://)")

if st.button('Check!'):
    try:
        response = re.get(url, verify=False, timeout=4)

        if response.status_code != 200:
            st.warning(f"⚠️ Unable to fetch page. HTTP Status: {response.status_code}")
        else:
            soup = BeautifulSoup(response.content, "html.parser")
            vector = [fe.create_vector(soup)]
            result = model.predict(vector)

            # -----------------------------------
            # OPTION B — Centered Badge UI
            # -----------------------------------
            if result[0] == 0:
                st.markdown(
                    """
                    <div style="text-align:center; margin-top:30px;">
                        <div style="
                            display:inline-block;
                            background:#28a745;
                            padding:18px 40px;
                            border-radius:60px;
                            color:white;
                            font-size:26px;
                            font-weight:bold;
                            box-shadow:0 4px 12px rgba(40,167,69,0.3);
                        ">
                            🟢 LEGITIMATE WEBSITE
                        </div>
                        <p style="font-size:18px; color:#155724; margin-top:15px;">
                            This webpage appears to be safe.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:
                st.markdown(
                    """
                    <div style="text-align:center; margin-top:30px;">
                        <div style="
                            display:inline-block;
                            background:#dc3545;
                            padding:18px 40px;
                            border-radius:60px;
                            color:white;
                            font-size:26px;
                            font-weight:bold;
                            box-shadow:0 4px 12px rgba(220,53,69,0.3);
                        ">
                            🔴 PHISHING DETECTED
                        </div>
                        <p style="font-size:18px; color:#721c24; margin-top:15px;">
                            This webpage is likely a phishing page.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    except re.exceptions.RequestException as e:
        st.error(f"Request error: {e}")
