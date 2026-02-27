import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
st.set_page_config(
    page_title="AI Spam Email Detector",
    page_icon="📧",
    layout="centered"
)
st.markdown("""
<style>
.main {
    background-color: #0f172a;
}
h1, h2, h3 {
    color: #22c55e;
    text-align: center;
}
.stTextArea textarea {
    background-color: #020617;
    color: white;
}
.result-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
}
.spam {
    background-color: #7f1d1d;
    color: #fecaca;
}
.notspam {
    background-color: #052e16;
    color: #bbf7d0;
}
</style>
""", unsafe_allow_html=True)
st.title("📧 AI Spam Email Detector")
st.write("Detect whether an email is **Spam or Not Spam** using Machine Learning")
emails = [
"Congratulations! You’ve won a free iPhone",
"Claim your lottery prize now",
"Exclusive deal just for you",
"Act fast! Limited-time offer",
"Click here to secure your reward",
"Win cash prizes instantly by signing up",
"Limited-time discount on luxury watches",
"Get rich quick with this secret method",
"Hello, how are you today",
"Please find the attached report",
"Thank you for your support",
"The project deadline is next week",
"Can we reschedule the meeting to tomorrow",
"Your invoice for last month is attached",
"Looking forward to our call later today",
"Don’t forget the team lunch tomorrow",
"Meeting agenda has been updated",
"Here are the notes from yesterday’s discussion",
"Please confirm your attendance for the workshop",
"Let’s finalize the budget proposal by Friday"
]

labels = [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1,2),
    max_df=0.9,
    min_df=1
)

X = vectorizer.fit_transform(emails)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.25, random_state=42, stratify=labels
)

svm_model = LinearSVC(C=1.0)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader(f"Model Accuracy: {round(acc*100,2)}%")
st.subheader("Check Your Email")

user_input = st.text_area("Paste email content here")

if st.button("Detect Spam"):
    if user_input.strip() == "":
        st.warning("Please enter an email message")
    else:
        vect = vectorizer.transform([user_input])
        pred = svm_model.predict(vect)

        if pred[0] == 1:
            st.markdown('<div class="result-box spam">🚨 This Email is SPAM</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box notspam">✅ This Email is NOT SPAM</div>', unsafe_allow_html=True)
st.markdown("---")
st.caption("Made with ❤️ using Streamlit & Machine Learning")

