import warnings
import tempfile
import os

# Suppress InconsistentHashingWarning (use with caution)
warnings.filterwarnings("ignore", category=UserWarning, message="InconsistentHashingWarning")

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import cv2
from PIL import Image

# Set page configuration
st.set_page_config(page_title='Survey Results')

# Anomaly detection of illegal drugs and substance system
st.balloons()
st.sidebar.subheader("Large Large Model for illegal drugs")
# Select color picker
color = st.sidebar.color_picker("Pick a color", "#ff6347")
st.sidebar.markdown(
    """
    <h1 style='text-align: center; color: #0080FF;'>Anomaly detection of illegal drugs and Substance System</h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .css-1l02zno {
        position: absolute;
        top: 10px;
        right: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    st.title("Anomaly detection of illegal drugs and Substance System")
    st.session_state.logged_in = True

    # Step 1: Load the dataset
    df = pd.read_csv('data_6.csv')

    # Step 2: Prepare the data
    X = df['Image']  # Image paths
    y = df['Label']  # Labels

    # Step 3: Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Feature Extraction
    # Define a function to extract features from image paths
    def extract_features(image_paths):
        features = [image.split('\\')[-1].split('.')[0] for image in image_paths]  # Extract file names without extensions
        return features

    # Extract features for training and testing data
    X_train_features = extract_features(X_train)
    X_test_features = extract_features(X_test)

    # Step 5: Model Training
    # For illustration purposes, let's use a simple SVM classifier with CountVectorizer for feature extraction
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train_features)
    X_test_vectorized = vectorizer.transform(X_test_features)

    clf = SVC(kernel='linear')
    clf.fit(X_train_vectorized, y_train)

    # Step 6: Model Evaluation
    y_pred = clf.predict(X_test_vectorized)
    stats_x_y = classification_report(y_test, y_pred)

    # Step 7: Apply Data Augmentation
    def apply_augmentation(image):
        # Apply transformations
        augmented_images = []
        augmented_images.append(cv2.flip(image, 1))  # Horizontal flip
        augmented_images.append(cv2.flip(image, 0))  # Vertical flip
        augmented_images.append(np.rot90(image))  # Rotate 90 degrees clockwise
        augmented_images.append(np.rot90(image, k=3))  # Rotate 90 degrees counterclockwise
        return augmented_images

    if st.session_state.get("logged_in", False):
        # Load model and tokenizer
        
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        # st.success(uploaded_file)
        # k=str(uploaded_file)
        # st.success(k[10:])
        image=""
        if uploaded_file is not None:
            # Save the uploaded image to a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            # Read the uploaded image using cv2
            user_input_image = cv2.imread(temp_file_path)

            # Display the uploaded image
            st.image(user_input_image, caption='Uploaded Image', use_column_width=True)

            # Remove the temporary file
            os.unlink(temp_file_path)

        if st.button("Detect"):
            with st.spinner("Generating response..."):
                # predicted_disease=uploaded_file
                augmented_images = apply_augmentation(user_input_image)
                augmented_features = extract_features([temp_file_path] * len(augmented_images))

                # augmented_features = extract_features([user_input_image_path] * len(augmented_images))

                # Extract features from augmented images
                augmented_vectorized = vectorizer.transform(augmented_features)

                # Make predictions on augmented images
                augmented_predictions = clf.predict(augmented_vectorized)
                
            st.info("Cross-validation Report")
            rows = stats_x_y.split('\n')
            rows = [row.split() for row in rows[2:-5]]
            df_stats_x_y = pd.DataFrame(rows, columns=['class', 'precision', 'recall', 'f1-score', 'support'])
            st.table(df_stats_x_y)
            st.info("Model Response:")
            ans ="Object / Substance From Picture Is: "+(str(augmented_predictions[0])).upper()
            st.success(ans)

        st.markdown("---")

        st.sidebar.header("About")
        st.sidebar.write("This Model Detects illegal and none illegal drugs from scanning pictures")
    else:
        st.error("UnicodeWarning")  # Display a message or redirect to the login page

if __name__ == "__main__":
    main()
