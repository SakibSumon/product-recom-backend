# import numpy as np
# import pickle as pkl
# import tensorflow as tf
# from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.layers import GlobalMaxPool2D

# from sklearn.neighbors import NearestNeighbors
# import os
# from numpy.linalg import norm
# import streamlit as st 
# import glob


# def extract_features_from_images(image_path, model):
#     img = image.load_img(image_path, target_size=(224,224))
#     img_array = image.img_to_array(img)
#     img_expand_dim = np.expand_dims(img_array, axis=0)
#     img_preprocess = preprocess_input(img_expand_dim)
#     result = model.predict(img_preprocess).flatten()
#     norm_result = result/norm(result)
#     return norm_result


# st.header('Fashion Recommendation System')


# Image_features = pkl.load(open('Images_features.pkl','rb'))

# print("Feature stats - Mean:", np.mean(Image_features), "Std:", np.std(Image_features))
# # print("Total features shape:", Image_features.shape)
# # st.write('Number of Images:', Image_features.shape[0])

# filenames = pkl.load(open('filenames.pkl','rb'))
# filenames = [os.path.join('images', os.path.basename(path)) for path in filenames]
# print("Sample filenames:", filenames[:5])

# # Model setup
# model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
# model.trainable = False
# model = tf.keras.models.Sequential([model, GlobalMaxPool2D()])

# # Verify model
# sample_input = np.random.rand(1, 224, 224, 3)
# sample_output = model.predict(sample_input)
# print("Sample model output shape:", sample_output.shape)

# # Nearest Neighbors
# neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
# neighbors.fit(Image_features)
# print("Nearest neighbors model fitted")

# upload_file = st.file_uploader("Upload Image")
# if upload_file is not None:
#     # Save and verify upload
#     upload_path = os.path.join('upload', upload_file.name)
#     with open(upload_path, 'wb') as f:
#         f.write(upload_file.getbuffer())
#     print(f"Saved uploaded file to: {upload_path}")
    
#     if not os.path.exists(upload_path):
#         st.error("Failed to save uploaded file!")
#     else:
#         st.subheader('Uploaded Image')
#         st.image(upload_file)
        
#         # Feature extraction with verification
#         try:
#             input_img_features = extract_features_from_images(upload_path, model)
#             print("Input features shape:", input_img_features.shape)
#             print("Input features norm:", norm(input_img_features))
#             print("First 10 features:", input_img_features[:10])
            
#             # Nearest neighbor search
#             distances, indices = neighbors.kneighbors([input_img_features])
#             print("Neighbor indices:", indices)
#             print("Neighbor distances:", distances)
            
#             # Display results
#             st.subheader('Recommended Images')
#             cols = st.columns(5)
#             for i, col in enumerate(cols, start=1):
#                 with col:
#                     st.image(filenames[indices[0][i]])
#         except Exception as e:
#             st.error(f"Error processing image: {str(e)}")
#             print(f"Error: {str(e)}")




from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from sklearn.neighbors import NearestNeighbors
import os

app = FastAPI()

#CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],  # Nuxt dev server
#     allow_credentials=True,
#     allow_methods=["*"],  # GET, POST, PUT, DELETE, etc.
#     allow_headers=["*"],  # Allow all headers
# )

# Model setup
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model = tf.keras.models.Sequential([model, GlobalMaxPool2D()])
model.trainable = False

features = pkl.load(open('Images_features.pkl', 'rb'))
filenames = pkl.load(open('filenames.pkl', 'rb'))
neighbors = NearestNeighbors(n_neighbors=6, metric='euclidean')
neighbors.fit(features)

os.makedirs("temp_uploads", exist_ok=True)


def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array)
    result = model.predict(img_preprocessed).flatten()
    return result / np.linalg.norm(result)


@app.post("/recommend")
async def recommend(file: UploadFile = File(...)):
    upload_path = f"temp_uploads/{file.filename}"
    with open(upload_path, "wb") as buffer:
        buffer.write(await file.read())
    try:
        features_vec = extract_features(upload_path)
        distances, indices = neighbors.kneighbors([features_vec])
        # result_paths = [filenames[i] for i in indices[0][1:6]]
        result_paths = [filenames[i].replace("\\", "/") for i in indices[0][1:6]]

        return {"results": result_paths}
    except Exception as e:
        return {"error": str(e)}