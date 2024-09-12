import streamlit as st
import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model('path/to/your_model.h5')

def preprocess_image(image):
    # Redimensionner l'image à la taille d'entrée du modèle
    image = image.resize((224, 224))  # Ajustez la taille selon votre modèle
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normaliser l'image
    image_array = np.expand_dims(image_array, axis=0)  # Ajouter une dimension pour le lot
    return image_array

def predict_image(image_array):
    predictions = model.predict(image_array)
    # Remplacez ceci par la logique de traitement de vos prédictions
    return predictions

#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element


import streamlit as st

# Définition des fonctions pour chaque page
def home():
    st.title("Page d'accueil")
    st.write("Bienvenue sur la page d'accueil de l'application.")

def disease_prediction():
    st.title("Prédiction des maladies")
    st.write("Cette section est dédiée à la prédiction des maladies du maïs.")
    # Ajoutez ici le code pour la prédiction des maladies

def treatment_advice():
    st.title("Conseils de traitement")
    st.write("Cette section fournit des conseils sur les traitements des maladies du maïs.")
    # Ajoutez ici le code pour afficher des conseils de traitement

def about():
    st.title("À propos de l'application")
    about_text = """
    # À Propos de l'Application

    Bienvenue dans notre application de **Prédiction des Maladies du Maïs**. Cette application utilise l'intelligence artificielle pour détecter et identifier diverses maladies du maïs à partir d'images de feuilles, et fournit des conseils sur les traitements appropriés.

    ## Fonctionnement de l'Application

    L'application est divisée en plusieurs sections :
    - **Prédiction des Maladies**
    - **Conseils de Traitement**
    - **À Propos**

    ## Explications du Modèle d'IA

    L'application utilise un modèle de **Deep Learning** basé sur une architecture de réseau de neurones convolutifs (CNN) pour la classification des images.
    """
    st.markdown(about_text)

def contact():
    st.title("Contactez-nous")
    st.write("Si vous avez des questions, suggestions ou commentaires, veuillez remplir le formulaire ci-dessous.")
    
    # Formulaire de contact
    with st.form(key='contact_form'):
        name = st.text_input("Nom")
        email = st.text_input("Email")
        message = st.text_area("Message")
        submit_button = st.form_submit_button(label='Envoyer')

        if submit_button:
            if name and email and message:
                st.success(f"Merci, {name}! Votre message a été envoyé avec succès.")
            else:
                st.error("Veuillez remplir tous les champs.")

def contributors():
    st.title("Contributeurs")
    st.write("Découvrez l'équipe de développement de cette application.")

    # Liste des contributeurs
    contributors_list = [
        {
            "name": "Jean Dupont",
            "matricule": "JD001",
            "photo": "https://example.com/photo1.jpg",  # Remplacez par l'URL de la photo
            "presentation": "Développeur backend passionné par l'IA et le machine learning."
        },
        {
            "name": "Marie Curie",
            "matricule": "MC002",
            "photo": "https://example.com/photo2.jpg",
            "presentation": "Scientifique des données avec une expérience en agronomie."
        },
        {
            "name": "Albert Einstein",
            "matricule": "AE003",
            "photo": "https://example.com/photo3.jpg",
            "presentation": "Expert en réseaux de neurones et en traitement d'images."
        }
    ]

    # Affichage des contributeurs
    for contributor in contributors_list:
        st.image(contributor["photo"], width=150)
        st.write(f"**Nom :** {contributor['name']}")
        st.write(f"**Matricule :** {contributor['matricule']}")
        st.write(f"**Présentation :** {contributor['presentation']}")
        st.write("---")

# Barre latérale de navigation
st.sidebar.title("Menu de Navigation")
menu = st.sidebar.radio(
    "Choisissez une page",
    ("Accueil", "Prédiction des Maladies", "Conseils de Traitement", "À Propos", "Contact", "Contributeurs")
)

# Affichage de la page en fonction de la sélection du menu
if menu == "Accueil":
    home()
elif menu == "Prédiction des Maladies":
    disease_prediction()
elif menu == "Conseils de Traitement":
    treatment_advice()
elif menu == "À Propos":
    about()
elif menu == "Contact":
    contact()
elif menu == "Contributeurs":
    contributors()
elif menu == 'Télécharger une Image':
    st.header('Téléchargez une Image')
    
    uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Afficher l'image téléchargée
        image = Image.open(uploaded_file)
        st.image(image, caption='Image téléchargée', use_column_width=True)
        
        # Prétraiter l'image et faire des prédictions
        image_array = preprocess_image(image)
        predictions = predict_image(image_array)
        
        st.write('Prédictions:', predictions)


#Sidebar
# st.sidebar.title("Dashboard")
# app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

# #Main Page
# if(app_mode=="Home"):
#     st.header("Plant Desease Detection")
#     # image_path = "home_page.jpeg"
#     # st.image(image_path,use_column_width=True)
#     st.markdown("""
#     Welcome to the Plant Disease Recognition System! 🌿🔍
    
#     Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

#     ### How It Works
#     1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
#     2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
#     3. **Results:** View the results and recommendations for further action.

#     ### Why Choose Us?
#     - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
#     - **User-Friendly:** Simple and intuitive interface for seamless user experience.
#     - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

#     ### Get Started
#     Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

#     ### About Us
#     Learn more about the project, our team, and our goals on the **About** page.
#     """)

# #About Project
# elif(app_mode=="About"):
#     st.header("About")
#     st.markdown("""
#                 #### About Dataset
#                 This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
#                 This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
#                 A new directory containing 33 test images is created later for prediction purpose.
#                 #### Content
#                 1. train (70295 images)
#                 2. test (33 images)
#                 3. validation (17572 images)

#                 """)

# #Prediction Page
# elif(app_mode=="Disease Recognition"):
#     st.header("Disease Recognition")
#     test_image = st.file_uploader("Choose an Image:")
#     if(st.button("Show Image")):
#         st.image(test_image,width=4,use_column_width=True)
#     #Predict button
#     if(st.button("Predict")):
#         st.snow()
#         st.write("Our Prediction")
#         result_index = model_prediction(test_image)
#         #Reading Labels
#         class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
#                     'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
#                     'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
#                     'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
#                     'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
#                     'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
#                     'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
#                     'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
#                     'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
#                     'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
#                     'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
#                     'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
#                     'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
#                       'Tomato___healthy']
#         st.success("Model is Predicting it's a {}".format(class_name[result_index]))
