import json
from pprint import pprint

import streamlit as st
import tensorflow as tf
import numpy as np


# Load Model
model = tf.keras.models.load_model('./model3.keras')

# Load database
with open("db.json", "r") as file:
    db = json.load(file)

diseases = db.get("diseases")
class_names = [key for key in diseases]


# Tensorflow Model Prediction
def model_prediction(input_image):
    image = tf.keras.preprocessing.image.load_img(input_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element


def send_message(mail_from: str, message: str):
    pass
############################################################################################################""

# Page functions 

def home():
    st.title("Prédiction des maladies de Maïs 🌿🔍")
    st.write("#### Benvenue dans notre système de Prédiction des maladies de Maïs")

    image = None

    upl_image = st.file_uploader("Sélectionnez une image de maïs", type=["png", "jpg","jpeg"], accept_multiple_files=False, help="Cliquez pour sélectionner une image de ma¨is sur votre appareil.")
    
    if upl_image is not None:
        image = upl_image
        st.image(image, caption='Image téléchargée', use_column_width=True)

    else:
        cam_image = st.camera_input("Ou utilisez votre caméra pour en capturer une.")
        if cam_image is not None:
            image = cam_image

    
    if image is not None:
        # st.snow()
        st.write("## Prédictions")
        with st.spinner("Analyse en cours !"):
            prediction_index = model_prediction(image)


            predicted_disease = diseases.get(class_names[prediction_index])

            st.write("### Maladie détectée")
            st.success(predicted_disease.get("french_name"))

            st.divider()
            st.write("### Sympt^omes")
            for symptom in predicted_disease.get("symptoms"):
                st.write(f"- {symptom}")

            st.divider()
            st.write("### Causes")
            for cause in predicted_disease.get("causes"):
                st.write(f"- {cause}")


            st.divider()
            st.write("### Conseils de traitement")
            for treatment in predicted_disease.get("treatments"):
                st.write(f"- {treatment}")


def about():
    st.title("À propos de l'application")
    about_text = """
    Bienvenue dans notre application de **Prédiction des Maladies du Maïs**. Cette application utilise l'intelligence artificielle pour détecter et identifier diverses maladies du maïs à partir d'images de leurs feuilles, et fournit des conseils sur les traitements appropriés.

    ## Comment ça marche ?

     1. **Télécharment de l'image:** Cliquez sur **Accueil** dans la barre de navigation latérale et téléchargez une image d'une feuille de mais ou utilisez votre caméra pour capturer une image (vous devez autoriser l'utilisation de la caméra pour cette fin).
     2. **Analyse:** Notre système va procéder au traitement de l'image en utilisant des algorithmes d'intelligence artificielle avancés pour identifier les maladies potentielles de la plante.
     3. **Resultats:** En quelques poussières de secondes, les résultats et des recommandations pour des actions ultérieures que vous pouvez effectuer pour protéger vos plantes.

    ## Pourquoi choisir notre système ?
     - **Précision:** notre système utilise des techniques d'apprentissage automatique de pointe pour une détection des maladies avec une présision de 98%.
     - **Convivial:** interface simple et intuitive pour une expérience utilisateur transparente.
     - **Rapide et efficace:** recevez des résultats en secondes, ce qui permet une prise de décision rapide.

    ## Explications du Modèle d'IA

    L'application utilise un modèle de **Deep Learning** basé sur une architecture de réseau de neurones convolutifs (CNN) pour la classification des images.


    ## Commencez maintenant !
    Rndez-vous sur la page d'accueil pour télécharger une image et expérimentez la puisssance de notre système de Détection des maladies de Mais !
 
    """
    st.markdown(about_text)

def contact():
    st.title("Contactez-nous")
    st.write("Si vous avez des questions, suggestions ou commentaires, veuillez remplir le formulaire ci-dessous.")
    
    # Contact form
    with st.form(key='contact_form'):
        name = st.text_input("Nom")
        email = st.text_input("Email")
        message = st.text_area("Message")
        submit = st.form_submit_button(label='Envoyer')

        if submit:
            if name and email and message:
                send_message(mail_from=email, message=message)
                st.success(f"Merci, {name}! Votre message a été envoyé avec succès.")
            else:
                st.error("Veuillez remplir tous les champs.")


def contributors():
    st.title("Contributeurs")
    st.write("Découvrez l'équipe de développement de cette application.")

    # Liste des contributeurs
    contributors_list = [
        {
            "name": "Mateso Emmanuel Prosper",
            "matricule": "22100",
            "photo": "https://gravatar.com/avatar/photo3.jpg",
            "presentation": "Expert en réseaux de neurones et en traitement d'images."
        },
        {
            "name": "Mugisho Bashige Olivier",
            "matricule": "22100313",
            "photo": "https://gravatar.com/avatar/d2499868c45cff812a99ac6c1946c372?s=200",
            "presentation": "Développeur backend passionné par l'IA et la sécurité informatique."
        },
        {
            "name": "Muhindo Muhaviri Archippe",
            "matricule": "22100",
            "photo": "https://gravatar.com/avatar/photo2.jpg",
            "presentation": "Scientifique des données avec une expérience en agronomie."
        },
        {
            "name": "Muhindo Rukeza Christian",
            "matricule": "22100",
            "photo": "https://gravatar.com/avatar/photo3.jpg",
            "presentation": "Expert en réseaux de neurones et en traitement d'images."
        },
        {
            "name": "Mwenyemali Jonathan Johnson",
            "matricule": "22100",
            "photo": "https://gravatar.com/avatar/photo3.jpg",
            "presentation": "Expert en réseaux de neurones et en traitement d'images."
        },
        {
            "name": "Saidi Abdul",
            "matricule": "22100",
            "photo": "https://gravatar.com/avatar/photo3.jpg",
            "presentation": "Expert en réseaux de neurones et en traitement d'images."
        },
        {
            "name": "Zedi Bulimwengu",
            "matricule": "22100",
            "photo": "https://gravatar.com/avatar/photo3.jpg",
            "presentation": "Expert en réseaux de neurones et en traitement d'images."
        },
    ]

    # Affichage des contributeurs
    for contributor in contributors_list:
        st.image(contributor["photo"], width=150)
        st.write(f"**Nom :** {contributor['name']}")
        st.write(f"**Matricule :** {contributor['matricule']}")
        st.write(f"**Présentation :** {contributor['presentation']}")
        st.write("---")


st.sidebar.title("Dashboard")
menu = st.sidebar.radio(
    "Choisissez une page",
    ("Accueil", "Prédiction des Maladies", "Conseils de Traitement", "À Propos", "Contact", "Contributeurs")
)

st.sidebar.divider()
st.sidebar.write("##### Comment trouvez-vous cette application ?")

feedback = st.sidebar.feedback("stars")
if feedback:
    if ("feedback" not in st.session_state) or (st.session_state["feedback"] != feedback):
        st.toast("Merci de votre avis. Cela nous permet d'améliorer cette application !")
        st.session_state["feedback"] = feedback


# Menu navigation

if menu == "Accueil":
    home()

elif menu == "À Propos":
    about()

elif menu == "Contact":
    contact()

elif menu == "Contributeurs":
    contributors()


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
