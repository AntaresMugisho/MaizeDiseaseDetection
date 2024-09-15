import json
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import streamlit as st
import tensorflow as tf
import numpy as np
from dotenv import load_dotenv

# Load dot env file
load_dotenv()


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


def send_mail(mail_from_name: str, mail_from_email: str, message: str):
    host = os.environ["MAIL_HOST"]
    sender_email = os.environ["MAIL_USERNAME"]
    password = os.environ["MAIL_PASSWORD"]
    receiver_email = os.environ["OWNER_EMAIL"]

    mail = MIMEMultipart()
    mail["From"] = f"{mail_from_name} <{mail_from_email}>"
    mail["To"] = receiver_email
    mail["ReplyTo"] = mail_from_name
    mail["Subject"] = "Maize Disease Detection"

    html_body = f"""
    <html>
    <body>
        <p>{message}</p>
    </body>
    </html>
    """
    mail.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(host, 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.send_message(mail)
    except Exception as e:
        print(e)
    
#####################################################################################

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
    Rendez-vous sur la page d'accueil pour télécharger une image et expérimentez la puisssance de notre système de Détection des maladies de Mais !
    """
    st.markdown(about_text)

def contact():
    st.title("Contactez-nous")
    st.write("Si vous avez des questions, suggestions ou commentaires, veuillez remplir le formulaire ci-dessous.")
    
    # Contact form
    with st.form(key='contact_form', clear_on_submit=True):
        name = st.text_input("Nom")
        email = st.text_input("Email")
        message = st.text_area("Message")
        submit = st.form_submit_button(label='Envoyer')

        if submit:
            if name and email and message:
                with st.spinner("Envoie en cours..."):
                    send_mail(mail_from_name=name, mail_from_email=email, message=message)
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
            "presentation": "Expert en Cybrsécurité."
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
            "presentation": "Data scientist avec une expérience en agronomie."
        },
        {
            "name": "Muhindo Rukeza Christian",
            "matricule": "22100",
            "photo": "https://gravatar.com/avatar/photo3.jpg",
            "presentation": "Expert en traitement d'images."
        },
        {
            "name": "Mwenyemali Jonathan Johnson",
            "matricule": "22100",
            "photo": "https://gravatar.com/avatar/photo3.jpg",
            "presentation": "Data scientist."
        },
        {
            "name": "Saidi Abdul",
            "matricule": "22100",
            "photo": "https://gravatar.com/avatar/photo3.jpg",
            "presentation": "Chercheur en IA."
        },
        {
            "name": "Zedi Bulimwengu",
            "matricule": "22100",
            "photo": "https://gravatar.com/avatar/photo3.jpg",
            "presentation": "Expert en réseaux."
        },
    ]

    # Affichage des contributeurs
    for contributor in contributors_list:
        st.markdown(f"""
            <div style="display:flex;flex-direction:column;align-items:center;gap:20px;">
                <img src="{contributor["photo"]}" width="150px" style="border-radius:50%;margin-left:auto;margin-right:auto;" />
                <p style="text-align:center;"> <span style="color:gray">{contributor["matricule"]}</span> </br> <strong style="font-size:24px">{contributor["name"]}</strong> </br>{contributor["presentation"]}</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.divider()



###############################

# App construction

st.sidebar.title("Dashboard")
menu = st.sidebar.radio(
    "Choisissez une page",
    ("Accueil", "À Propos", "Contact", "Contributeurs")
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
