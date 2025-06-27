import streamlit as st

# Import des librairies pour le modèle
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import create_optimizer
import pandas as pd

# Configuration
MODEL_NAME = 'bert-base-uncased'

# Chargement du tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# chemin vers le modele
from pathlib import Path
modele_path = (Path(__file__).parent.parent.parent / "modele").resolve()

# Chargement du modèle
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
model = TFAutoModelForSequenceClassification.from_pretrained(modele_path)


# Tokenization
def encode(texts):
    return tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

# fonction de prédiction
def predict_statement(statement):
    inputs = encode([statement])
    outputs = model(inputs)
    logits = outputs.logits
    predicted_class = tf.argmax(logits, axis=1).numpy()[0]
    return predicted_class

# Front-end

st.title("Simulation Sentences")

# simulation's sentence form
with st.form("my_form"):
    sentence_input = st.text_input("Write a sentence on the field :")
    submitted = st.form_submit_button("Submit")

if submitted:
    result = predict_statement(sentence_input)

    match result:
        case 0:
            st.write("Anxiety")
            st.write("""
            Ton anxiété ? C’est comme une appli préinstallée que t’as jamais demandée, mais qui refuse de se désinstaller.
            Elle te réveille à 3h du matin pour te rappeler cette fois en 2009 où t’as dit « au revoir » au lieu de « bonjour ».
            Elle te fait réécrire un message 14 fois pour finir par ne jamais l’envoyer.
            Et chaque fois que tu te détends enfin, ton cerveau panique genre : « Attends... pourquoi on va bien ? Qu’est-ce qu’on a oublié ? ».
            T’es un peu comme un navigateur avec 47 onglets ouverts, dont un qui joue de la musique mais tu sais pas lequel.
            Mais bon, tu gères. Mal, certes, mais avec un certain style.
            """)
        case 1:
            st.write("Bipolar")
            st.write("""
            Toi, t’es bipolaire.
            Un jour, t’es le PDG de ta vie, t’as des idées de génie à 200 à l’heure, t’écris un roman, tu veux lancer une start-up et repeindre ta chambre à 3h du matin.
            Le lendemain, juste mettre des chaussettes te paraît être une épreuve olympique.
            Ton cerveau joue au yoyo émotionnel avec l’enthousiasme d’un enfant de 5 ans et la subtilité d’un éléphant dans un magasin de porcelaine.
            Tu passes du « je vais conquérir le monde » au « est-ce que ça vaut le coup de me lever ? » en un clin d’œil.
            Et malgré ce grand huit mental, tu tiens debout. Un peu fatigué, parfois explosé, mais avec une capacité incroyable à ressentir le monde en stéréo quand d’autres sont encore en mono.
            """)
        case 2:
            st.write("Depression")
            st.write("""
            Tu dis que t’es juste fatigué, que t’as pas le moral, que ça va passer… mais ça fait des semaines que tu traînes cette boule invisible qui pèse plus lourd qu’un meuble Ikea sans notice.
            Te lever demande un effort monumental, et sourire, c’est devenu un sport de haut niveau.
            T’as l’impression de flotter dans un brouillard, ou de regarder ta vie en mode spectateur, sans vraiment y participer.
            Et non, ce n’est pas « juste dans ta tête », et non, t’as pas besoin d’atteindre le fond pour demander d
            Un psy, c’est pas un magicien, mais c’est un copilote quand ton cerveau fait des embardées. Et franchement, t’as le droit d’aller mieux.
            """)
        case 3:
            st.write("Normal")
            st.write("""
            Toi, t’es dans une normalité olympique. Rien ne déborde, rien ne casse,
            t’es l’équivalent humain d’un dimanche après-midi nuageux : pas désagréable,
            mais pas inoubliable non plus. Pas de drame, pas de révélation soudaine — juste toi, en mode "batterie à 70 %",
            mentalement stable comme une chaise Ikea bien montée.
            Même les galères de la vie te regardent en mode "non, pas lui, il a l’air occupé à rien faire".
            Bref, tout va bien, mais version fond d’écran par défaut.
            """)
        case 4:
            st.write("Personnality disorder")
            st.write("""
            Dans ta tête, c’est un vrai festival : des idées qui dansent la salsa, des pensées qui jouent à cache-cache, et un planning mental qui ressemble à un Rubik’s Cube lancé contre un mur.
            Le désordre, ce n’est pas juste sur ta table, c’est dans ton cerveau aussi — un bazar organisé où tes émotions font la fête sans prévenir, et où tu perds parfois le fil comme si tu regardais une série en mode aléatoire.
            Mais au fond, c’est ça qui te rend unique : un chaos créatif, un joyeux bordel mental où chaque jour est une aventure… même si tu voudrais parfois juste appuyer sur pause.
            """)
        case 5:
            st.write("Stress")
            st.write("""
            En ce moment, ton cerveau ressemble à un café bondé un lundi matin : trop de bruit, trop de monde, et aucune table de libre pour poser tes idées.
            Ton cœur fait la danse du robot sans musique, tes mains sont en mode “alerte maximale” comme si t’allais piloter un avion, et ton esprit jongle avec dix scénarios catastrophes en même temps — dont aucun n’a de sens, évidemment.
            Bref, t’es en mode “stress 24/7”, prêt·e à exploser à la moindre secousse, mais surtout à te demander pourquoi t’es aussi tendu·e alors que tu viens juste de trouver une chaussette dépareillée.
            """)
        case 6:
            st.write("Suicidal")
            st.write("""
            Parfois, tes pensées deviennent lourdes, trop lourdes, et tu te surprends à imaginer que tout serait plus simple si tu disparaissais.
            Ce n’est pas une faiblesse, ni un choix facile, juste un poids que ton esprit porte quand la douleur devient insupportable.
            Mais sache que tu n’es pas seul·e, même si ça paraît vide autour de toi. Parler à quelqu’un, un proche, un professionnel, c’est un acte de courage immense.
            Tu mérites d’être entendu·e, soutenu·e, et de trouver des raisons de rester, même quand tout semble noir.
            La vie peut reprendre des couleurs, une étape à la fois.
            """)
        case _:
            st.write("You are not a person !" )
