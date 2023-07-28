import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import numpy as np
from matplotlib.lines import Line2D
from io import BytesIO


## Fonctions utilisées ##

# fonction pour obtenir via l'API la liste des id des nouveaux clients
def get_client_ids():
    response = requests.get(ids_url)
    return pd.read_json(response.json(), orient = 'split')

# fonction pour obtenir via l'API la liste des features par type
def get_features():
    response = requests.get(feat_url)
    return response.json()

# fonction pour obtenir via l'API l'image de l'explicabilité globale
def get_image():
    response = requests.get(exp_url)
    return BytesIO(response.content)

# fonction pour obtenir via l'API les infos du client sélectionné
def get_client_info(row_number):
    data = {'data': row_number}
    response = requests.post(client_url, json=data)
    return pd.read_json(response.json(), orient = 'split')

# fonction d'envoi du numéro client à l'API et de récupération de la prédiction initiale
def get_prediction(row_number):
    data = {'data': row_number}
    response = requests.post(predict_url, json=data)
    return response.json(),response.status_code

# fonction de récupération via l'API des infos du train (référence) limitées aux features voulues
def get_feat_info(feat):
    data = {'data': feat}
    response = requests.post(featinfo_url, json=data)
    return pd.read_json(response.json(), orient = 'split')

# fonction d'envoi à l'API des mises à jour des infos du client et de récupération des nouvelles prédictions
def submit_data(dico):
    data = {'data': dico}
    response = requests.post(update_url, json=data)
    return response.json(),response.status_code

# fonction de callback de la sélection du client (entrée pour vérification de la validité)
def callback1():
    st.session_state['client'] = True
    st.session_state['btn'] = False

# fonction de callback du click button de validation (affichage des informations du client)
def callback2():
    st.session_state['client'] = st.session_state['client']
    st.session_state['btn'] = True

# fonction de callback du click button de modifications (affichage des nouvelles prédictions et positions)
def callback3():
    st.session_state['client'] = st.session_state['client']
    st.session_state['btn'] = st.session_state['btn']
    st.session_state['update'] = True

# fonction de callback du multiselect des features pour changer les features par défaut du client sélectionné
def update_select():
    st.session_state['select_def'] = st.session_state['select_feat']
    st.session_state['update'] = st.session_state['update']

## URL d'API et récupéraiton des données de base ###

# URL de l'API Flask
ids_url = 'https://mbcreditmodelapi.azurewebsites.net/reflist'
feat_url = 'https://mbcreditmodelapi.azurewebsites.net/features'
exp_url = 'https://mbcreditmodelapi.azurewebsites.net/shap'
client_url = 'https://mbcreditmodelapi.azurewebsites.net/clientinfo'
predict_url = 'https://mbcreditmodelapi.azurewebsites.net/predict'
update_url = 'https://mbcreditmodelapi.azurewebsites.net/update'
featinfo_url = 'https://mbcreditmodelapi.azurewebsites.net/featureinfo'

# Récupération de la liste des nouveaux clients
test_ids = get_client_ids()
id_list = sorted(list(test_ids['SK_ID_CURR']))

# Récupération des listes de features
features = get_features()
num_col = list(features['num'])
cat_col = list(features['cat'])
features = sorted(num_col + cat_col)

# Récupération de l'image de l'explicabilité Shap
shap_exp = get_image()


## Construction de l'application ##

# Logo de présentation et état à la première connexion
st.image("Logo.png")

# initialisation de la session client
if 'client' not in st.session_state:
    st.session_state['client'] = False

# champ d'entrée du numéro de client
client_id = st.number_input('''Sélectionner ou entrer l\'ID du client et vérifier la sélection en appuyant sur "Entrer"''', 
min_value=100000, max_value=max(id_list), on_change = callback1)

# initialisation de la session exploration des informations
if 'btn' not in st.session_state:
    st.session_state['btn'] = False

if st.session_state['client']:
    
    if client_id in id_list:
        
        but = st.button("Afficher les informations du client", on_click = callback2)

        if st.session_state['btn']:

            # initialisation de la session sélection des features pour un client donné
            if 'select' not in st.session_state:
                st.session_state['select'] = False
            
            # récupération des infos client et des prédictions au click button
            client_data = get_client_info(client_id)
            prediction,status = get_prediction(client_id)

            if status == 200:
                st.session_state['select'] = True

                # définition des variables de données récupérées
                feature_importance = prediction['feature_importance']
                prediction_probabilities = prediction['prob']
                pred = prediction['prediction']
                gauge = prediction['gauge']

                # défintion des features importance pour l'explicabilité locale et des couleurs correspondantes
                feature_importance = dict(sorted(feature_importance.items(), key=lambda item: np.abs(item[1]),reverse=True))
                top_importances = [-i for i in feature_importance.values()]
                colors = ['red' if importance < 0 else 'blue' for importance in top_importances]
                threshold = 0.5157
                proba = round(prediction_probabilities[0]*100,2)
                thresh = round(threshold*100,2)
                gauge_color = 'green' if prediction_probabilities[0] <= threshold else 'red'

                # définition des 2 features créant le plus de risque d'insolvabilité pour le client sélectionné (affichage par défaut dans le multiselect)
                most_imp = []
                neg_imp = []
                for indic in feature_importance.keys():
                    if (feature_importance[indic] > 0) and (indic[0] not in ['0','1','2','3','4','5','6','7','8','9']):
                        neg_imp.append(indic)
                for feat_imp in neg_imp[0:2]:
                    i = min([feat_imp.index(elem) for elem in set(feat_imp).intersection(['<', '>', '='])])
                    most_imp.append(feat_imp[0:i-1])

                # distinction entre features renseignées ou non
                unknown = []
                known = []
                for feat in features:
                    if list(client_data.loc[client_data['SK_ID_CURR'] == client_id, feat].isna())[0] == True:
                        unknown.append(feat)
                    else:
                        known.append(feat)
                
                if st.session_state['select']:

                    # initialisation de la session du multiselect pour la mise à jour des features
                    if 'select_feat' not in st.session_state:
                        st.session_state['select_feat'] = most_imp 
                    if 'select_def' not in st.session_state:
                        st.session_state['select_def'] = most_imp           

                    # affichage des indicateurs du client
                    st.write('_'*100)
                    st.markdown("<h5 style='text-align: center; '>Indicateurs du client</h5>", unsafe_allow_html=True)
                    st.write(client_data)
                    
                    # création des colonnes de modification ou complétion des données 
                    # avec distinction entre données numériques (st.number_input avec valeurs min et max) ou catégorielles (liste à choix unique sous forme de st.radio)
                    cola, colb = st.columns([3,3])
                    dico_update = {}
                    dico_update['client_id'] = client_id

                    # pour les indicateurs connus
                    with cola:
                        st.markdown(f'<p style="font-size: 18px; text-align: center"> Indicateurs connus</p>', unsafe_allow_html=True)
                        known_indic = st.multiselect('Modifier une valeur', known)
                        if known_indic:
                            for feat in known_indic:
                                train = get_feat_info([feat])
                                if feat in num_col:
                                    min_val = float(min(train[feat]))
                                    max_val = float(max(train[feat]))
                                try:
                                    client_val = float(client_data.loc[client_data['SK_ID_CURR']==client_id, feat])
                                except:
                                    client_val = str(client_data.loc[client_data['SK_ID_CURR']==client_id, feat].values[0])
                                if type(client_val) == float:
                                    st.write("Valeur actuelle pour {ind} : {val}".format(ind = feat, 
                                    val = round(client_val,2)))
                                    new_val = st.number_input('Modifier', value=client_val,
                                    min_value=min_val, max_value=max_val)
                                    dico_update[feat] = new_val
                                else:
                                    st.write("Valeur actuelle pour {ind} : {val}".format(ind = feat,
                                    val = client_val))
                                    opt = list(set(train[feat]))
                                    idx = opt.index(list(client_val)[0])
                                    new_val = st.radio('Modifier', options = opt, index = idx)
                                    dico_update[feat] = new_val

                    # pour les indicateurs inconnus
                    with colb:
                        st.markdown(f'<p style="font-size: 18px; text-align: center"> Indicateurs inconnus</p>', unsafe_allow_html=True)
                        unknown_indic = st.multiselect('Entrer une valeur', unknown)
                        if unknown_indic:
                            for feat in unknown_indic:
                                train = get_feat_info([feat])
                                if feat in num_col:
                                    st.write(f'Pas de valeur renseignée')
                                    new_val = st.number_input('Entrer',
                                    min_value=float(min(train[feat])), max_value=float(max(train[feat])))
                                    dico_update[feat] = new_val
                                else:
                                    st.write(f'Pas de valeur renseignée')
                                    opt = list(set(train[feat]))
                                    new_val = st.radio('Modifier', options = opt)
                                    dico_update[feat] = new_val

                    if 'update' not in st.session_state:
                        st.session_state['update'] = False

                    if st.button('Valider les modifications', on_click = callback3):
                        # appel de la fonction submit_data() pour envoyer les données modifiées à l'API Flask seulement en cas de click button
                        update, stat = submit_data(dico_update)
                        if stat == 200:
                            # comme pour la prédiction initiale, définition des nouvelles variables et des couleurs puis affichage des nouveaux résultats
                            update_importance = update['feature_importance']
                            update_probabilities = update['prob']
                            update_pred = update['prediction']
                            update_gauge = update['gauge']

                            update_importance = dict(sorted(update_importance.items(), key=lambda item: np.abs(item[1]),reverse=True))
                            update_top_importances = [-i for i in update_importance.values()]
                            colors = ['red' if importance < 0 else 'blue' for importance in update_top_importances]
                            update_proba = round(update_probabilities[0]*100,2)
                            update_gauge_color = 'green' if update_probabilities[0] <= threshold else 'red'

                            st.write('_'*100)
                            st.markdown(f'<h5 style="text-align: center;"> Probabilité après modifications: {update_proba}%</h5>', 
                            unsafe_allow_html=True)
                            st.markdown(f"<h3 style='text-align: center; color: {update_gauge_color}; font-size: 24px;'> {update_gauge}</h3>", 
                            unsafe_allow_html=True)

                            # affichage compratif (colonnes) des positions (scatterplot des probabilité selon l'indicateur et point du client)
                            # avant et après modifications / complétions pour les indicateurs modifiés / complétés
                            if list(dico_update.keys())[1::] != []: 
                                for feat in list(dico_update.keys())[1::]:
                                    train = get_feat_info([feat])
                                    st.write('_'*100)
                                    st.markdown(f"<h5 style='text-align: center; '>{feat}</h1>", unsafe_allow_html=True)
                                    colc, cold = st.columns([3,3])
                                    with colc:
                                        st.markdown("<h6 style='text-align: center; '>Position actuelle</h6>", 
                                        unsafe_allow_html=True)
                                        if feat in known:     
                                            fig, ax = plt.subplots()
                                            sns.scatterplot(x=feat, y ='proba', hue = 'TARGET', data = train, ax = ax)
                                            x_min, x_max, y_min, y_max = plt.axis()
                                            plt.hlines(y = 0.5157, xmin = x_min, xmax = x_max, color = 'red')
                                            xp = client_data.loc[client_data['SK_ID_CURR']==client_id, feat]
                                            yp = prediction_probabilities[0]
                                            sns.scatterplot(x=xp, y=[yp], color=gauge_color, marker='o', s=100)
                                            plt.xticks(rotation = 45, ha = 'right')
                                            plt.xlabel('')
                                            plt.ylabel('Probabilité')
                                            legend_elements = [Line2D([0], [0], color='w',  markerfacecolor='b', marker='o', label='Accepté'),
                                            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', label='Refusé'),
                                            Line2D([0], [0], marker='o', color='w', markerfacecolor=gauge_color, label='Client sélectionné')]
                                            plt.legend(handles = legend_elements)
                                            st.pyplot(fig)
                                        else:
                                            st.markdown("<h6 style='text-align: center; '> Inconnue</h6>", unsafe_allow_html=True)

                                    with cold:
                                        st.markdown("<h6 style='text-align: center; '>Position avec modifications</h6>", 
                                        unsafe_allow_html=True)   
                                        fig, ax = plt.subplots()
                                        sns.scatterplot(x=feat, y ='proba', hue = 'TARGET', data = train, ax = ax)
                                        x_min, x_max, y_min, y_max = plt.axis()
                                        plt.hlines(y = 0.5157, xmin = x_min, xmax = x_max, color = 'red')
                                        xp = dico_update[feat]
                                        yp = update_probabilities[0]
                                        sns.scatterplot(x=[xp], y=[yp], color=update_gauge_color, marker='o', s=100)
                                        plt.xticks(rotation = 45, ha = 'right')
                                        plt.xlabel('')
                                        plt.ylabel('Probabilité')
                                        legend_elements = [Line2D([0], [0], color='w',  markerfacecolor='b', marker='o', label='Accepté'),
                                        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', label='Refusé'),
                                        Line2D([0], [0], marker='o', color='w', markerfacecolor=update_gauge_color, label='Client sélectionné')]
                                        plt.legend(handles = legend_elements)
                                        st.pyplot(fig)        

                            else:
                                st.write('_'*100)
                                st.markdown("<h5 style='text-align: center; '> Aucun changement effectué</h5>", unsafe_allow_html=True)       

                        # rappel des résultats initiaux
                        st.write('_'*100)
                        st.markdown("<h5 style='text-align: center; '> Position actuelle du client</h5>", unsafe_allow_html=True)

                    # affichage systématique des résultats initiaux du client
                    col1bis, col2bis = st.columns([3,3])

                    with col1bis:
                        st.markdown(f'<h5 style="text-align: center;"> Probabilité de non solvabilité : {proba}%</h5>', 
                        unsafe_allow_html=True)

                    with col2bis:
                        st.markdown(f'<p style="font-size: 18px; text-align: center"> Seuil de risque : {thresh}%</p>', unsafe_allow_html=True)

                    st.markdown(f"<h3 style='text-align: center; color: {gauge_color}; font-size: 24px;'> {gauge}</h3>", unsafe_allow_html=True)
                    st.write('_'*100)

                    # affichage des features importances globale et locale
                    col4, col5 = st.columns([3,3])

                    # feature importance globale avec Shap
                    with col4:
                        st.markdown("<h5 style='text-align: center; '>Indicateurs les plus importants (Shap)</h5>", 
                            unsafe_allow_html=True)
                        st.image(shap_exp)

                    # feature importance locale avec lime
                    with col5:
                        st.markdown("<h5 style='text-align: center; '>Position relative du client (Lime)</h5>", 
                        unsafe_allow_html=True)
                        fig, ax = plt.subplots(figsize = (8, 15))
                        ax.barh(range(len(top_importances)), top_importances, color=colors)
                        ax.set_yticks(range(len(top_importances)))
                        ax.set_yticklabels([i for i in feature_importance.keys()])
                        ax.invert_yaxis() 
                        ax.spines[['right', 'top']].set_visible(False)
                        plt.yticks (fontsize=14)

                        plt.xlabel('')
                        plt.ylabel('')
                        st.pyplot(fig)

                    st.write('_'*100)

                    # affichage du multiselect de sélection des features à observer
                    selected_features = st.multiselect('Sélectionner les indicateurs souhaités (par défaut les deux les plus critiques pour le client sont affichés)', 
                    options = features, key = 'select_feat', default = st.session_state['select_def'], on_change = update_select)
                    if selected_features:
                        # distinction entre données numériques et catégorielles
                        list_feat = list(selected_features)
                        train = get_feat_info(list_feat)
                        for feature in selected_features:
                            if list(client_data.loc[client_data['SK_ID_CURR']==client_id, feature].isna())[0] == True:
                                client_value = ['non renseignée']
                            else:
                                try:
                                    client_value = float(client_data.loc[client_data['SK_ID_CURR']==client_id, feature])
                                except:
                                    client_value = str(client_data.loc[client_data['SK_ID_CURR']==client_id, feature].values[0])
                            st.write('_'*100)
                            st.markdown(f"<h5 style='text-align: center; '>{feature}</h1>", unsafe_allow_html=True)
                            st.markdown(f'<p style="font-size: 18px; text-align: center; "> Valeur du client : {client_value}</p>',
                            unsafe_allow_html=True)

                            col7, col6 = st.columns([3,3])

                            # scatterplots des probabilités en fonction des indicateurs
                            with col6:
                                st.markdown("<h6 style='text-align: center; '>Position relative du client</h6>", 
                                unsafe_allow_html=True)
                                if list(client_data.loc[client_data['SK_ID_CURR']==client_id, feature].isna())[0] != True:
                                    fig, ax = plt.subplots()
                                    sns.scatterplot(x=feature, y ='proba', hue = 'TARGET', data = train, ax = ax)
                                    x_min, x_max, y_min, y_max = plt.axis()
                                    plt.hlines(y = 0.5157, xmin = x_min, xmax = x_max, color = 'red')
                                    xp = client_data.loc[client_data['SK_ID_CURR']==client_id, feature]
                                    yp = prediction_probabilities[0]
                                    sns.scatterplot(x=xp, y=[yp], color=gauge_color, marker='o', s=100)
                                    plt.xticks(rotation = 45, ha = 'right')
                                    plt.xlabel('')
                                    plt.ylabel('Probabilité')
                                    legend_elements = [Line2D([0], [0], color='w',  markerfacecolor='b', marker='o', label='Accepté'),
                                    Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', label='Refusé'),
                                    Line2D([0], [0], marker='o', color='w', markerfacecolor=gauge_color, label='Client sélectionné')]
                                    plt.legend(handles = legend_elements)
                                    st.pyplot(fig)
                                else:
                                    st.markdown(f'<p style="font-size: 18px; text-align: center; "> Le client devrait renseigner cet indicateur. </p>', 
                                    unsafe_allow_html=True)
                                    st.markdown(f'<p style="font-size: 18px; text-align: center; "> Sa valeur pourrait modifier la décision. </p>', 
                                    unsafe_allow_html=True)

                            # boxplots pour les donées numériques et histogrammes pour les données catégorielles avec distinction des targets (0 ou 1)
                            with col7:
                                st.markdown("<h6 style='text-align: center; '>Distribution globale des clients</h6>", 
                                unsafe_allow_html=True)
                                fig, ax = plt.subplots()
                                if feature in num_col:
                                    prob = 0 if prediction_probabilities[0] <= threshold else 1
                                    sns.boxplot(x= 'TARGET', y = feature, data = train)
                                    xp = client_value
                                    plt.plot(prob, xp, marker = 'o', color = gauge_color)
                                    plt.xlabel('')
                                    plt.ylabel('')
                                    plt.xticks([0,1],['Accepté', 'Refusé'])
                                    st.pyplot(fig)
                                else:
                                    sns.histplot(train[train['TARGET']==0][feature], label = 'Accepté')
                                    sns.histplot(train[train['TARGET']==1][feature], label = 'Refusé')
                                    plt.xlabel('')
                                    plt.ylabel('')
                                    plt.xticks(rotation = 45, ha = 'right')
                                    plt.legend()
                                    st.pyplot(fig)

                        # graphique bi-varié en cas de sélection de 2 indicateurs (prise en compte des deux indicateurs par défaut du client au démarage)
                        if len(selected_features) == 2:
                            st.write('_'*100)
                            st.markdown("<h5 style='text-align: center; '>Positionnement relatif pour les 2 indicateurs sélectionnés</h5>", 
                            unsafe_allow_html=True)                      
                            if (list(client_data.loc[client_data['SK_ID_CURR']==client_id, 
                            selected_features[0]].isna())[0] != True) and (list(client_data.loc[client_data['SK_ID_CURR']==client_id, 
                            selected_features[1]].isna())[0] != True):
                                try:
                                    client_value1 = float(client_data.loc[client_data['SK_ID_CURR']==client_id, selected_features[0]])
                                except:
                                    client_value1 = str(client_data.loc[client_data['SK_ID_CURR']==client_id, selected_features[0]].values[0])
                                try:
                                    client_value2 = float(client_data.loc[client_data['SK_ID_CURR']==client_id, selected_features[1]])
                                except:
                                    client_value2 = str(client_data.loc[client_data['SK_ID_CURR']==client_id, selected_features[1]].values[0])
                        
                                fig, ax = plt.subplots()
                                sns.scatterplot(data=train, x=selected_features[0], y=selected_features[1], hue='TARGET',s=10)
                                sns.scatterplot(x=[client_value1], y=[client_value2], color=gauge_color, marker='o', s=100)
                                legend_elements = [Line2D([0], [0], color='w', markerfacecolor='b', marker='o', label='Accepté'),
                                Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', label='Refusé'),
                                Line2D([0], [0], marker='o', color='w', markerfacecolor=gauge_color, label='Client sélectionné')]
                                plt.legend(handles = legend_elements)
                                plt.xticks(rotation = 45, ha = 'right')
                                st.pyplot(fig) 
                            else:
                                st.markdown(f'''<p style="font-size: 18px; text-align: center; "> Au moins un des indicateurs n'est pas renseigné</p>''',
                                unsafe_allow_html=True)                                       

            else:
                st.error('Erreur lors de la récupération des prédictions')

    else:
        st.write('Entrer un ID client valide')
