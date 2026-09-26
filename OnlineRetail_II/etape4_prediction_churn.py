"""
Projet : Segmentation client & prédiction de churn - Online Retail II
Étape 4 : Prédiction du risque de churn

L'objectif : à partir du comportement passé d'un client, prédire s'il va
arrêter d'acheter dans le futur proche, pour pouvoir agir avant qu'il ne
soit trop tard.

Point important : on ne peut pas simplement
dire "un client qui n'a pas acheté depuis longtemps est un client churné"
et s'arrêter là, ça ne prédit rien, ça ne fait que constater un fait déjà
passé. Pour faire une vraie prédiction, on se place à un instant donné dans
le passé (une "date de coupure"), on regarde ce qu'on savait du client À CE
MOMENT-LÀ (ses achats jusque-là), et on vérifie s'il a effectivement acheté
ou non dans les 90 jours qui ont suivi. C'est cette relation passé -> futur
que le modèle apprend, exactement comme il devra le faire en conditions
réelles pour un client actuel.

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1. Chargement des données

df = pd.read_csv("online_retail_clean.csv", parse_dates=["InvoiceDate"])

# 2. La date de coupure
# On se place 90 jours avant la fin des données disponibles. Tout ce qui se
# passe AVANT sert à décrire le client (les "features"). Tout ce qui se
# passe APRÈS sert à vérifier s'il a churné ou non.

date_max = df["InvoiceDate"].max()
date_coupure = date_max - pd.Timedelta(days=90)
print(f"Date de coupure : {date_coupure.date()} (données jusqu'à {date_max.date()})")

df_avant = df[df["InvoiceDate"] <= date_coupure]
df_apres = df[df["InvoiceDate"] > date_coupure]

# 3. Consutrctuion des features (sur la période avant)
# On ne recalcule pas exactement le RFM de l'étape 2 (qui utilise toute la
# période) : ici on le recalcule volontairement seulement jusqu'à la date de
# coupure, pour ne pas tricher en donnant au modèle des informations qu'il
# n'aurait pas dans la vraie vie.

features = df_avant.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (date_coupure - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("LineTotal", "sum"),
).reset_index()

print(f"Clients avec historique avant la coupure : {features.shape[0]}")

# 4. Consutrction de la cible (churn = pas d'achat après)

clients_actifs_apres = set(df_apres["CustomerID"].unique())
features["Churn"] = (~features["CustomerID"].isin(clients_actifs_apres)).astype(int)

taux_churn = features["Churn"].mean()
print(f"Taux de churn observé sur la période de test : {taux_churn:.1%}")

# 5. Séparation entrainement / test
# On garde 20% des clients de côté pour évaluer le modèle sur des clients
# qu'il n'a jamais "vus" pendant l'entraînement -> ça simule des clients
# futurs inconnus, et ça évite de se mentir sur la vraie performance.

X = features[["Recency", "Frequency", "Monetary"]]
y = features["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
#Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Modèle 1 : Régression logistique
# Avantage principal à mentionner en entretien : chaque variable a un poids
# interprétable ("plus la Récence augmente, plus le risque augmente"), ce
# qui est précieux pour convaincre un client non-technique.

logit = LogisticRegression(random_state=42)
logit.fit(X_train_scaled, y_train)
y_pred_logit = logit.predict(X_test_scaled)
y_proba_logit = logit.predict_proba(X_test_scaled)[:, 1]

print("\n--- Régression logistique ---")
print(classification_report(y_test, y_pred_logit, target_names=["Actif", "Churn"]))
print(f"AUC : {roc_auc_score(y_test, y_proba_logit):.3f}")

# 7. Modèle 2 : RANDOM FOREST (plus flexible, capture des relations non
#    linéaires, souvent plus performant mais moins "explicable" variable
#    par variable)

rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)  # Random Forest n'a pas besoin de standardisation
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

print("\n--- Random Forest ---")
print(classification_report(y_test, y_pred_rf, target_names=["Actif", "Churn"]))
print(f"AUC : {roc_auc_score(y_test, y_proba_rf):.3f}")

# 8. Pourquoi le recall compte plus que la précision ici ?
# Le "recall" (rappel) sur la classe Churn répond à : "parmi les clients qui
# ont vraiment churné, combien en a-t-on détecté ?"
# La "precision" répond à : "parmi ceux qu'on a signalés comme à risque,
# combien le sont vraiment ?"
# Ici, rater un client qui part (faux négatif) coûte plus cher à
# l'entreprise que de relancer par erreur un client qui n'allait pas partir
# (faux positif, ça coûte juste un email de relance). On privilégie donc un
# modèle avec un bon RECALL sur la classe Churn, quitte à accepter une
# précision un peu plus faible.

# 9. Prédiction sur tout les clients
# On choisit ici le modèle Random Forest (à ajuster selon lequel des deux
# donne le meilleur recall sur l'exécution) pour produire une probabilité
# de churn pour l'ensemble des clients.

features["Churn_proba"] = rf.predict_proba(X[["Recency", "Frequency", "Monetary"]])[:, 1]

export_churn = features[["CustomerID", "Churn_proba"]]
export_churn.to_csv("churn_predictions.csv", index=False)
print(f"\nFichier de prédictions sauvegardé : churn_predictions.csv ({len(export_churn)} clients)")
