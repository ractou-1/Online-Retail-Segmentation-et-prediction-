"""
Projet : Segmentation client & prédiction de churn - Online Retail II
Étape 2 : Analyse RFM (Récence, Fréquence, Montant)

L'analyse RFM est une méthode simple et très utilisée en marketing pour
comprendre la valeur de chaque client, sans modèle compliqué. Elle répond
à 3 questions pour chaque client :

  - Récence (R)   : "Depuis combien de temps n'a-t-il pas acheté ?"
                     Plus c'est récent, mieux c'est.
  - Fréquence (F) : "Combien de fois a-t-il commandé ?"
                     Un client qui revient souvent est plus fidèle.
  - Montant (M)   : "Combien a-t-il dépensé au total ?"
                     Indique la valeur économique du client.

En croisant ces 3 informations, on peut distinguer par exemple :
  - un client qui a beaucoup dépensé mais n'est pas revenu depuis longtemps
    (gros client en train de partir -> à recontacter en priorité)
  - un client qui achète souvent et récemment mais pour de petits montants
    (client fidèle mais avec un potentiel de vente additionnelle)

C'est la base qui permettra ensuite de segmenter les clients (étape 3)
et de prédire lesquels risquent de partir (étape 4).
===========================================================================
"""

import pandas as pd

# 1. Chargement du fichier nettoyé

df = pd.read_csv("online_retail_clean.csv", parse_dates=["InvoiceDate"])

# 2. Date de référence
# Pour calculer la Récence, il faut un point de repère dans le temps : "aujourd'hui".
# Comme les données datent de 2009-2011, on ne peut pas utiliser la vraie date
# du jour (tous les clients paraîtraient inactifs depuis 15 ans !). On utilise
# donc la date de la dernière commande du dataset + 1 jour, comme si on se
# plaçait "le lendemain de la dernière vente enregistrée".
date_reference = df["InvoiceDate"].max() + pd.Timedelta(days=1)
print(f"Date de référence utilisée pour le calcul de la Récence : {date_reference.date()}")

# 3. Calcul des 3 indicateurs, par client

rfm = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (date_reference - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("LineTotal", "sum"),
).reset_index()

# Explication simple de ce que chaque colonne veut dire, pour te la répéter
# facilement en entretien :
#   Recency   = nombre de jours depuis le dernier achat (plus petit = mieux)
#   Frequency = nombre de commandes distinctes passées (plus grand = mieux)
#   Monetary  = somme totale dépensée sur toute la période (plus grand = mieux)

print("\n--- Aperçu du tableau RFM ---")
print(rfm.head())

print(f"\nNombre de clients analysés : {rfm.shape[0]}")

# 4. Statistiques descriptives

print("\n--- Statistiques générales ---")
print(rfm[["Recency", "Frequency", "Monetary"]].describe().round(1))

# 5. SCORES RFM (transformation en note de 1 à 4, pour comparer les clients
#    entre eux facilement)
# On découpe chaque indicateur en 4 groupes de taille égale (quartiles) :
#   - Pour la Récence : le score 4 va aux clients ayant acheté le plus récemment
#   - Pour la Fréquence et le Montant : le score 4 va aux clients qui achètent
#     le plus souvent / dépensent le plus

rfm["R_score"] = pd.qcut(rfm["Recency"], 4, labels=[4, 3, 2, 1]).astype(int) #on inverse les labels car plus c'est bas mieux c'est
rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
rfm["M_score"] = pd.qcut(rfm["Monetary"], 4, labels=[1, 2, 3, 4]).astype(int)

# Score global RFM : simple somme des 3 scores (de 3 = client peu engagé,
# à 12 = client très engagé). Pratique pour un premier tri rapide.
rfm["RFM_score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]

print("\n--- Exemple de lecture des scores ---")
print("Un client avec RFM_score = 12 est un client très récent, très fréquent")
print("et à forte valeur : c'est un client à garder en priorité.")
print("Un client avec RFM_score = 3 n'a pas acheté depuis longtemps, achète")
print("rarement et dépense peu : c'est un client à faible engagement.")

# 6. Sauvegarde

rfm.to_csv("rfm_clients.csv", index=False)
print("\nFichier RFM sauvegardé : rfm_clients.csv")