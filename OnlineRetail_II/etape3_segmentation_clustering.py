"""
Projet : Segmentation client & prédiction de churn - Online Retail II
Étape 3 : Segmentation client (clustering)

Le clustering est une méthode qui regroupe automatiquement les clients qui
se ressemblent, sans qu'on ait besoin de définir les groupes à l'avance.
On lui donne les 3 indicateurs RFM de chaque client (Récence, Fréquence,
Montant), et l'algorithme forme des paquets de clients aux profils proches.

Ici, l'algorithme utilisé s'appelle K-means. On doit juste lui indiquer
combien de groupes on veut. Pour choisir ce nombre, on teste
plusieurs valeurs et on regarde à partir de quand ajouter un groupe
supplémentaire n'apporte plus grand-chose (la méthode du coude).
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Chargement du fichier étape 2

rfm = pd.read_csv("rfm_clients.csv")

# 2. Standardisation
# La standardisation ramène les 3 indicateurs à une échelle comparable, pour que chacun compte à poids égal.

features = ["Recency", "Frequency", "Monetary"]
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[features])

# 3. Choix du nbr de groupes
# On teste plusieurs valeurs de K et on mesure à chaque fois l'inertie
# (à quel point les clients d'un même groupe sont proches les uns des
# autres). Plus K augmente, plus l'inertie baisse — mais on
# cherche le point où ça n'apporte plus grand-chose (le coude de la
# courbe), qui donne un bon compromis simplicité / pertinence.

inerties = []
valeurs_k = range(2, 8)
for k in valeurs_k:
    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_test.fit(rfm_scaled)
    inerties.append(kmeans_test.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(list(valeurs_k), inerties, marker="o")
plt.xlabel("Nombre de groupes (K)")
plt.ylabel("Inertie")
plt.title("Méthode du coude - choix du nombre de segments")
plt.tight_layout()
plt.savefig("methode_du_coude.png", dpi=150)
print("Graphique sauvegardé : methode_du_coude.png")

# 4. Clustring final

K_CHOISI = 4

kmeans = KMeans(n_clusters=K_CHOISI, random_state=42, n_init=10)
rfm["Segment"] = kmeans.fit_predict(rfm_scaled)

# 5. Profil de chaque segment (moyennes RFM par groupe)

profil_segments = rfm.groupby("Segment")[features].mean().round(1)
profil_segments["Nb_clients"] = rfm.groupby("Segment").size()

print("\n--- Profil moyen de chaque segment ---")
print(profil_segments)

# 6. Nommage des segments
# IMPORTANT : les numéros de segment (0, 1, 2, 3) sortent dans un ordre
# arbitraire à chaque exécution. Regarde le tableau "profil_segments"
# affiché ci-dessus et adapte le dictionnaire ci-dessous en conséquence
# avant de continuer (ex. le segment avec la plus petite Récence et le plus
# gros Montant est probablement tes "clients fidèles").

noms_segments = {
    0: "Clients occasionnels", #Clients avec des fréquences moyennes
    1: "Clients inactifs", #Dernier achat très ancien, faible fréquence et faible valeur
    2: "Gros clients", #Très forte fréquence et valeur, achats extrêmement récents
    3: "Clients fidèles", #Très actifs, très fréquents et forte valeur
}
rfm["Segment_nom"] = rfm["Segment"].map(noms_segments)

# 7. Sauvegarde

rfm.to_csv("rfm_segments.csv", index=False)
print("\nFichier segmenté sauvegardé : rfm_segments.csv")
