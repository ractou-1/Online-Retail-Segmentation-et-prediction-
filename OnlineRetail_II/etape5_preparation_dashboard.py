"""
Projet : Segmentation client & prédiction de churn - Online Retail II
Étape 5 (préparation) : Export d'un fichier propre pour le dashboard

Avant de construire le dashboard dans Power BI, on prépare un seul fichier
final, propre et lisible, qui regroupe tout ce qui doit être visible :
le segment de chaque client, son risque de churn (s'il est calculé), et
ses indicateurs clés. L'idée est que Power BI n'ait plus qu'à lire ce
fichier et l'afficher, sans recalcul compliqué de son côté.
"""

import pandas as pd
import os

# 1. Chargement des segments (étape 3)

rfm = pd.read_csv("rfm_segments.csv")

# 2. Ajout des prédictions

churn = pd.read_csv("churn_predictions.csv")
export = rfm.merge(churn[["CustomerID", "Churn_proba"]], on="CustomerID", how="left")

# 3. Catégorie de risques lisibles

def categorie_risque(proba):
    if pd.isna(proba):
        return "Non calculé"
    elif proba >= 0.7:
        return "Risque élevé"
    elif proba >= 0.4:
        return "Risque modéré"
    else:
        return "Risque faible"

export["Risque_churn"] = export["Churn_proba"].apply(categorie_risque)

# 4. COLONNE DE VALEUR CLIENT LISIBLE (pour croiser valeur x risque dans
#    le dashboard : les clients "forte valeur + risque élevé" sont la
#    priorité absolue à afficher en évidence)

export["Valeur_client"] = pd.qcut(
    export["Monetary"], 3, labels=["Faible valeur", "Valeur moyenne", "Forte valeur"]
)

# 5. EXPORT FINAL

colonnes_utiles = [
    "CustomerID", "Recency", "Frequency", "Monetary",
    "Segment", "Segment_nom", "Valeur_client",
    "Churn_proba", "Risque_churn",
]
export = export[[c for c in colonnes_utiles if c in export.columns]]

export.to_csv("dashboard_export.csv", index=False)
print(f"\nFichier prêt pour Power BI : dashboard_export.csv ({len(export)} clients)")
