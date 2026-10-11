"""
Projet : Segmentation client & prédiction de churn - Online Retail II
Étape 7 : Visualisations en Python (remplace le dashboard Power BI)

Il produit 4 images dans le dossier "figures/" :
  1. Répartition des clients par segment
  2. Dépense moyenne par segment
  3. Matrice valeur client x risque de churn (où sont les clients à sauver)
  4. Liste des 15 clients prioritaires (forte valeur + risque le plus élevé)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. CHARGEMENT ET PARAMÈTRES D'AFFICHAGE

df = pd.read_csv("dashboard_export.csv")
os.makedirs("figures", exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.0)
PALETTE = sns.color_palette("Set2")

# 2. VISUEL 1 : Répartition des segments
# Répond à : "combien de clients dans chaque groupe ?"

repartition = df["Segment_nom"].value_counts()

fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(
    repartition.values,
    labels=repartition.index,
    autopct="%1.0f%%",
    startangle=90,
    colors=PALETTE[: len(repartition)],
    wedgeprops={"width": 0.4, "edgecolor": "white"},  # width < 1 => donut
)
ax.set_title("Répartition des clients par segment")
plt.tight_layout()
plt.savefig("figures/1_repartition_segments.png", dpi=150)
plt.close()

# 3. VISUEL 2 : dépenses moyennes par segement (barres)
# Répond à : "quels segments rapportent le plus ?"

depense_moyenne = (
    df.groupby("Segment_nom")["Monetary"].mean().sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(
    x=depense_moyenne.index, y=depense_moyenne.values,
    hue=depense_moyenne.index, palette="Set2", legend=False, ax=ax,
)
for i, valeur in enumerate(depense_moyenne.values):
    ax.text(i, valeur, f"{valeur:,.0f} £", ha="center", va="bottom", fontsize=9)
ax.set_xlabel("")
ax.set_ylabel("Dépense moyenne par client (£)")
ax.set_title("Dépense moyenne par segment")
plt.tight_layout()
plt.savefig("figures/2_depense_moyenne_segment.png", dpi=150)
plt.close()

# 4. VISUEL 3 : matrice valeur x risque churn 
# Répond à : "où sont les clients à forte valeur qui risquent de partir ?"
# La case en haut à droite (forte valeur + risque élevé) est la priorité.

if churn_disponible:
    ordre_valeur = ["Faible valeur", "Valeur moyenne", "Forte valeur"]
    ordre_risque = ["Risque faible", "Risque modéré", "Risque élevé"]

    matrice = pd.crosstab(df["Risque_churn"], df["Valeur_client"])
    matrice = matrice.reindex(index=ordre_risque, columns=ordre_valeur, fill_value=0)
    # On inverse l'axe vertical pour avoir "Risque élevé" en haut
    matrice = matrice.iloc[::-1]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        matrice, annot=True, fmt="d", cmap="Reds",
        cbar_kws={"label": "Nombre de clients"}, ax=ax,
    )
    # Encadre la case prioritaire (forte valeur + risque élevé) : en haut à droite.
    # Sa couleur est pâle car peu de clients y sont, mais c'est la plus importante.
    ax.add_patch(plt.Rectangle((2, 0), 1, 1, fill=False, edgecolor="#2E5266", lw=3))
    ax.set_xlabel("Valeur du client")
    ax.set_ylabel("Risque de churn")
    ax.set_title("Valeur client x risque de churn")
    plt.tight_layout()
    plt.savefig("figures/3_matrice_valeur_risque.png", dpi=150)
    plt.close()
else:
    print("Churn non calculé : visuels 3 et 4 ignorés (lance d'abord l'étape 4 puis l'étape 5).")

# 5. VISUEL 4 : TOP 15 clients prioritaires
if churn_disponible:
    prioritaires = (
        df[df["Valeur_client"] == "Forte valeur"]
        .sort_values("Churn_proba", ascending=False)
        .head(15)
        [["CustomerID", "Segment_nom", "Monetary", "Churn_proba"]]
        .copy()
    )
    prioritaires["Monetary"] = prioritaires["Monetary"].map(lambda v: f"{v:,.0f} £")
    prioritaires["Churn_proba"] = prioritaires["Churn_proba"].map(lambda p: f"{p:.0%}")
    prioritaires.columns = ["Client", "Segment", "Dépense totale", "Risque de churn"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axis("off")
    tableau = ax.table(
        cellText=prioritaires.values,
        colLabels=prioritaires.columns,
        cellLoc="center", loc="center",
    )
    tableau.auto_set_font_size(False)
    tableau.set_fontsize(9)
    tableau.scale(1, 1.4)
    for (ligne, _), cellule in tableau.get_celld().items():
        if ligne == 0:
            cellule.set_facecolor("#2E5266")
            cellule.set_text_props(color="white", weight="bold")
    ax.set_title("Top 15 des clients prioritaires à relancer", pad=10)
    plt.tight_layout()
    plt.savefig("figures/4_clients_prioritaires.png", dpi=150)
    plt.close()

    # Même liste en CSV : directement utilisable par une équipe marketing
    prioritaires.to_csv("figures/clients_prioritaires.csv", index=False)

print("Figures sauvegardées dans le dossier figures/")
