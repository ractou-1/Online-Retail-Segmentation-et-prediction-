"""
Projet : Segmentation client & prédiction de churn - Online Retail II
Étape 1 : Chargement et nettoyage des données

Dataset : Online Retail II (UCI Machine Learning Repository, id=502)
Lien sur kaggle : https://www.kaggle.com/datasets/jillwang87/online-retail-ii

Colonnes attendues : InvoiceNo (A 6-digit integral number uniquely assigned to each transaction),
 StockCode (A 5-digit integral number uniquely assigned to each distinct product.),
 Description (Product (item) name.),
 Quantity (The quantities of each product (item) per transaction.),
 InvoiceDate (The day and time when a transaction was generated.),
 UnitPrice (Product price per unit in sterling ),
 CustomerID (A 5-digit integral number uniquely assigned to each customer.),
 Country (The name of the country where a customer resides.)
"""

import pandas as pd

# 1. CHARGEMENT

# Le dataset est livré en 2 fichiers (un par période) : on les charge et on les concatane
FICHIERS = [
    "online_retail_II_2009-2010.csv",
    "online_retail_II_2010-2011.csv",
]
morceaux = [pd.read_csv("online_retail_09_10.csv", encoding="ISO-8859-1"),
            pd.read_csv("online_retail_10_11.csv", encoding="ISO-8859-1")]

df = pd.concat(morceaux, ignore_index=True)

print(f"Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")
print(df.dtypes)
print(df.head())

# 2. Première exploration

print("\n--- Valeurs manquantes par colonne ---")
print(df.isna().sum())

df.columns.values[0] = "InvoiceNo"
print(f"\nNombre de factures d'annulation (commencent par 'C') : "
      f"{df['InvoiceNo'].astype(str).str.startswith('C').sum()}")

print(f"Quantités négatives ou nulles : {(df['Quantity'] <= 0).sum()}")
print(f"Prix unitaires négatifs ou nuls : {(df['UnitPrice'] <= 0).sum()}")

# 3. NETTOYAGE

df_clean = df.copy()
n_avant = len(df_clean)

# a) Retirer les annulations (InvoiceNo commençant par 'C')
df_clean = df_clean[~df_clean["InvoiceNo"].astype(str).str.startswith("C")]

# b) Retirer les lignes sans CustomerID (on ne peut pas les rattacher à un client
#    pour la segmentation ni la prédiction de churn)
df_clean = df_clean.dropna(subset=["CustomerID"])

# c) Retirer les quantités et prix aberrants (négatifs, nuls, ou tests internes à 0)
df_clean = df_clean[(df_clean["Quantity"] > 0) & (df_clean["UnitPrice"] > 0)]

# d) Retirer les doublons exacts
n_doublons = df_clean.duplicated().sum()
df_clean = df_clean.drop_duplicates()

# e) Conversion de types
df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])
df_clean["CustomerID"] = df_clean["CustomerID"].astype(int)

# f) Colonne montant de la ligne
df_clean["LineTotal"] = df_clean["Quantity"] * df_clean["UnitPrice"]

n_apres = len(df_clean)


# 4. Après nettoyage

print("\n--- Récapitulatif du nettoyage ---")
print(f"Lignes avant nettoyage       : {n_avant}")
print(f"Doublons exacts supprimés    : {n_doublons}")
print(f"Lignes après nettoyage       : {n_apres}")
print(f"Lignes supprimées au total   : {n_avant - n_apres} "
      f"({(n_avant - n_apres) / n_avant:.1%})")
print(f"Nombre de clients uniques    : {df_clean['CustomerID'].nunique()}")
print(f"Période couverte             : {df_clean['InvoiceDate'].min()} "
      f"-> {df_clean['InvoiceDate'].max()}")

# 5. Sauvegarde des données nettoyées

df_clean.to_csv("online_retail_clean.csv", index=False)
print("\nFichier nettoyé sauvegardé : online_retail_clean.csv")