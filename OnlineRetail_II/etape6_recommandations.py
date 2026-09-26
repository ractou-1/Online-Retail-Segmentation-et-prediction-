"""
Projet : Segmentation client & prédiction de churn - Online Retail II
Étape 6 : Recommandations business chiffrées

Cette dernière étape ne rajoute pas de modèle : elle traduit les résultats
déjà obtenus (segments + risque de churn) en une recommandation concrète et
chiffrée, comme le ferait un consultant à la fin d'une mission. L'idée :
ne pas s'arrêter à "voici un dashboard", mais dire clairement quoi faire,
POUR qui, et avec quel gain attendu.
"""

import pandas as pd

# 1. Chargement du fichier

df = pd.read_csv("dashboard_export.csv")

# 2. Identification de la cible prioritaire
# Les clients "forte valeur" ET "risque élevé" de churn : ce sont ceux dont
# la perte coûterait le plus cher à l'entreprise, et pour qui une action de
# relance a le plus de sens économique.

cible = df[(df["Valeur_client"] == "Forte valeur") & (df["Risque_churn"] == "Risque élevé")]

nb_clients_cible = len(cible)
valeur_totale_cible = cible["Monetary"].sum()

print(f"Clients à forte valeur ET à risque élevé de churn : {nb_clients_cible}")
print(f"Valeur historique cumulée de ces clients : {valeur_totale_cible:,.0f} £")

# 3. ESTIMATION DU GAIN D'UNE CAMPAGNE DE RELANCE CIBLÉE
# Hypothèse de taux de récupération : à ajuster / discuter avec un
# interlocuteur métier dans un cas réel. On prend ici 25% par prudence
# (ordre de grandeur bas-moyen pour une campagne de relance ciblée).

TAUX_RECUPERATION_HYPOTHESE = 0.25

clients_recuperables_estimes = nb_clients_cible * TAUX_RECUPERATION_HYPOTHESE
gain_estime = valeur_totale_cible * TAUX_RECUPERATION_HYPOTHESE

print(f"\nHypothèse de taux de récupération de la campagne : {TAUX_RECUPERATION_HYPOTHESE:.0%}")
print(f"Clients potentiellement retenus (estimation) : {clients_recuperables_estimes:.0f}")
print(f"Gain potentiel estimé : {gain_estime:,.0f} £")

# 4. Action différente (ONBOARDING)
# Deuxième recommandation, moins coûteuse à mettre en place : un onboarding
# renforcé pour transformer plus de nouveaux clients en clients fidèles.

nouveaux = df[df["Segment_nom"] == "Clients occasionnels"]
print(f"\nNombre de nouveaux clients identifiés : {len(nouveaux)}")

# 5. RECOMMANDATIONS FINALES

print("\n" + "=" * 70)
print("RECOMMANDATIONS")
print("=" * 70)
print(f"""
1. Lancer une campagne de relance ciblée sur les {nb_clients_cible} clients
   à forte valeur et à risque élevé de churn (valeur historique cumulée de
   {valeur_totale_cible:,.0f} £). Avec un taux de récupération estimé à
   {TAUX_RECUPERATION_HYPOTHESE:.0%} (hypothèse à valider), le gain potentiel
   est de l'ordre de {gain_estime:,.0f} £.

2. Mettre en place un programme d'onboarding renforcé (ex. email de bienvenue
   personnalisé, offre de bienvenue sur le 2e achat) pour les
   {len(nouveaux)} nouveaux clients identifiés, afin d'augmenter la
   proportion qui devient fidèle plutôt que ponctuelle.

""")
