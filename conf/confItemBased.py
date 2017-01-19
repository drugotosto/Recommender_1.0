__author__ = 'maury'

"""
Valori "migliori" di WeightSim per i diversi RS sono:
ItemBased: 10
TagBased: 15 (con weightSim più alto crolla il coverage utente)
"""
# Parametro x il SIGNIFICANCE WEIGHTING (pesare le somiglianze a seconda del numero di users/tags in comune)
weightSim=5
# Parametro relativo al numero di Vicini da prendere in considerazione nella predizione del rate
nNeigh=10
# Settaggio del tipo di misura di somiglianza da adottare (Cosine/Pearson)
typeSimilarity="Pearson"