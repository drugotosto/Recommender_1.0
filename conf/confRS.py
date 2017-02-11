__author__ = 'maury'

""" File di configurazione generico del RS. """
typeRecommender="CommunityBased"
# Numero di Folds in cui si dividono gli users/ratings per effettuare la valutazione del RS (K-Fold Cross-Validation)
nFolds=5
# Parametro relativo al numero di elementi della lista dei sugerimenti finali rilasciati all'utente
topN=10
# Scelta della tag su cui andare a costruire il corrispondente DataFrame/RDD
# tag="Nightlife"
tag="Restaurants"
# Categorie di Default da   non prendere in considerazione quando si filtra il DataSet
tagToFilter=["Restaurants","Bars"]
# Numero di recensioni minime (con rate) che ogni utente deve aver stilato
numRec=20
# Percentuale del numero di rates da prendere (per ogni utente) che faranno parte del test set sul numero totale di ratings disponibili
percTestRates=0.2


