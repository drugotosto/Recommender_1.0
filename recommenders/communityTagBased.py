import shutil

__author__ = 'maury'

import json
import os

from conf.confItemBased import weightSim
from recommenders.tagBased import TagBased
from recommenders.communityBased import CommunityBased
from conf.confDirFiles import userTagJSON, dirPathInput, dirPathCommunities


class CommunityTagBased(CommunityBased,TagBased):
    def __init__(self,name,friendships,communityType):
        CommunityBased.__init__(self,name,friendships,communityType)

    def builtModel(self,spEnv,directory):
        """
        Costruzione del modello a secondo l'approccio CF ItemCommunityBased
        :return:
        """
        """
        Calcolo media dei Ratings (per ogni user) e creazione della corrispondente broadcast variable
        """
        # Unisco tutti i dati (da tutti i files contenuti nella directory train_k) ottengo (user,[(item,score),(item,score),...]
        user_item_pair=spEnv.getSc().textFile(directory+"/*").map(lambda line: CommunityTagBased.parseFileUser(line)).groupByKey()
        user_meanRatesRatings=user_item_pair.map(lambda p: CommunityTagBased.computeMean(p[0],p[1])).collectAsMap()
        dictUser_meanRatesRatings=spEnv.getSc().broadcast(user_meanRatesRatings)

        """
        Calcolo media dei valori associati ai Tags (per ogni user) e creazione della corrispondente broadcast variable (Utilizzo per misura Pearson)
        """
        user_tagVal_pairs=spEnv.getSc().textFile(userTagJSON).map(lambda line: CommunityTagBased.parseFileUser(line)).groupByKey()
        user_meanRatesTags=user_tagVal_pairs.map(lambda p: CommunityTagBased.computeMean(p[0],p[1])).collectAsMap()
        dictUser_meanRatesTags=spEnv.getSc().broadcast(user_meanRatesTags)
        # print("\nDIZ COM: {}".format(user_meanRatesTags))

        """
        Calcolo delle somiglianze tra users in base ai TAGS e creazione del corrispondente RDD (Recupero tutti i vicini "filtrati")
        """
        if not os.path.exists(dirPathInput+"user_simsOrd(weightSim="+str(weightSim)+")/"):
            print("\nNon esistono ancora i valori di somiglianza tra Users. Vado a calcolarli!")
            nNeigh=self.nNeigh
            # Ottengo l'RDD con elementi (tag,[(user1,score),(user2,score),...]
            tag_userVal_pairs=spEnv.getSc().textFile(userTagJSON).map(lambda line: TagBased.parseFileItem(line)).groupByKey().cache()
            user_simsTags=self.computeSimilarityTag(spEnv,tag_userVal_pairs,dictUser_meanRatesTags.value).map(lambda p: TagBased.filterSimilarities(p[0],p[1])).filter(lambda p: p!=None)
            user_simsTags.map(lambda x: json.dumps(x)).saveAsTextFile(dirPathInput+"user_simsOrd(weightSim="+str(weightSim)+")/")
        else:
            print("\nLa somiglianza tra Users e già presente")
            user_simsTags=spEnv.getSc().textFile(dirPathInput+"user_simsOrd(weightSim="+str(weightSim)+")/").map(lambda x: json.loads(x))

        # print("\nSOMIGLIANZE tra Users in base a TAGS:")
        # for user,user_valPairs in user_simsTags.take(5):
        #     print("\nUser : {}".format(user))
        #     print("User - PairVal :{}".format(list(user_valPairs)))

        """
        Calcolo delle somiglianze tra users in base alle COMMUNITIES e creazione del corrispondente RDD (Recupero tutti i vicini "filtrati")
        """
        if self.getCurrentFold()==0 and not os.path.exists(dirPathCommunities+"/"+self.communityType+"/user_simsOrd/"):
            nNeigh=self.nNeigh
            # Recupero RDD con l'elenco delle communities associate ognuna ad una lista di utenti di cui ne fanno parte
            comm_listUsers=spEnv.getSc().textFile(dirPathCommunities+"/"+self.communityType+"/communitiesFriends.json").map(lambda x: json.loads(x))
            # Calcolo del valore di somiglianza tra tutti gli utenti appartenenti alla stessa communities ritorno i primi "N" Amici più simili: (user,[(user,valSim),...])
            user_simsOrd=self.computeSimilarityFriends(spEnv,comm_listUsers).map(lambda p: CommunityBased.filterSimilarities(p[0],p[1])).filter(lambda p: p!=None).map(lambda p: CommunityBased.nearestFriendsNeighbors(p[0],p[1],nNeigh))
            user_simsOrd.map(lambda x: json.dumps(x)).saveAsTextFile(dirPathCommunities+"/"+self.communityType+"/user_simsOrd/")

        user_simsFriends=spEnv.getSc().textFile(dirPathCommunities+"/"+self.communityType+"/user_simsOrd/").map(lambda x: json.loads(x))

        # print("\nSOMIGLIANZE tra Users in base a AMICI:")
        # for user,user_valPairs in user_simsFriends.take(5):
        #     print("\nUser : {}".format(user))
        #     print("User - PairVal :{}".format(list(user_valPairs)))

        """
        Creazione RDD delle somiglianze finale che tiene conto dei due RDD.
        """
        # Modifica dell'RDD relativo ai TAGS
        user_simsTags=user_simsTags.mapValues(lambda x: CommunityTagBased.RemoveTags(x))
        # print("\nSemplificato RDD_TAGS Somiglianze!")
        nNeigh=self.nNeigh
        user_simsTot=user_simsFriends.union(user_simsTags).reduceByKey(lambda listPair1,listPair2: CommunityTagBased.joinPairs(listPair1,listPair2)).filter(lambda p: p!=None).map(lambda p: CommunityTagBased.nearestTagsNeighbors(p[0],p[1],nNeigh)).cache()
        # print("\nHo finito di calcolare valori di Somiglianze Globali tra utenti!")
        # print("\nUSER SIM_GLOB: {}".format(user_simsTot.take(10)))

        """
        Calcolo delle raccomandazioni personalizzate per i diversi utenti
        """
        user_item_hist=user_item_pair.collectAsMap()
        userHistoryRates=spEnv.getSc().broadcast(user_item_hist)
        # Calcolo per ogni utente la lista di TUTTI gli items suggeriti ordinati secondo predizione. Ritorno un pairRDD del tipo (user,[(scorePred,item),(scorePred,item),...])
        user_item_recs = user_simsTot.map(lambda p: CommunityTagBased.recommendationsUserBasedSocial(p[0],p[1],userHistoryRates.value,dictUser_meanRatesRatings.value)).map(lambda p: CommunityTagBased.convertFloat_Int(p[0],p[1])).collectAsMap()
        # Immagazzino la lista dei suggerimenti finali prodotti per sottoporla poi a valutazione
        self.setDictRec(user_item_recs)
        # print("\nLista suggerimenti: {}".format(self.dictRec))

    @staticmethod
    def RemoveTags(listPairs):
        return [(user,pair[0]) for user,pair in listPairs if pair[0]>0.0]

    @staticmethod
    def joinPairs(listPair1,listPair2):
        """
        Unisco l'RDD delle amicizie a quello dei tag in base allo stesso user andando a sommare
        i valori di somiglianza nel momento in cui il vicino è lo stesso in entrambi gli RDD, in
        caso contrario mantengo solamente i vicini rispetto alle amicizie
        (Se oltre ad essere somigliante per gusti lo è anche per via delle amicizie dò maggiore importanza)
        :param listPair1: lista di coppie (user,valSim) per RDD dei friends
        :param listPair2: lista di coppie (user,valSim) per RDD dei tags
        :return:
        """
        lista=[]
        for pair1 in listPair1:
            valSim=pair1[1]
            for pair2 in listPair2:
                # Stesso vicino in entrambi gli RDD
                if pair1[0]==pair2[0]:
                    valSim=pair1[1]+pair2[1]
                    break
            lista.append((pair1[0],valSim))
        return lista

