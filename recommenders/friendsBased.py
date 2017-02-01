import shutil

__author__ = 'maury'

import json
from collections import defaultdict
import os
import operator

from tools.sparkEnvLocal import SparkEnvLocal
from recommenders.itemBased import ItemBased
from recommenders.tagBased import TagBased
from conf.confDirFiles import dirPathInput
from recommenders.socialBased import SocialBased

class FriendsBased(ItemBased):
    def __init__(self,name,friendships):
        ItemBased.__init__(self,name=name)
        # Creazione del dizionario delle amicizie che ad ogni amico associa un valore di somiglianza data dalla Jaccard metrix
        self.friendships=self.createDizFriendships(friendships)

    def builtModel(self,spEnv,directory):
        """
        Costruzione del modello a secondo l'approccio CF ItemBased
        :param spEnv: SparkContext di riferimento
        :type spEnv: SparkEnvLocal
        :param directory: Directory che contiene insieme di File che rappresentano il TestSet
        :return:
        """
        """
        Calcolo media dei Ratings (per ogni user) e creazione della corrispondente broadcast variable
        """
        # Unisco tutti i dati (da tutti i files contenuti nella directory train_k) ottengo (user,[(item,score),(item,score),...]
        user_item_pair=spEnv.getSc().textFile(directory+"/*").map(lambda line: ItemBased.parseFileUser(line)).groupByKey()
        user_meanRatesRatings=user_item_pair.map(lambda p: ItemBased.computeMean(p[0],p[1])).collectAsMap()
        dictUser_meanRatesRatings=spEnv.getSc().broadcast(user_meanRatesRatings)

        """
        Calcolo delle somiglianze tra ogni utente e i suoi friends in base alle amicizie in comune e creazione del corrispondente RDD
        """
        if os.path.exists(dirPathInput+"/FriendsSim/"):
            shutil.rmtree(dirPathInput+"/FriendsSim/")

        # print("\nfriendships: {}".format(self.friendships.items()))
        # Creo RDD che rappresenta la lista di utenti del dataset con associati friends e relativo valore di Somiglianza (Solo Filtraggio Senza Top-N vicini)
        users=spEnv.getSc().parallelize(self.friendships.items()).map(lambda p: SocialBased.filterSimilarities(p[0],p[1])).filter(lambda p: p!=None)
        users.map(lambda x: json.dumps(x)).saveAsTextFile(dirPathInput+"/FriendsSim/")

        user_simsOrd=spEnv.getSc().textFile(dirPathInput+"/FriendsSim/").map(lambda x: json.loads(x))

        # print("\nDOPO")
        # print(user_sims.take(2))
        # for user,user_valPairs in user_sims.take(1):
        #     print("\nUser : {}".format(user))
        #     print("User - PairVal :{}".format(list(user_valPairs)))

        """
        Calcolo delle raccomandazioni personalizzate per i diversi utenti
        """
        user_item_hist=user_item_pair.collectAsMap()
        userHistoryRates=spEnv.getSc().broadcast(user_item_hist)
        # Calcolo per ogni utente la lista di TUTTI gli items suggeriti ordinati secondo predizione. Ritorno un pairRDD del tipo (user,[(scorePred,item),(scorePred,item),...])
        user_item_recs = user_simsOrd.map(lambda p: SocialBased.recommendationsUserBasedSocial(p[0],p[1],userHistoryRates.value,dictUser_meanRatesRatings.value)).map(lambda p: TagBased.convertFloat_Int(p[0],p[1])).collectAsMap()
        # Immagazzino la lista dei suggerimenti finali prodotti per sottoporla poi a valutazione
        self.setDictRec(user_item_recs)
        print("\nLista di raccomandazioni calcolata!")
        # print("\nLista suggerimenti: {}".format(self.dictRec))


    @staticmethod
    def nearestFriendsNeighbors(user_id,users_and_sims,n):
        """
        Per ogni utente faccio ritornare solamente i primi "n" amici con somiglianza più grande
        :param user_id: Active user
        :param users_and_sims: lista di pairs (user,ValSim)
        :param n: Numero di vicini da far ritornare
        :return: RDD del tipo (activeUser,[(user,ValSim),...])
        """
        s=sorted(users_and_sims,key=operator.itemgetter(1),reverse=True)
        return user_id, s[:n]


    def createDizFriendships(self,friendships):
        def createPairs(user,listFriends):
            return [(user,friend) for friend in listFriends]

        """ Creo gli archi del grafo mancanti """
        listaList=[createPairs(user,listFriends) for user,listFriends in friendships.items()]
        archiPresenti={coppia for lista in listaList for coppia in lista}
        archiMancanti={(arco[1],arco[0]) for arco in archiPresenti if (arco[1],arco[0]) not in archiPresenti}
        # print("\n- Numero di archi mancanti: {}".format(len(archiMancanti)))
        archiDoppi=archiPresenti.union(archiMancanti)
        # print("\n- Numero di archi/Amicizie (doppie) totali presenti sono: {}".format(len(archiDoppi)))

        """ Costruisco il dizionario con ARCHI DOPPI senza peso sugli archi """
        dizFriendshipsDouble=defaultdict(list)
        for k, v in archiDoppi:
            dizFriendshipsDouble[k].append(v)
        # print("\n- Numero di utenti: {}".format(len([user for user in dizFriendshipsDouble])))

        """ Costruisco il dizionario che per ogni utente contiene la lista degli amici con peso associato dato da Jaccard similarity """
        def createListFriendsDoubleWeight(user,dizFriendshipsDouble):
            return [(friend,(len(set(dizFriendshipsDouble[user])&(set(dizFriendshipsDouble[friend])))+1)/(len(set(dizFriendshipsDouble[user])|(set(dizFriendshipsDouble[friend])))+1)) for friend in dizFriendshipsDouble[user]]

        dictFriendships={user:createListFriendsDoubleWeight(user,dizFriendshipsDouble) for user in dizFriendshipsDouble}

        print("\nNumero di AMICIZIE (doppie) presenti sono: {}".format(sum([len(lista) for lista in dictFriendships.values()])))
        print("\nNumero di UTENTI che hanno almeno 1 amico: {}".format(len(list(dictFriendships.keys()))))
        print("\nDizionario delle amicizie pesato creato e settato!")
        return dictFriendships

    def setFriendships(self,friendships):
        self.friendships=friendships

    def getFriendships(self):
        return self.friendships

    def getCommunityType(self):
        return self.communityType

    def setCommunityType(self,type):
        self.communityType=type

