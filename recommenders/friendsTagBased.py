__author__ = 'maury'

import json
import os
import numpy as np
from collections import defaultdict
from numpy import dot
from statistics import mean
from numpy.linalg import norm
import operator
import shutil

from recommenders.friendsBased import FriendsBased
from conf.confDirFiles import dirPathInput
from recommenders.socialBased import SocialBased
from tools.tools import deep_getsizeof


class FriendsTagBased(FriendsBased):
    def __init__(self,name,friendships,dictBus_Tags):
        FriendsBased.__init__(self,name,friendships)
        self.dictBus_Tags=dictBus_Tags

    def builtModel(self,spEnv,directory):
        """
        Costruzione del modello a secondo l'approccio CF ItemSocialBased
        :return:
        """
        """
        Calcolo media dei Ratings (per ogni user) e creazione della corrispondente broadcast variable
        """
        # Unisco tutti i dati (da tutti i files contenuti nella directory train_k) ottengo (user,[(item,score),(item,score),...]
        user_item_pair=spEnv.getSc().textFile(directory+"/*").map(lambda line: FriendsBased.parseFileUser(line)).groupByKey()
        user_meanRatings=user_item_pair.map(lambda p: FriendsBased.computeMean(p[0],p[1])).collectAsMap()
        dictUser_meanRatings=spEnv.getSc().broadcast(user_meanRatings)

        """
        Calcolo del dizionario che per ogni utente rappresenta la somiglianza di gusti dei vari amici in funzione dei diversi tags
        """
        # Creazione del dizionario che rappresenta lo storico dei ratings dati dai vari utenti
        dictUserRatings=self.retrieveTrainData(directory)
        # print("\nuser_item_pair: {} - Size: {}".format(dictUserRatings,deep_getsizeof(dictUserRatings, set())))
        dictUser_Tag_Friends=self.computeUserTagSimilarity(dictUserRatings,dictUserRatings.copy(),user_meanRatings)
        user_Tag_Friends=spEnv.getSc().broadcast(dictUser_Tag_Friends)
        # print("\ndictUser_Tag_Friends {}".format(dictUser_Tag_Friends))

        """
        Calcolo delle somiglianze tra ogni utente e i suoi friends in base alle amicizie in comune e creazione del corrispondente RDD
        """
        if os.path.exists(dirPathInput+"/FriendsSim/"):
            shutil.rmtree(dirPathInput+"/FriendsSim/")

        # print("\nfriendships: {}".format(self.friendships.items()))
        # Creo RDD che rappresenta la lista di utenti del dataset con associati friends e relativo valore di Somiglianza
        users=spEnv.getSc().parallelize(self.friendships.items()).map(lambda p: SocialBased.filterSimilarities(p[0],p[1])).filter(lambda p: p!=None)
        users.map(lambda x: json.dumps(x)).saveAsTextFile(dirPathInput+"/FriendsSim/")

        user_simsFriends=spEnv.getSc().textFile(dirPathInput+"/FriendsSim/").map(lambda x: json.loads(x))

        # print("\nSOMIGLIANZE tra Users in base a AMICI:")
        # for user,user_valPairs in user_simsFriends.take(5):
        #     print("\nUser : {}".format(user))
        #     print("User - PairVal :{}".format(list(user_valPairs)))

        """
        Calcolo delle raccomandazioni personalizzate per i diversi utenti
        """
        user_item_hist=user_item_pair.collectAsMap()
        userHistoryRates=spEnv.getSc().broadcast(user_item_hist)
        dictBus_Tags=self.dictBus_Tags
        # Calcolo per ogni utente la lista di TUTTI gli items suggeriti ordinati secondo predizione. Ritorno un pairRDD del tipo (user,[(scorePred,item),(scorePred,item),...])
        user_item_recs = user_simsFriends.map(lambda p: FriendsTagBased.recommendationsUserBasedFriends(p[0],p[1],userHistoryRates.value,dictUser_meanRatings.value,user_Tag_Friends.value,dictBus_Tags)).map(lambda p: FriendsBased.convertFloat_Int(p[0],p[1])).collectAsMap()
        # Immagazzino la lista dei suggerimenti finali prodotti per sottoporla poi a valutazione
        self.setDictRec(user_item_recs)
        # print("\nLista suggerimenti: {}".format(self.dictRec))

    def computeUserTagSimilarity(self,dictUserRatings,dictUserRatingsCopy,user_meanRatesRatings):
        """
        Creazione del dizionario che ad ogni tag di ogni user associa la lista degli amici per i quali si è riscontrata somiglianza di gusti
        :param dictUserRatings: dizionario dello storico dei ratings degli utenti
        :return:
        """
        dictUser_Tag_Friends=defaultdict(dict)
        listLenCommItems=[]
        numOkConf=0
        numTotConf=0
        numUtenti=0
        # Ciclo su tutti gli utenti del TrainSet
        for user,dictRatingsUser in dictUserRatings.items():
            # print("\nUser: {}".format(user))
            # print("\nratingsUser: {}".format(ratingsUser))
            itemsUser=dictRatingsUser.keys()
            # Ciclo su tutti gli amici del dato utente se presenti
            if user in self.friendships and len(self.friendships[user])>0:
                numUtenti+=1
                for friend in list(zip(*self.friendships[user]))[0]:
                    numTotConf+=1
                    dictRatingsFriend=dictUserRatingsCopy[friend]
                    itemsFriend=dictRatingsFriend.keys()
                    commItems=set(itemsUser)&set(itemsFriend)
                    if len(commItems)>0:
                        numOkConf+=1
                        # Aggiungo il numero degli items condivisi dagli utenti
                        listLenCommItems.append(len(commItems))
                        for tag in {tag for item in commItems for tag in self.dictBus_Tags[item]}:
                            ratesUser=[dictRatingsUser[item] for item in commItems if tag in self.dictBus_Tags[item]]
                            ratesFriend=[dictRatingsFriend[item] for item in commItems if tag in self.dictBus_Tags[item]]
                            valSim=self.pearsonTagSimilarity(np.array(ratesUser),user,np.array(ratesFriend),friend,user_meanRatesRatings)
                            if valSim>=0.5:
                                if tag not in dictUser_Tag_Friends[user]:
                                    dictUser_Tag_Friends[user][tag]=[]
                                dictUser_Tag_Friends[user][tag].append(friend)
                # print("\nFinito di elaborare utente! Num Utenti elaborati: {}".format(numUtenti))
        print("\nNumero medio di business in comune: {} - Percentule confronti tra utenti OK: {}".format(mean(listLenCommItems),(numOkConf/numTotConf)))
        return dictUser_Tag_Friends

    def pearsonTagSimilarity(self,ratesUser,user,ratesFriend,friend,user_meanRatesRatings):
        """
        Calcolo della somiglianza in base alla metrica Pearson tra l'utente e l'amico rispetto ad un certo tag
        :param ratesUser: rates dati dall'utente
        :param user: active user
        :param ratesFriend: rates dati dall'amico
        :param friend: amico
        :param user_meanRatesRatings: medie voti degli utenti
        :return:
        """
        den=norm((ratesUser-user_meanRatesRatings[user]))*norm((ratesFriend-user_meanRatesRatings[friend]))
        return dot((ratesUser-user_meanRatesRatings.get(user)),(ratesFriend-user_meanRatesRatings.get(friend)))/den

    @staticmethod
    def recommendationsUserBasedFriends(user_id,users_with_sim,userHistoryRates,user_meanRates,dictUser_Tag_Friends,dictBus_Tags):
        """
        Per ogni utente ritorno una lista (personalizzata) di items sugeriti in ordine di rate.
        N.B: Per alcuni user non sarà possibile raccomandare alcun item -> Lista vuota
        :param user_id: utente per il quale rilasciare la lista di raccomandazioni
        :param users_with_sim: Lista di elementi del tipo [(user,valSim),(user,ValSim),...]
        :param userHistoryRates: Dizionario del tipo user:[(item,score),(item,score),...]
        :param user_meanRates: Dizionario che per ogni utente contiene valore Medio Rating
        :param dictUser_Tag_Friends: Dizionario che ad ogni tag di ogni utente associa la lista di amici da considerare
        :return:
        """
        totals = defaultdict(int)
        sim_sums = defaultdict(int)
        # Ciclo su tutti i vicini dell'utente
        for (vicino,sim) in users_with_sim:
            # Recupero tutti i rates dati dal tale vicino dell'utente
            listRatings = userHistoryRates.get(vicino)
            if listRatings:
                # Ciclo su tutti i Rates
                for (item,rate) in listRatings:
                    considero=False
                    # Controllo se l'amico è nella lista associata ad almeno uno dei tags del business preso in considerazione
                    for tag in dictBus_Tags[item]:
                        if tag in dictUser_Tag_Friends[user_id] and vicino in dictUser_Tag_Friends[user_id][tag]:
                            considero=True
                            break
                    if considero:
                        # Aggiorno il valore di rate e somiglianza per l'item preso in considerazione
                        totals[item] += sim * (rate-user_meanRates.get(vicino))
                        sim_sums[item] += abs(sim)
        """
            Rilascio una lista di suggerimenti composta da tutti gli items votati dagli amici dell'active user appartenenti alla stessa community
        """
        # Creo la lista dei rates normalizzati associati agli items per ogni user
        scored_items = [(user_meanRates.get(user_id)+(total/sim_sums[item]),item) for item,total in totals.items()]
        # Ordino la lista secondo il valore dei rates
        return user_id,sorted(scored_items,key=operator.itemgetter(0),reverse=True)