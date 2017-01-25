
__author__ = 'maury'

from collections import defaultdict, Counter
import json
import os
from statistics import mean
from igraph import Graph
import time
import operator
import math
import numpy as np

from tools.sparkEnvLocal import SparkEnvLocal
from recommenders.socialBased import SocialBased
from recommenders.tagBased import TagBased
from conf.confCommunitiesFriends import communitiesTypes, communityType
from conf.confDirFiles import dirPathCommunities, dirPathInput, userTagJSON, userFriendsGraph
from conf.confItemBased import weightSim
from recommenders.tagSocialImpersonalBased import TagSocialImpersonalBased
from tools.tools import saveJsonData

class TagSocialPersonalBased(SocialBased,TagBased):
    def __init__(self,name,friendships,communityType):
        SocialBased.__init__(self,name,friendships,communityType)
        # Numero di utenti che presentano almeno 1 clusers di utenti
        self.numUtentiWithClusters=None
        # Numero medio dei Clusters trovati usando l'algoritmo di detection community definito
        self.numMedioClusters=None
        # Granezza media dei Clusters trovati usando l'algoritmo di detection community definito
        self.sizeMedioClusters=None

    def builtModel(self,spEnv,directory):
        """
        Costruzione del modello a secondo l'approccio CF ItemSocialBased
        :return:
        """
        """
        Calcolo media dei Ratings (per ogni user) e creazione della corrispondente broadcast variable
        """
        # Unisco tutti i dati (da tutti i files contenuti nella directory train_k) ottengo (user,[(item,score),(item,score),...]
        user_item_pair=spEnv.getSc().textFile(directory+"/*").map(lambda line: TagSocialImpersonalBased.parseFileUser(line)).groupByKey()
        user_meanRatesRatings=user_item_pair.map(lambda p: TagSocialImpersonalBased.computeMean(p[0],p[1])).collectAsMap()
        dictUser_meanRatesRatings=spEnv.getSc().broadcast(user_meanRatesRatings)

        """
        Calcolo media dei valori associati ai Tags (per ogni user) e creazione della corrispondente broadcast variable (Utilizzo per misura Pearson)
        """
        user_tagVal_pairs=spEnv.getSc().textFile(userTagJSON).map(lambda line: TagSocialImpersonalBased.parseFileUser(line)).groupByKey()
        user_meanRatesTags=user_tagVal_pairs.map(lambda p: TagSocialImpersonalBased.computeMean(p[0],p[1])).collectAsMap()
        dictUser_meanRatesTags=spEnv.getSc().broadcast(user_meanRatesTags)
        # print("\nDIZ COM: {}".format(user_meanRatesTags))

        """
        Calcolo delle somiglianze tra users in base ai TAGS e creazione del corrispondente RDD
        """
        if not os.path.exists(dirPathInput+"user_simsOrd(weightSim="+str(weightSim)+")/"):
            print("\nNon esistono ancora i valori di somiglianza tra Users. Vado a calcolarli!")
            nNeigh=self.nNeigh
            # Ottengo l'RDD con elementi (tag,[(user1,score),(user2,score),...]
            tag_userVal_pairs=spEnv.getSc().textFile(userTagJSON).map(lambda line: TagBased.parseFileItem(line)).groupByKey().cache()
            user_simsTags=self.computeSimilarityTag(spEnv,tag_userVal_pairs,dictUser_meanRatesTags.value).map(lambda p: TagBased.filterSimilarities(p[0],p[1])).map(lambda p: TagBased.nearestTagsNeighbors(p[0],p[1],nNeigh)).filter(lambda p: p!=None)
            user_simsTags.map(lambda x: json.dumps(x)).saveAsTextFile(dirPathInput+"user_simsOrd(weightSim="+str(weightSim)+")/")
        else:
            print("\nLa somiglianza tra Users e già presente")
            user_simsTags=spEnv.getSc().textFile(dirPathInput+"user_simsOrd(weightSim="+str(weightSim)+")/").map(lambda x: json.loads(x))

        # for user,user_valPairs in user_simsTags.take(1):
        #     print("\nUser : {}".format(user))
        #     print("User - PairVal :{}".format(list(user_valPairs)))

        """
        Calcolo delle somiglianze tra users in base alle CommunitiesFriends e creazione del corrispondente RDD
        """
        if not os.path.exists(dirPathCommunities+"/"+self.communityType+"/user_simsOrd/"):
            nNeigh=self.nNeigh
            # Recupero RDD con l'elenco delle communities associate ognuna ad una lista di utenti di cui ne fanno parte
            comm_listUsers=spEnv.getSc().textFile(dirPathCommunities+"/"+self.communityType+"/communitiesFriends.json").map(lambda x: json.loads(x))
            # Calcolo del valore di somiglianza tra tutti gli utenti appartenenti alla stessa communities ritorno i primi "N" Amici più simili: (user,[(user,valSim),...])
            user_simsFriends=self.computeSimilarityFriends(spEnv,comm_listUsers).map(lambda p: SocialBased.nearestFriendsNeighbors(p[0],p[1],nNeigh)).filter(lambda p: p!=None)
            user_simsFriends.map(lambda x: json.dumps(x)).saveAsTextFile(dirPathCommunities+"/"+self.communityType+"/user_simsOrd/")
        else:
            print("\nLe somiglianze tra friends appartenenti alle stesse communities (trovate dall'algoritmo {}) sono già presenti!".format(self.communityType))
            user_simsFriends=spEnv.getSc().textFile(dirPathCommunities+"/"+self.communityType+"/user_simsOrd/").map(lambda x: json.loads(x))

        # for user,user_valPairs in user_simsFriends.take(1):
        #     print("\nUser : {}".format(user))
        #     print("User - PairVal :{}".format(list(user_valPairs)))

        """
        Per ogni utente creazione dei files che modellano le varie Communities come un vettore di Varianze associate ognuna ad una determinata categoria/tag
        """
        for filename in os.listdir(dirPathCommunities+"/"+communityType):
            users=spEnv.getSc().textFile(filename).values().flatMap(lambda listUser: listUser).collect()
            print("\n\nUSERS: {}")


        """
        Creazione RDD delle somiglianze che tiene conto dei due RDD (Senza filtro Neighbors su TAGS)
        """
        # Modifica dell'RDD relativo ai TAGS
        user_simsTags=user_simsTags.mapValues(lambda x: TagSocialImpersonalBased.RemoveTags(x))
        # print("\nSemplificato RDD_TAGS Somiglianze!")
        nNeigh=self.nNeigh
        user_simsTot=user_simsTags.union(user_simsFriends).reduceByKey(lambda listPair1,listPair2: TagSocialPersonalBased.joinPairs(listPair1,listPair2)).map(lambda p: TagSocialImpersonalBased.filterSimilarities(p[0],p[1])).filter(lambda p: p!=None).map(lambda p: TagSocialPersonalBased.nearestTagsNeighbors(p[0],p[1],nNeigh)).cache()
        print("\nHo finito di calcolare valori di Somiglianze Globali tra utenti!")
        print("\nUSER SIM_GLOB: {}".format(user_simsTot.take(10)))

        """
        Calcolo delle raccomandazioni personalizzate per i diversi utenti
        """
        user_item_hist=user_item_pair.collectAsMap()
        userHistoryRates=spEnv.getSc().broadcast(user_item_hist)
        # Calcolo per ogni utente la lista di TUTTI gli items suggeriti ordinati secondo predizione. Ritorno un pairRDD del tipo (user,[(scorePred,item),(scorePred,item),...])
        user_item_recs = user_simsTot.map(lambda p: TagSocialPersonalBased.recommendationsUserBasedSocial(p[0],p[1],userHistoryRates.value,dictUser_meanRatesRatings.value)).map(lambda p: TagSocialPersonalBased.convertFloat_Int(p[0],p[1])).collectAsMap()
        # Immagazzino la lista dei suggerimenti finali prodotti per sottoporla poi a valutazione
        self.setDictRec(user_item_recs)
        # print("\nLista suggerimenti: {}".format(self.dictRec))

    @staticmethod
    def recommendationsUserBasedSocial(user_id,users_with_sim,userHistoryRates,user_meanRates):
        """
        Per ogni utente ritorno una lista (personalizzata) di items sugeriti in ordine di rate.
        N.B: Per alcuni user non sarà possibile raccomandare alcun item -> Lista vuota
        Versione che tiene conto solamente dei valori di somiglianza derivanti dagli amici dell'active user
        :param user_id: utente per il quale rilasciare la lista di raccomandazioni
        :param users_with_sim: Lista di elementi del tipo [(user,valSim),(user,ValSim),...]
        :param userHistoryRates: Dizionario del tipo user:[(item,score),(item,score),...]
        :param user_meanRates: Dizionario che per ogni utente contiene valore Medio Rating
        :return:
        """
        def mapCommTag():
            pass

        def updateSimUsers(dictCommTag):
            pass

        # Vado ad associare ad ogni community dell'utente utente il corrispondete tag/categoria di business che più rappresenta l'interesse di quegli amici
        dictCommTag=mapCommTag()

        # Modifico la lista dei vicini dell'utente inserendo in cima gli amici della community che possiedono un interesse in comune
        updateSimUsers(dictCommTag)

        # Dal momento che ogni item potrà essere il vicino di più di un item votato dall'utente dovrò aggiornare di volta in volta i valori
        totals = defaultdict(int)
        sim_sums = defaultdict(int)
        # Ciclo su tutti i vicini dell'utente
        for (vicino,sim) in users_with_sim:
            # Recupero tutti i rates dati dal tale vicino dell'utente
            listRatings = userHistoryRates.get(vicino)
            if listRatings:
                # Ciclo su tutti i Rates
                for (item,rate) in listRatings:
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


    @staticmethod
    def RemoveTags(listPairs):
        return [(user,pair[0]) for user,pair in listPairs if pair[0]>0.0]

    @staticmethod
    def joinPairs(listPair1,listPair2):
        """
        Unisco l'RDD dei tags a quello delle amicizie in base allo stesso user andando a sommare
        i valori di somiglianza nel momento in cui il vicino è lo stesso in entrambi gli RDD, in
        caso contrario mantengo solamente i vicini rispetto ai tags
        :param listPair1: lista di coppie (user,valSim) per RDD dei tags
        :param listPair2: lista di coppie (user,valSim) per RDD dei friends
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

    def createFriendsCommunities(self):
        if not os.path.exists(dirPathCommunities+self.communityType):
            print("\nCreazione del grafo delle amicizie per i vari utenti con algoritmo scelto!")
            # Creazione del dizionario delle amicizie
            self.createDizFriendships()
            # Ciclo su tutti gli utenti
            count=0
            listInfoUsersClusters=[]
            for user in self.friendships.keys():
                # Creazione Grafo delle amicizie
                self.createGraphFriendships(user)
                if len(self.g.es)>0 and len(self.g.vs)>0:
                    count+=1
                    # Calcolo le communities delle amicizie
                    numClusters,sizeClusters=self.createCommunitiesFriendships(user)
                    # Per ogni utente memorizzo il numero di Clusters di amici rilevati e la grandezza media riscontrata
                    listInfoUsersClusters.append((numClusters,mean(sizeClusters)))
            self.setNumUtentiWithClusters(count)
            self.setNumMedioClusters(mean(list(zip(*listInfoUsersClusters))[0]))
            self.setSizeMedioClusters(mean(list(zip(*listInfoUsersClusters))[1]))
            print("\n\nzNumero di utenti per i quali è presente almeno 1 community: {}".format(self.numUtentiWithClusters))
            print("Numero medio di clusters presenti: {}".format(self.numMedioClusters))
            print("Grandezza media dei clusters presenti: {}".format(self.sizeMedioClusters))

        else:
            print("\nLe communities di amici per i vari utenti sono già presenti!")


    def createGraphFriendships(self,user):
        def saveGraphs(g):
            if not os.path.exists(userFriendsGraph):
                os.makedirs(userFriendsGraph)
            g.write_graphml(f=open(userFriendsGraph+"/"+str(user).strip("_")+".graphml","wb"))

        g=Graph()
        """ Creo i nodi (utenti) del grafo """
        friends=list(zip(*self.friendships[user]))[0]

        """ Creo gli archi (amicizie) del grafo SINGOLE """
        print("\n\nUser: {}".format(user))
        # print("\nself.friendships[user] LUNGH: {}: {}".format(len(self.friendships[user]),self.friendships[user]))
        archi=set()
        nodi=set()
        # Ciclo su tutti gli amici dell'utente
        for friend in friends:
            # Ciclo sugli amici degli amici dell'utente
            for friend_friend,valSim_friendFriend in self.friendships[friend]:
                # Controllo che l'amico dell'amico si anche un amico dell'utente, non sia l'utente in questione e non sia già presente l'arco
                if friend_friend in friends and (friend_friend,friend,valSim_friendFriend) not in archi and friend_friend!=user:
                    # Aggiungo l'arco
                    archi.add((friend,friend_friend,valSim_friendFriend))
                    # Aggiungo i due nodi (amici) se non ancora presenti
                    if friend_friend not in nodi:
                        nodi.add(friend_friend)
                        g.add_vertex(name=friend_friend,gender="user",label=friend_friend)
                    if friend not in nodi:
                        nodi.add(friend)
                        g.add_vertex(name=friend,gender="user",label=friend)

        print("Num Archi: {} ".format(len(archi)))
        print("Num Nodi: {} ".format(len(nodi)))

        for friend,friend_friend,weight in archi:
            g.add_edge(friend,friend_friend,weight=weight)

        saveGraphs(g)
        self.g=g
        # print("\nGRAFO: {}".format(self.g))
        # print("\nGrafo delle amicizie creato per user:".format(user))

    def createCommunitiesFriendships(self,utente):
        startTime=time.time()
        g=self.g
        # calculate dendrogram
        dendrogram=None
        clusters=None
        if self.communityType=="all":
            types=communitiesTypes
        else:
            types=[self.communityType]

        for type in types:
            if type=="fastgreedy":
                dendrogram=g.community_fastgreedy(weights="weight")
            elif type=="walktrap":
                dendrogram=g.community_walktrap(weights="weight")
            elif type=="label_propagation":
                clusters=g.community_label_propagation(weights="weight")
            elif type=="multilevel":
                clusters=g.community_multilevel(weights="weight",return_levels=False)
            elif type=="infomap":
                clusters=g.community_infomap(edge_weights="weight")

            # convert it into a flat clustering (VertexClustering)
            if type!="label_propagation" and type!="multilevel" and type!="infomap":
                clusters = dendrogram.as_clustering()

            # Aggiungo grandezza di ogni singolo clusters
            sizeClusters=[]
            for cluster in clusters:
                sizeClusters.append(len(cluster))

            # get the membership vector
            membership = clusters.membership
            communitiesFriends=defaultdict(list)
            for user,community in [(name,membership) for name, membership in zip(g.vs["name"], membership)]:
                communitiesFriends[community].append(user)
            saveJsonData(communitiesFriends.items(),dirPathCommunities+"/"+type,dirPathCommunities+"/"+type+"/"+str(user).strip("_")+".json")
            print("Clustering Summary for '{}' : \n{}".format(type,clusters.summary()))
            return len(clusters),sizeClusters


    def setNumMedioClusters(self, numMedioClusters):
        self.numMedioClusters=numMedioClusters

    def setSizeMedioClusters(self, sizeMedioClusters):
        self.sizeMedioClusters=sizeMedioClusters

    def setNumUtentiWithClusters(self, numUtentiWithClusters):
        self.numUtentiWithClusters=numUtentiWithClusters

    @staticmethod
    def mappingUser(userList,dictUser_TagsScores):
        """
        Ogni user che appartiene alla lista lo sostiuisco con la lista [(tag1,score),(tag2,score),...] a lui associata
        :param userList: lista di utenti
        :param dictUser_TagsScores: dizionario che per ogni user associa la lista [(tag1,score),(tag2,score),...]
        :return:
        """
        lista=[]
        for user in userList:
            lista.extend(dictUser_TagsScores[user])
        return lista

    @staticmethod
    def joinTags(listTagScore):
        """
        Data la lista di pair del tipo (tag,score) dove il tag si può ripetere, restituisco una lista in cui il tag non si ripete e i valori sono stati messi insieme
        :param listTagScore: lista del tipo [(tag1,score),(tag2,score),(tag1,score),...]
        :return: lista del tipo [(tag1,[score1,score2,..]),(tag2,[score1,score2,..]),...]
        """
        dictTags=defaultdict(list)
        for tag,score in listTagScore:
            dictTags[tag].append(score)
        return list(dictTags.items())

    @staticmethod
    def computeVariances(comm,tag_listScores,dictCommNpers):
        # print("\nNumero persone community: {}".format(dictCommNpers[comm]))
        # print("\nNumero persone Tags: {}".format([len(listScores) for _,listScores in tag_listScores]))
        for tag,listScores in tag_listScores:
            val=(mean(np.power(listScores,2))-math.pow(mean(listScores),2))*(dictCommNpers[comm]/len(listScores))
            if val>0.0:
                lista.append((tag,val))
        return (comm,lista)

if __name__ == '__main__':
    spEnv=SparkEnvLocal()
    # Ciclo su tutti i files dei vari utenti che contengono le communities di amici
    for filename in os.listdir(dirPathCommunities+"/"+communityType):
        if os.path.exists(dirPathCommunities+"/"+communityType+"/"+filename):
            print("\n\nFILE: {}".format(filename))
            # Creazione del dizionario che associa per ogni community il numero di persone che ne fanno parte
            dictCommNpers={json.loads(line)[0]: len(json.loads(line)[1]) for line in open(dirPathCommunities+"/"+communityType+"/"+filename)}

            # Colleziono i vari utenti/amici che appartengono alle diverse communities
            users=spEnv.getSc().textFile(dirPathCommunities+"/"+communityType+"/"+filename).map(lambda x: json.loads(x)).values().flatMap(lambda x: x).collect()
            # print("\n\nUsers: {}".format(users))
            # Creo il relativo dizionario che per ogni utente associa una lista di coppie (tag,score)
            user_TagsScores=spEnv.getSc().textFile(userTagJSON).map(lambda line: TagBased.parseFileUser(line)).groupByKey().filter(lambda userListPairs: userListPairs[0] in users).collectAsMap()
            dictUser_TagsScores=spEnv.getSc().broadcast(user_TagsScores)
            # print("DICT: {}".format(user_TagsScores))

            comm_TagsListScores=spEnv.getSc().textFile(dirPathCommunities+"/"+communityType+"/"+filename).map(lambda x: json.loads(x)).mapValues(lambda userList: TagSocialPersonalBased.mappingUser(userList,dictUser_TagsScores.value)).mapValues(lambda listTagScore: TagSocialPersonalBased.joinTags(listTagScore))
            for comm,lista in comm_TagsListScores.collect():
                print("\nCOMM:{}".format(comm))
                print("LISTA: {}".format(lista))

            comm_TagsVars=comm_TagsListScores.map(lambda x: TagSocialPersonalBased.computeVariances(x[0],x[1],dictCommNpers)).collect()
            for comm,lista in comm_TagsVars:
                print("\nCOMM:{}".format(comm))
                print("LISTA: {}".format(lista))
                minimo=list(zip(*lista))[1]
                print("MINIMO: "+str(minimo))

