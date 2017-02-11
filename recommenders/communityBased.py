import shutil

__author__ = 'maury'

from itertools import combinations
from igraph import *
import json
import time

from tools.sparkEnvLocal import SparkEnvLocal
from recommenders.itemBased import ItemBased
from recommenders.tagBased import TagBased
from recommenders.recommender import Recommender
from conf.confDirFiles import userFriendsGraph
from conf.confCommunitiesFriends import *
from tools.tools import saveJsonData

class CommunityBased(ItemBased):
    def __init__(self,name,friendships,communityType):
        ItemBased.__init__(self,name=name)
        # Settaggio del dizionazio delle amicizie
        self.friendships=friendships
        self.communityType=communityType

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
        user_item_pair=spEnv.getSc().textFile(directory+"/*").map(lambda line: Recommender.parseFileUser(line)).groupByKey()
        user_meanRatesRatings=user_item_pair.map(lambda p: CommunityBased.computeMean(p[0],p[1])).collectAsMap()
        dictUser_meanRatesRatings=spEnv.getSc().broadcast(user_meanRatesRatings)

        """
        Calcolo delle somiglianze tra users in base alle CommunitiesFriends e creazione del corrispondente RDD
        """
        if self.getCurrentFold()==0 and not os.path.exists(dirPathCommunities+"/"+self.communityType+"/user_simsOrd/"):
            nNeigh=self.nNeigh
            # Recupero RDD con l'elenco delle communities associate ognuna ad una lista di utenti di cui ne fanno parte
            comm_listUsers=spEnv.getSc().textFile(dirPathCommunities+"/"+self.communityType+"/communitiesFriends.json").map(lambda x: json.loads(x))
            # Calcolo del valore di somiglianza tra tutti gli utenti appartenenti alla stessa communities ritorno i primi "N" Amici più simili: (user,[(user,valSim),...])
            user_simsOrd=self.computeSimilarityFriends(spEnv,comm_listUsers).map(lambda p: CommunityBased.filterSimilarities(p[0],p[1])).filter(lambda p: p!=None).map(lambda p: CommunityBased.nearestFriendsNeighbors(p[0],p[1],nNeigh))
            # Calcolo del numero di amici presenti nelle varie liste di k vicini
            friendships=self.friendships
            # print("\nSomma delle percentuali di amici presenti nelle liste dei k Vicini: {}".format(user_simsOrd.map(lambda x: CommunityBased.computePercFriends(x[0],x[1],friendships)).sum()))
            print("\nMedia delle percentiali di amici presenti nelle liste dei k Vicini: {}".format(user_simsOrd.map(lambda x: CommunityBased.computePercFriends(x[0],x[1],friendships)).mean()))
            user_simsOrd.map(lambda x: json.dumps(x)).saveAsTextFile(dirPathCommunities+"/"+self.communityType+"/user_simsOrd/")

        user_simsOrd=spEnv.getSc().textFile(dirPathCommunities+"/"+self.communityType+"/user_simsOrd/").map(lambda x: json.loads(x))

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
        user_item_recs = user_simsOrd.map(lambda p: CommunityBased.recommendationsUserBasedSocial(p[0],p[1],userHistoryRates.value,dictUser_meanRatesRatings.value)).map(lambda p: TagBased.convertFloat_Int(p[0],p[1])).collectAsMap()
        # Immagazzino la lista dei suggerimenti finali prodotti per sottoporla poi a valutazione
        self.setDictRec(user_item_recs)
        print("\nLista di raccomandazioni calcolata!")
        # print("\nLista suggerimenti: {}".format(self.dictRec))

    def computeSimilarityFriends(self,spEnv,rdd):
        """
        Calcolo il valore di somiglianza tra tutte le coppie di users che appartengono alla stessa community e salvo i valori su files
        :param spEnv: SparkContext di riferimento
        :param rdd: RDD con elementi che associano ad ogni community una lista degli utenti appartenenti
        :return: RDD che associa ad ogni coppia di utenti il corrispondente valore di somiglianza
        """
        if not os.path.exists(dirPathCommunities+self.communityType+"/SimilaritiesFiles"):
            print("\nVado a calcolare le somiglianze tra friends che appartengono alle stesse community trovate dall'algortimo {}!".format(self.communityType))
            os.makedirs(dirPathCommunities+self.communityType+"/SimilaritiesFiles")
            friendships=spEnv.getSc().broadcast(self.friendships)
            communityType=self.communityType
            rdd.foreach(lambda x: CommunityBased.computeCommunitiesSimilarity(x[0],x[1],friendships.value,communityType))

        """ Recupero i valori appena calcolati per costruire l'RDD finale """
        user_sims=spEnv.getSc().textFile(dirPathCommunities+self.communityType+"/SimilaritiesFiles/*").map(lambda x: json.loads(x)).groupByKey()
        return user_sims

    @staticmethod
    def computeCommunitiesSimilarity(comm,listUsers,friendships,communityType):
        with open(dirPathCommunities+communityType+"/SimilaritiesFiles/communitiesFriendsSim_"+str(comm)+".json","w") as f:
            """ Ciclo su tutte le possibili combinazioni di users che appartengono alla stessa community e ne calcolo la Jaccard similarity """
            for user1,user2 in combinations(listUsers,2):
                # Recupero la lista di amici del primo utente
                amici_user1=list(zip(*friendships[user1]))[0]
                # Recupero la lista di amici del secondo utente
                amici_user2=list(zip(*friendships[user2]))[0]
                num=len(set(amici_user1).intersection(set(amici_user2)))+1
                den=len(set(amici_user1).union(set(amici_user2)))
                val=num/den
                linea=((user1,(user2,val)))
                f.write(json.dumps(linea)+"\n")
                linea=((user2,(user1,val)))
                f.write(json.dumps(linea)+"\n")


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

    def createFriendsCommunities(self):
        if not os.path.exists(userFriendsGraph):
            print("\nCreazione del grafo delle amicizie!")
            # Creazione Grafo delle amicizie
            self.createGraph()
        else:
            print("\nIl grafo delle amicizie è già presente!")

        if self.communityType!="all":
            if not os.path.exists(dirPathCommunities+"/"+self.communityType+"/communitiesFriends.json"):
                # Calcolo le communities delle amicizie
                self.createCommunities()
            else:
                print("\nIl file delle communities dell'algoritmo scelto è già presente!")
        else:
            print("\nHo deciso di trovare le communities per tutti i vari algoritmi!")
            self.createCommunities()


    def createGraph(self):
        def saveGraphs(g):
            g.write_pickle(fname=open(userFriendsGraph,"wb"))
            g.write_graphml(f=open(userFriendsGraph+".graphml","wb"))

        g=Graph()
        """ Creo i nodi (utenti) del grafo """
        for user in self.friendships.keys():
            g.add_vertex(name=user,gender="user",label=user)

        """ Creo gli archi (amicizie) (inserisco arco una sola volta)"""
        listaArchi=[(user,friend,weight) for user,listPairs in self.friendships.items() for friend,weight in listPairs]
        archi=[]
        for user,friend,weight in listaArchi:
            if (friend,user,weight) not in archi:
                archi.append((user,friend,weight))
                g.add_edge(user,friend,weight=weight)

        saveGraphs(g)
        print("\nSummary:\n{}".format(summary(g)))
        print("\nGrafo delle amicizie creato e salvato!")

    def createCommunities(self):
        startTime=time.time()
        g=Graph.Read_Pickle(fname=open(userFriendsGraph,"rb"))
        # calculate dendrogram
        dendrogram=None
        clusters=None
        if self.communityType=="all":
            types=communitiesTypes
        else:
            types=[self.communityType]

        for type in types:
            if not os.path.exists(dirPathCommunities+"/"+type+"/communitiesFriends.json"):
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
                # get the membership vector
                membership = clusters.membership
                communitiesFriends=defaultdict(list)
                for user,community in [(name,membership) for name, membership in zip(g.vs["name"], membership)]:
                    communitiesFriends[community].append(user)
                saveJsonData(communitiesFriends.items(),dirPathCommunities+"/"+type,dirPathCommunities+"/"+type+"/communitiesFriends.json")
                print("\nClustering Summary for '{}' : \n{}".format(type,clusters.summary()))

        print("\nFinito di calcolare le communities!")

    @staticmethod
    def filterSimilarities(user_id,users_and_sims):
        """
        Rimuovo tutti quei vicini per i quali il valore di somiglianza è < del valore stabilito
        :param user_id: active user preso in considerazione
        :param users_and_sims: users e associati valori di somiglianze per l'active user
        :return: Ritorno un nuovo pairRDD filtrato
        """
        lista=[item for item in users_and_sims if item[1]>=weightFriendsSim]
        if len(lista)>0:
            return user_id,lista

    def setFriendships(self,friendships):
        self.friendships=friendships

    def getFriendships(self):
        return self.friendships

    def getCommunityType(self):
        return self.communityType

    def setCommunityType(self,type):
        self.communityType=type

    @staticmethod
    def computePercFriends(user_id, users_and_sims, friendships):
        """
        Ritorna per ogni utente il numero di utenti presenti
        :param user_id: Active user
        :param users_and_sims: Lista di pairs (utente,valSim)
        :param friendships: Dizionario delle amicizie per i vari utenti
        :return: Elemento che contiene il numero di amici che ogni volta fanno parte della lista dei k vicini finale
        """
        if len(users_and_sims)>0:
            return (len([user for user in list(zip(*users_and_sims))[0] if user in friendships[user_id]])/len(users_and_sims))

