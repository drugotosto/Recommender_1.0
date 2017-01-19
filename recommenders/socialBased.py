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

class SocialBased(ItemBased):
    def __init__(self,name,friendships,communityType):
        ItemBased.__init__(self,name=name)
        # Settaggio del dizionazio delle amicizie (inzialmente non pesato)
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
        user_meanRatesRatings=user_item_pair.map(lambda p: SocialBased.computeMean(p[0],p[1])).collectAsMap()
        dictUser_meanRatesRatings=spEnv.getSc().broadcast(user_meanRatesRatings)

        """
        Calcolo delle somiglianze tra users in base alle CommunitiesFriends e creazione del corrispondente RDD
        """
        if not os.path.exists(dirPathCommunities+"/"+self.communityType+"/user_simsOrd/"):
            nNeigh=self.nNeigh
            # Recupero RDD con l'elenco delle communities associate ognuna ad una lista di utenti di cui ne fanno parte
            comm_listUsers=spEnv.getSc().textFile(dirPathCommunities+"/"+self.communityType+"/communitiesFriends.json").map(lambda x: json.loads(x))
            # Calcolo del valore di somiglianza tra tutti gli utenti appartenenti alla stessa communities ritorno i primi "N" Amici più simili: (user,[(user,valSim),...])
            user_simsOrd=self.computeSimilarityFriends(spEnv,comm_listUsers).map(lambda p: SocialBased.nearestFriendsNeighbors(p[0],p[1],nNeigh)).filter(lambda p: p!=None)
            user_simsOrd.map(lambda x: json.dumps(x)).saveAsTextFile(dirPathCommunities+"/"+self.communityType+"/user_simsOrd/")
        else:
            print("\nLe somiglianze tra friends appartenenti alle stesse communities (trovate dall'algoritmo {}) sono già presenti!".format(self.communityType))
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
        user_item_recs = user_simsOrd.map(lambda p: SocialBased.recommendationsUserBasedSocial(p[0],p[1],userHistoryRates.value,dictUser_meanRatesRatings.value)).map(lambda p: TagBased.convertFloat_Int(p[0],p[1])).collectAsMap()
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
            rdd.foreach(lambda x: SocialBased.computeCommunitiesSimilarity(x[0],x[1],friendships.value,communityType))

        """ Recupero i valori appena calcolati per costruire l'RDD finale """
        user_sims=spEnv.getSc().textFile(dirPathCommunities+self.communityType+"/SimilaritiesFiles/*").map(lambda x: json.loads(x)).groupByKey()
        return user_sims

    @staticmethod
    def computeCommunitiesSimilarity(comm,listUsers,friendships,communityType):
        with open(dirPathCommunities+communityType+"/SimilaritiesFiles/communitiesFriendsSim_"+str(comm)+".json","w") as f:
            """ Ciclo su tutte le possibili combinazioni di users che appartengono alla stessa community """
            for user1,user2 in combinations(listUsers,2):
                amici_user1=list(zip(*friendships[user1]))[0]
                amici_user2=list(zip(*friendships[user2]))[0]
                num=len(set(amici_user1+tuple(user1)).intersection(set(amici_user2+tuple(user2))))+1
                den=len(set(amici_user1+tuple(user1)).union(set(amici_user2+tuple(user2))))+1
                linea=((user1,(user2,num/den)))
                f.write(json.dumps(linea)+"\n")
                linea=((user2,(user1,num/den)))
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

    def createFriendsCommunities(self):
        if not os.path.exists(dirPathCommunities+self.communityType):
            print("\nCreazione del grafo delle amicizie per i vari utenti con algoritmo scelto!")
            # Creazione del dizionario delle amicizie
            self.createDizFriendships()
            # Ciclo su tutti gli utenti
            for user in self.friendships.keys():
                # Creazione Grafo delle amicizie
                self.createGraph(user)
                if len(self.g.es)>0 and len(self.g.vs)>0:
                    # Calcolo le communities delle amicizie
                    self.createCommunities(user)
            print("\nFinito di calcolare le communities!")
        else:
            print("\nLe communities di amici per i vari utenti sono già presenti!")

    def createDizFriendships(self):
        def createPairs(user,listFriends):
            return [(user,friend) for friend in listFriends]

        """ Creo gli archi del grafo mancanti """
        listaList=[createPairs(user,listFriends) for user,listFriends in self.friendships.items()]
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

        """ Costruisco il dizionario con gli archi pesati (dato dal numero di amicizie in comune tra utenti) """
        def createListFriendsDoubleWeight(user,dizFriendshipsDouble):
            return [(friend,len(set(dizFriendshipsDouble[user])&set(dizFriendshipsDouble[friend]))+1) for friend in dizFriendshipsDouble[user]]

        friendships={user:createListFriendsDoubleWeight(user,dizFriendshipsDouble) for user in dizFriendshipsDouble}

        # """ Per ogni arco (user1-user2) vado ad eliminare la controparte (user2-user1) """
        # archi=set()
        # for user,listFriends in dizFriendshipsDoubleWeight.items():
        #     for elem in listFriends:
        #         if (user,elem[0],elem[1]) not in archi and (elem[0],user,elem[1]) not in archi:
        #             archi.add((user,elem[0],elem[1]))
        #
        # """ Costruisco il dizionario finale da salvare """
        # friendships=defaultdict(list)
        # for k,v,r in archi:
        #     friendships[k].append((v,r))

        print("\nNumero di AMICIZIE (doppie) presenti sono: {}".format(sum([len(lista) for lista in friendships.values()])))
        # numUtenti=len(set([user for user in friendships]).union(set([user for lista in friendships.values() for user,_ in lista])))
        numUtenti=len(list(friendships.keys()))
        print("\nNumero di UTENTI che sono presenti in communities: {} (alcuni non avevano amicizie...)".format(numUtenti))
        self.setFriendships(friendships)
        print("\nDizionario delle amicizie pesato creato e settato!")

    def createGraph(self,user):
        g=Graph()
        """ Creo i nodi (utenti) del grafo """
        friends=list(zip(*self.friendships[user]))[0]
        # for user in friends:
        #     g.add_vertex(name=user,gender="user",label=user)

        """ Creo gli archi (amicizie) del grafo SINGOLE """
        # print("\nUser: {}".format(user))
        # print("\nself.friendships[user]: {}".format(self.friendships[user]))
        amici=[(friend,friend_friend,valSim_friendFriend) for friend,_ in self.friendships[user] for friend_friend,valSim_friendFriend in self.friendships[friend] if friend_friend in friends]
        archi=[]
        for friend,friend_friend,weight in amici:
            if (friend_friend,friend,weight) not in archi:
                g.add_vertex(name=friend,gender="user",label=friend)
                g.add_vertex(name=friend_friend,gender="user",label=friend_friend)
                archi.append((friend,friend_friend,weight))
                g.add_edge(friend,friend_friend,weight=weight)

        self.g=g
        print("\nGRAFO: {}".format(self.g))
        print("\nGrafo delle amicizie creato per user:".format(user))

    def createCommunities(self,user):
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

            # get the membership vector
            membership = clusters.membership
            communitiesFriends=defaultdict(list)
            for user,community in [(name,membership) for name, membership in zip(g.vs["name"], membership)]:
                communitiesFriends[community].append(user)
            saveJsonData(communitiesFriends.items(),dirPathCommunities+"/"+type,dirPathCommunities+"/"+type+"/communitiesFriends_"+str(user)+".json")
            print("\nClustering Summary for '{}' : \n{}".format(type,clusters.summary()))

    def setFriendships(self,friendships):
        self.friendships=friendships

    def getFriendships(self):
        return self.friendships

    def getCommunityType(self):
        return self.communityType

    def setCommunityType(self,type):
        self.communityType=type
