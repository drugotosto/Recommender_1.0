__author__ = 'maury'

import os
import time
import numpy as np
import pandas as pd
from numpy.random import choice

from conf.confRS import nFolds, typeRecommender
from conf.confDirFiles import  datasetJSON, dirPathOutput, dirPathInput, dirTrain, dirTest, dirFolds
from conf.confItemBased import typeSimilarity, weightSim, nNeigh
from recommenders.itemBased import ItemBased
from tools.tools import saveJsonData, printRecVal
from tools.sparkEnvLocal import SparkEnvLocal
from tools.dataSetAnalyzer import DataScienceAnalyzer
from recommenders.communityBased import CommunityBased
from recommenders.tagBased import TagBased
from conf.confCommunitiesFriends import communityType, weightFriendsSim, communitiesTypes
from recommenders.communityTagBased import CommunityTagBased
from recommenders.friendsBased import FriendsBased
from recommenders.friendsRateBased import FriendsRateBased

if __name__ == '__main__':
    startTime=time.time()
    # Creazione dello SparkContext
    spEnv=SparkEnvLocal()
    # Instanzio Analizzatore dei Dati
    analyzer=DataScienceAnalyzer()

    if not os.path.exists(datasetJSON):
        os.makedirs(dirPathInput)
        """ Recupero dati da files (json) per la creazione del Dataframe finale e salvataggio su file 'dataSet.json' """
        analyzer.createDataSet()
        # Creazione File CSV
        analyzer.getDataFrame().to_csv(dirPathInput+"dataset.csv")
        # Creazione File dei ratings (filtrati) del Dataset
        saveJsonData(analyzer.getDataFrame()[["user_id","business_id","stars"]].values.tolist(),dirPathInput,datasetJSON)
        # Salvo il DataSet finale
        analyzer.getDataFrame().to_pickle(dirPathInput+"dataset")
    else:
        print("\n******** Il DataSet era già presente! *********")
        analyzer.setProperties(pd.read_pickle(dirPathInput+"dataset"))
        analyzer.printValuesDataset()
        numFriends=1
        print("Percentuale di utenti con almeno {} amici è: {}".format(numFriends,analyzer.getNumUsersWithFriends(numFriends)/analyzer.getNumUsers()))
        # print("Distribuzione degli Utenti con almeno {} amici è : {}".format(numFriends,analyzer.getDistrFriends(numFriends)))

    rs=None
    # Instanzio il tipo di Recommender scelto
    if typeRecommender=="ItemBased":
        rs=ItemBased(name=typeRecommender)
    elif typeRecommender=="TagBased":
        rs=TagBased(name=typeRecommender)
    elif "Community" in typeRecommender:
        friendships=analyzer.createDizFriendships()
        if typeRecommender=="CommunityBased":
            rs=CommunityBased(name=typeRecommender,friendships=friendships,communityType=communityType)
        elif typeRecommender=="CommunityTagBased":
            rs=CommunityTagBased(name=typeRecommender,friendships=friendships,communityType=communityType)
        # Calcolo le communities delle amicizie
        rs.createFriendsCommunities()
    elif "Friends" in typeRecommender:
        friendships=analyzer.createDizFriendships()
        if typeRecommender=="FriendsBased":
            rs=FriendsBased(name=typeRecommender,friendships=friendships)
        elif typeRecommender=="FriendsRateBased":
            dictBus_Tags=analyzer.getBusTags()
            rs=FriendsRateBased(name=typeRecommender,friendships=friendships,dictBus_Tags=dictBus_Tags)


    if not os.path.exists(dirFolds):
        """ Creazione dei files (trainSetFold_k/testSetFold_k) per ogni prova di valutazione"""
        # SUDDIVISIONE DEGLI UTENTI IN K_FOLD
        usersFolds=np.array_split(choice(analyzer.getDataFrame()["user_id"].unique(),analyzer.getDataFrame()["user_id"].unique().shape[0],replace=False),nFolds)
        rs.createFolds(spEnv,list(map(list,usersFolds)))
        print("\nCreazione files  all'interno delle cartelle 'Folds/train_k' -  'Folds/test_k' terminata!")
    else:
        print("\nLe cartelle 'Folds/train_k' erano gia presenti!")

    # fold=0
    # # Ciclo su tutti i folds files (train/test)
    # while fold<nFolds:
    #     rs.setCurrentFold(fold)
    #     """ Costruzione del modello a seconda dell'approccio utilizzato """
    #     rs.builtModel(spEnv,dirTrain+str(fold))
    #     print("\nModello costruito!")
    #     """
    #     Ho calcolato tutte le TOP-N raccomandazioni personalizzate per i vari utenti con tanto di predizione per ciascun item
    #     Eseguo la valutazione del recommender utilizzando le diverse metriche (MAE,RMSE,Precision,Recall,...)
    #     """
    #     rs.retrieveTestData(dirTest+str(fold))
    #     rs.getEvaluator().computeEvaluation(rs.getDictRec(),analyzer)
    #     print("\nEseguita Valutazione per Fold {}".format(fold))
    #     fold+=1
    #
    # """
    # Salvataggio su file (json) dei risultati finali di valutazione (medie dei valori sui folds)
    # """
    # fileName=None
    # if rs.getName()=="TagBased" or rs.getName()=="ItemBased":
    #     fileName=dirPathOutput+rs.getName()+"/"+typeSimilarity+"_(weightSim="+str(weightSim)+")"
    # elif rs.getName()=="CommunityBased":
    #     fileName=dirPathOutput+rs.getName()+"/"+communityType+"_(weightFriendsSim="+str(weightFriendsSim)+")"
    # elif rs.getName()=="CommunityTagBased":
    #     fileName=dirPathOutput+rs.getName()+"/"+communityType+"_"+typeSimilarity+"_(weightSim="+str(weightSim)+",weightFriendsSim="+str(weightFriendsSim)+")"
    # elif rs.getName()=="FriendsBased" or rs.getName()=="FriendsRateBased":
    #     fileName=dirPathOutput+rs.getName()+"/jaccard_(weightFriendsSim="+str(weightFriendsSim)+")"
    # printRecVal(evaluator=rs.getEvaluator(),directory=dirPathOutput+rs.getName()+"/",fileName=fileName)
    # print("\nComputazione terminata! Durata totale: {} min.".format((time.time()-startTime)/60))


    """
    CODICE DA UTILIZZARE PER TESTARE I VARI ALGORTIMI DI DETECTION COMMUNITY
    """
    # Ciclo su tutti i folds files (train/test)
    if communityType=="all":
        for type in communitiesTypes:
            fold=0
            rs.evaluator.dataEval={"nTestRates":[],"hits":[],"mae":[],"rmse":[],"precision":[],"recall":[],"f1":[],"covUsers":[],"covMedioBus":[]}
            rs.setCommunityType(type)
            while fold<nFolds:
                """ Costruzione del modello a seconda dell'approccio utilizzato """
                rs.builtModel(spEnv,dirTrain+str(fold))
                print("\nModello costruito!")
                """
                Ho calcolato tutte le TOP-N raccomandazioni personalizzate per i vari utenti con tanto di predizione per ciascun item
                Eseguo la valutazione del recommender utilizzando le diverse metriche (MAE,RMSE,Precision,Recall,...)
                """
                rs.retrieveTestData(dirTest+str(fold))
                rs.getEvaluator().computeEvaluation(rs.getDictRec(),analyzer)
                print("\nEseguita Valutazione per Fold {}".format(fold))
                fold+=1

            """
            Salvataggio su file (json) dei risultati finali di valutazione (medie dei valori sui folds)
            """
            if rs.getName()=="CommunityBased":
                fileName=dirPathOutput+rs.getName()+"/"+type+"_(weightFriendsSim="+str(weightFriendsSim)+")"
            else:
                fileName=dirPathOutput+rs.getName()+"/"+type+"_"+typeSimilarity+"_(weightSim="+str(weightSim)+",weightFriendsSim="+str(weightFriendsSim)+")"
            printRecVal(evaluator=rs.getEvaluator(),directory=dirPathOutput+rs.getName()+"/",fileName=fileName)
            print("\nComputazione terminata! Durata totale: {} min.".format((time.time()-startTime)/60))


