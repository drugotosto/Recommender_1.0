__author__ = 'maury'

from conf.confDirFiles import dirPathCommunities

"""
Valori ottimali per weightFriendsSim a seconda del DataSet preso in considerazione (socialBased):
- DataSet_(numRec=4 numTags=1) --> weightFriendsSim=0.1
- DataSet_(numRec=10 numTags=1) --> weightFriendsSims=0.25
- DataSet_(numRec=20 numTags=1) --> weightFriendsSim=0.3
"""

"""
Valori ottimali per weightFriendsSim a seconda del DataSet preso in considerazione (friendsBased):
- DataSet_(numRec=20 numTags=1) --> weightFriendsSim=0.3
"""

weightFriendsSim=0.2

# Elenco di tutte gli algoritmi di detection community possibili
communitiesTypes=["fastgreedy", "walktrap", "label_propagation", "infomap", "multilevel"]
# Scelta di uno tra i vari algortimi di detection communities ("tipo"/"all")
communityType="infomap"

# File delle communities scelto da utilizzare dal quale andare a calcolare le somiglianze tra utenti
fileFriendsCommunities=dirPathCommunities+"/"+communityType+"/communitiesFriends.json"