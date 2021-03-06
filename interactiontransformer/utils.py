from openml import datasets, tasks, runs, flows, config
import os, pandas, sklearn, arff, pprint, numpy, seaborn
import numpy as np, pandas as pd, pickle
import copy
from sklearn.model_selection import StratifiedKFold
import dask

MANUSCRIPT_IDs=np.array([4154, 4329, 4136, 4137, 73, 143, 142, 72, 77, 120, 135, 139, 146, 161, 162, 273, 274, 262, 264, 267, 269, 256, 257, 258, 260, 293, 246, 351, 354, 346, 1161, 1162, 1163, 1164, 1165, 1166, 1141, 1144, 1145, 1146, 890, 1147, 1149, 1150, 1181, 1182, 1178, 1039, 1016, 1236, 1372, 1374, 1375, 1376, 1205, 1212, 1216, 1217, 1219, 1237, 1238, 1241, 1151, 1152, 1153, 1154, 1155, 1157, 1158, 1159, 1160, 1561, 1562, 1563, 1564, 1597, 1085, 446, 1042, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1458, 1412, 1441, 1448, 1449, 1450, 1131, 1132, 1133, 1134, 1135, 1137, 1138, 1139, 1502, 1136, 1369, 41946, 41945, 1371, 1180, 40900, 41949, 41967, 41966, 41964, 1218, 1373, 1211, 1148, 40645, 40646, 40647, 40648, 40649, 40650, 40660, 40588, 40589, 40590, 40665, 40666, 40669, 41228, 41672, 41674, 41675, 41679, 41680, 41682, 41684, 41685, 40514, 40515, 40517, 40518, 40922, 40910, 41144, 41145, 41156, 41157, 41158, 40693, 40702, 40704, 40705, 40706, 40710, 40680, 40681, 40683, 40690, 40713, 40714, 40999, 41026, 41005, 41007, 41538, 41521, 40591, 40592, 40593, 40594, 40595, 40596, 40597, 41496, 4134, 1485, 40994, 40983, 40978, 4534, 132, 1220, 1442, 1443, 1444, 1446, 1547, 1467, 1447, 1451, 1452, 1460, 1463, 1566, 1556, 1495, 1506, 1507, 1496, 1498, 1510, 1511, 1524, 1484, 1488, 1473, 1490, 121, 40701, 126, 41973, 376, 41977, 41978, 41976, 1600, 124, 153, 1471, 1462, 23499, 41896, 41897, 41714, 41715, 41716, 41711, 41712, 41713, 41719, 41720, 41721, 41722, 41723, 41717, 41718, 41709, 41710, 41708, 41814, 41815, 41816, 41803, 41804, 41805, 41806, 41807, 41808, 41800, 41801, 41802, 41817, 41818, 41819, 41811, 41812, 41813, 41798, 41799, 41820, 41821, 41809, 41810, 41879, 41880, 41881, 41893, 41895, 41874, 41875, 41876, 41888, 41889, 41890, 41882, 41883, 41884, 41871, 41872, 41873, 41885, 41886, 41887, 41891, 41892, 41877, 41878, 140, 41759, 41760, 41761, 41767, 41768, 41769, 41754, 41755, 41756, 41748, 41749, 41750, 41751, 41752, 41753, 41762, 41763, 41764, 41770, 41771, 41772, 41765, 41766, 41757, 41758, 41868, 41869, 41870, 41849, 41850, 41851, 41846, 41847, 41848, 41862, 41863, 41864, 41865, 41866, 41867, 41857, 41858, 41859, 41852, 41853, 41854, 41860, 41861, 41855, 41856, 41737, 41738, 41739, 41740, 41741, 41742, 41727, 41707, 41728, 41729, 41730, 41731, 41745, 41746, 41747, 41724, 41725, 41726, 41734, 41735, 41736, 41743, 41744, 41732, 41733, 41786, 41787, 41788, 41789, 41790, 41791, 41792, 41793, 41794, 41795, 41796, 41797, 41775, 41776, 41777, 41778, 41779, 41780, 41781, 41782, 41783, 41773, 41774, 41784, 41785, 41838, 41839, 41840, 41833, 41834, 41835, 41825, 41826, 41827, 41841, 41842, 41843, 41822, 41823, 41824, 41828, 41829, 41830, 41844, 41845, 41836, 41837, 41831, 41832, 128, 41998, 122, 152, 41544, 1480, 1122, 41142, 41161, 41150, 41159, 41143, 41146, 1169, 1140, 1142, 1486, 4135, 40981, 23517, 131, 1167, 1242, 41894, 1130, 1143, 1235, 1461, 1377, 1558, 1464, 164, 59, 137, 459, 450, 444, 448, 41898, 40, 53, 43, 312, 316, 336, 337, 334, 742, 743, 744, 745, 752, 753, 748, 749, 740, 741, 746, 747, 750, 751, 754, 755, 756, 758, 759, 761, 770, 771, 766, 767, 764, 765, 768, 769, 772, 773, 762, 763, 788, 789, 784, 785, 787, 775, 776, 782, 783, 779, 780, 777, 778, 938, 941, 953, 951, 946, 947, 945, 949, 950, 942, 943, 955, 956, 958, 954, 959, 962, 964, 965, 983, 978, 979, 980, 969, 974, 976, 977, 970, 971, 973, 1038, 1045, 1022, 1040, 1046, 1048, 1025, 1026, 1020, 1021, 1075, 1104, 1073, 1107, 1069, 1071, 1068, 927, 928, 919, 920, 916, 917, 918, 923, 924, 922, 925, 926, 921, 915, 914, 910, 911, 907, 905, 906, 913, 909, 912, 908, 896, 903, 904, 894, 901, 902, 882, 884, 895, 886, 900, 934, 935, 936, 937, 931, 932, 933, 929, 991, 987, 996, 997, 988, 994, 995, 1019, 1013, 1014, 1015, 1006, 1009, 1011, 1012, 1004, 1005, 1066, 1067, 1050, 1064, 1065, 1054, 1055, 1063, 1056, 1061, 1062, 1049, 1059, 1060, 1120, 1121, 1370, 251, 310, 311, 350, 357, 801, 793, 794, 790, 797, 799, 791, 792, 796, 795, 800, 774, 806, 803, 807, 808, 805, 479, 476, 467, 472, 461, 463, 464, 465, 717, 718, 719, 720, 682, 683, 713, 714, 715, 716, 721, 722, 734, 735, 723, 724, 732, 733, 736, 737, 728, 729, 730, 731, 726, 727, 725, 812, 818, 819, 816, 817, 811, 814, 821, 815, 804, 813, 833, 834, 824, 825, 832, 829, 830, 820, 826, 827, 828, 823, 851, 837, 838, 849, 850, 843, 847, 848, 845, 846, 841, 835, 836, 862, 863, 867, 868, 864, 855, 857, 860, 859, 853, 865, 866, 879, 873, 874, 869, 870, 877, 878, 875, 876, 871, 887, 888, 889, 891, 892, 893, 885, 880, 881, 31, 3, 1489, 1504, 1494, 50, 1116, 1479, 1453, 1156, 44, 151, 37, 1487, 335, 333, 1455])

def download_openml(ID=0,api_key='',tmp='tmp',dataset_path='datasets'):
    config.apikey = api_key
    os.makedirs(tmp,exist_ok=True)
    os.makedirs(dataset_path,exist_ok=True)
    config.set_cache_directory(os.path.abspath('tmp'))
    if not os.path.exists('{}/{}.p'.format(datasets_path,ID)):
        try:
            odata = datasets.get_dataset(int(ID))
            X, y, categorical, attribute_names = odata.get_data(
             target=odata.default_target_attribute)
            y=y.astype(str)
            if not isinstance(y,np.ndarray):
                y=y.values
            X=np.hstack((X,y.reshape(-1,1)))
            df=pd.DataFrame(X, columns=attribute_names+[odata.default_target_attribute])
            pickle.dump(dict(cat=categorical,names=attribute_names+[odata.default_target_attribute],X=df),open('{}/{}.p'.format(dataset_path,ID),'wb'))
            return 1
        except Exception as e:
            return 0
        return 2

def download_multiple_openml(IDs=MANUSCRIPT_IDs,api_key='',tmp='tmp',dataset_path='datasets'):
    dask.compute(*[dask.delayed(download_openml)(ID,api_key,tmp,dataset_path) for ID in IDs], scheduler='processes', num_workers=10)

def get_dataset(dataset='datasets/0.p'):
    d=pickle.load(open(dataset,'rb'))
    return d['X'].iloc[:,:-1],d['X'].iloc[:,-1],np.array(d['cat'])

def preprocess_data(X,y,cat, return_xy=False):
    if not X.isnull().sum().sum():
        X_nocat,X_cat=X.loc[:,~cat],X.loc[:,cat]
        if X_cat.shape[1]:
            ncat=np.array(X_cat.nunique().tolist())
            X_cat=X_cat.loc[:,ncat<100]
            ncat=np.array(X_cat.nunique().tolist())
            X_cat=pd.get_dummies(X_cat,columns=X_cat.columns.values[ncat>2]).apply(lambda x: x==x.unique()[0],axis=0)
            X_cat=X_cat.astype(int)
        X=pd.concat([X_nocat,X_cat],axis=1).astype(float)
    y=(y==y.unique()[0]).astype(float)
    if return_xy:
        yield X,y
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        info_dict=dict(n=X.shape[0],p=X.shape[1],pcat=X_cat.shape[1],cb=max(1.-np.mean(y),np.mean(y)))
        for train, test in cv.split(X,y):
            X_train, X_test, y_train, y_test = X.iloc[train, :], X.iloc[test, :], y.iloc[train], y.iloc[test]
            yield X_train, X_test, y_train, y_test, copy.deepcopy(info_dict)
