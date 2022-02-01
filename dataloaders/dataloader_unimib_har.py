import pandas as pd
import numpy as np
import os
from dataloaders.dataloader_base import BASE_DATA
# ========================================       UNIMIB_HAR_DATA               =============================
class UNIMIB_HAR_DATA(BASE_DATA):

    """

    In this article, we present a new smartphone accelerometer dataset designed for activity recognition.
    The dataset includes 11,771 activities performed by 30 subjects of ages ranging from 18 to 60 years.
    Activities are divided in 17 fine grained classes grouped in two coarse grained classes: 9 types of activities of daily living (ADL) and 8 types of falls. 
    The dataset has been stored to include all the information useful to select samples according to different criteria, 
    such as the type of ADL performed, the age, the gender, and so on. Finally, the dataset has been benchmarked with two different classifiers and with different configurations. 
    The best results are achieved with k-NN classifying ADLs only, considering personalization, and with both windows of 51 and 151 samples.

    data is already splited with window size **151**  sampling freq 58.99 , total window size = 11735

    """
    def __init__(self, args):


        """
        root_path : Root directory of the data set
        difference (bool) : Whether to calculate the first order derivative of the original data
        datanorm_type (str) : Methods of data normalization: "standardization", "minmax" , "per_sample_std", "per_sample_minmax"
        
        spectrogram (bool): Whether to convert raw data into frequency representations
            scales : Depends on the sampling frequency of the data 
            wavelet : Methods of wavelet transformation

        """


        self.used_cols = [0,1,2,3,4]
        # there are total only 1 sensor
        self.col_names = ["accX", "accY", "accZ", "activity_id","sub"]

        self.label_map = [(0, 'StandingUpFS'), 
                          (1, 'StandingUpFL'), 
                          (2, 'Walking'), 
                          (3, 'Running'), 
                          (4, 'GoingUpS'), 
                          (5, 'Jumping'), 
                          (6, 'GoingDownS'), 
                          (7, 'LyingDownFS'), 
                          (8, 'SittingDown'), 
                          (9, 'FallingForw'), 
                          (10, 'FallingRight'), 
                          (11, 'FallingBack'), 
                          (12, 'HittingObstacle'), 
                          (13, 'FallingWithPS'), 
                          (14, 'FallingBackSC'),
                          (15, 'Syncope'),
                          (16, 'FallingLeft')]
        # There are in totoal 17 activities, 9 daily activites and 8 falls !
        self.drop_activities = []
        # drop_activities = [0,1,2,3,4,5,6,7,8]  # 9 ADL
        # drop_activities = [9,10,11,12,13,14,15,16] # 8 types of falls

        # There are in total 30 subjects.
        self.train_keys   = [0, 1, 2, 3, 4, 5,
                             6, 7, 8, 9, 10,11,
                             12,13,14,15,16,17,
                             18,19,20,21,22,23]
        self.vali_keys    = []
        self.test_keys    = [24,25,26,27,28,29]

        self.exp_mode     = args.exp_mode
        self.split_tag = "sub"

        self.LOCV_keys = [[0, 1, 2],
                          [3, 4, 5],
                          [6, 7, 8],
                          [9, 10,11],
                          [12,13,14],
                          [15,16,17],
                          [18,19,20],
                          [21,22,23],
                          [24,25,26],
                          [27,28,29]]

        self.all_keys = [0, 1, 2, 3, 4, 5,
                         6, 7, 8, 9, 10,11,
                         12,13,14,15,16,17,
                         18,19,20,21,22,23,
                         24,25,26,27,28,29]

        self.sub_ids_of_each_sub = {}

        self.file_encoding = {}  # no use 
        
        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        super(UNIMIB_HAR_DATA, self).__init__(args)


    def load_all_the_data(self, root_path):

        print(" ----------------------- load all the data -------------------")
        df_all = pd.read_csv(os.path.join(root_path,"UniMiB.csv"))


        self.sub_ids_of_each_sub = {}

        for sub_id in df_all["sub_id"].unique():
            sub = int(sub_id.split("_")[0])
            if sub not in self.sub_ids_of_each_sub.keys():
                self.sub_ids_of_each_sub[sub] = []
            self.sub_ids_of_each_sub[sub].append(sub_id)


        df_all = df_all.set_index('sub_id')

        # Label Transformation
        df_all["activity_id"] = df_all["activity_id"].map(self.labelToId)

        # reorder the columns as sensor1, sensor2... sensorn, sub, activity_id
        df_all = df_all[["acc_x","acc_y","acc_z"]+["sub"]+["activity_id"]]

        data_y = df_all.iloc[:,-1]
        data_x = df_all.iloc[:,:-1]

        data_x = data_x.reset_index()
        # sub_id, sensor1, sensor2... sensorn, sub, 

        return data_x, data_y