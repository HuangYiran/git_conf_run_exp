import pandas as pd
import numpy as np
import os
from dataloaders.dataloader_base import BASE_DATA
# ========================================       HHAR_HAR_DATA               =============================
class HHAR_HAR_DATA(BASE_DATA):

    """
    Activities: ‘Biking’, ‘Sitting’, ‘Standing’, ‘Walking’, ‘Stair Up’ and ‘Stair down’.
    Sensors: Two embedded sensors, i.e., Accelerometer and Gyroscope sampled at the highest frequency possible by the device
    Devices: 4 smartwatches (2 LG watches, 2 Samsung Galaxy Gears)
    8 smartphones (2 Samsung Galaxy S3 mini, 2 Samsung Galaxy S3, 2 LG Nexus 4, 2 Samsung Galaxy S+)
    Recordings: 9 users currently named: a,b,c,d,e,f,g,h,i consistently across all files.

    spampling HZ = 100
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


        self.used_cols = [0,1,2,3,4,5,6,7,8]  
        # there are total only 1 sensor
        self.col_names = ['acc_x', 'acc_y', 'acc_z', 
                          'gyo_x', 'gyo_y', 'gyo_z', 
                          'sub',  'activity_id', 'sub_id']

        self.label_map = [(0, 'bike'), 
                          (1, 'sit'), 
                          (2, 'stand'), 
                          (3, 'walk'), 
                          (4, 'stairsup'), 
                          (5, 'stairsdown')]

        self.drop_activities = []


        # There are in total 30 subjects.
        self.train_keys   = [0,1,2,3,4,5,6]
        self.vali_keys    = []
        self.test_keys    = [7,8]

        self.exp_mode     = args.exp_mode
        self.split_tag = "sub"

        self.LOCV_keys = [[0],[1],[2],[3],[4],[5],[6],[7],[8]]

        self.all_keys = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        self.sub_ids_of_each_sub = {}

        self.file_encoding = {}  # no use 
        
        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        super(HHAR_HAR_DATA, self).__init__(args)


    def load_all_the_data(self, root_path):

        print(" ----------------------- load all the data -------------------")
        df_all = pd.read_csv(os.path.join(root_path,"Hhar.csv"))

        label_mapping = {item[1]:item[0] for item in self.label_map}
        df_all["activity_id"] = df_all["activity_id"].map(label_mapping)


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
        df_all = df_all[['acc_x', 'acc_y', 'acc_z', 'gyo_x', 'gyo_y', 'gyo_z', 'sub',  'activity_id']]

        data_y = df_all.iloc[:,-1]
        data_x = df_all.iloc[:,:-1]

        data_x = data_x.reset_index()
        # sub_id, sensor1, sensor2... sensorn, sub, 

        return data_x, data_y