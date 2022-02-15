import pandas as pd
import numpy as np
import os
import scipy.io as sio
from dataloaders.dataloader_base import BASE_DATA
# ========================================       WARD_HAR_DATA               =============================
class WARD_HAR_DATA(BASE_DATA):

    """
    Reading: A cell structure.
    The number of cells is equal to the number of sensors in the network. 
    Each cell contains an array structure of dimension 5×t, w
    here 5 corresponds to the sensor readings from 3-axis accelerometer and 2-axis gyroscope on one sensor node, 
    and t represents the length of the trial sequence.

    • Sensor 1: Outside center of the lower left forearm joint. The y-axis of the gyroscope points to the hand.
    • Sensor 2: Outside center of the lower right forearm joint. The y-axis of the gyroscope points to the hand.
    • Sensor 3: Front center of the waist. The x-axis of the gyroscope points down.
    • Sensor 4: Outside center of the left ankle. The y-axis of the gyroscope points to the foot.
    • Sensor 5: Outside center of the right ankle. The y-axis of the gyroscope points to the foot.
	
    20Hz
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


        self.used_cols = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]  
        # there are total 5 sensors, each has 5 channels

        poses = ["left forearm", "right forearm", "waist", "left ankle", "right ankle"]
        channel = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y"]
        self.col_names = [item for sublist in [[col+'_'+pos for col in channel] for pos in poses] for item in sublist]


        self.label_map = [(1, 'Rest at Standing'), 
                          (2, 'Rest at Sitting'), 
                          (3, 'Rest at Lying'), 
                          (4, 'Walk forward'), 
                          (5, 'Walk forward left-circle'), 
                          (6, 'Walk forward right-circle'), 
                          (7, 'Turn left'), 
                          (8, 'Turn right'), 
                          (9, 'Go upstairs'), 
                          (10, 'Go downstairs'), 
                          (11, 'Jog'), 
                          (12, 'Jump'), 
                          (13, 'Push wheelchair ')]
        # There are in totoal 13 !
        self.drop_activities = []


        # There are in total 20 subjects.
        self.train_keys   = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
        self.vali_keys    = []
        self.test_keys    = [18,19,20]

        self.exp_mode     = args.exp_mode
        self.split_tag = "sub"

        self.LOCV_keys = [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[17,18],[19,20]]

        self.all_keys = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

        self.sub_ids_of_each_sub = {}

        self.file_encoding = {}  # no use 
        
        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        super(WARD_HAR_DATA, self).__init__(args)


    def load_all_the_data(self, root_path):

        print(" ----------------------- load all the data -------------------")

        df_dict = {}

        file_list = os.listdir(root_path)
        file_list = [file for file in file_list if "Sub" in file]

        assert len(file_list)==20

        for file in file_list:
            files_of_sub = os.listdir(os.path.join(root_path,file))
            sub = int(file[7:])
            for mat in files_of_sub:
                activity_id = int(mat[1:-6])
                trial = int(mat[-5])
                sub_id = "{}_{}_{}".format(sub, activity_id, trial)

                data = sio.loadmat(file_name=os.path.join(root_path, file, mat))

                values = np.concatenate([data["WearableData"][0][0][5][0][0],
                                         data["WearableData"][0][0][5][0][1],
                                         data["WearableData"][0][0][5][0][2],
                                         data["WearableData"][0][0][5][0][3],
                                         data["WearableData"][0][0][5][0][4],],axis=1)

                df = pd.DataFrame(values, columns= self.col_names)
                df["sub"] = sub
                df["activity_id"] = activity_id
                df["sub_id"] = sub_id


                if sub not in self.sub_ids_of_each_sub.keys():
                    self.sub_ids_of_each_sub[sub] = []
                self.sub_ids_of_each_sub[sub].append(sub_id)
                df_dict[sub_id] = df
        # all data
        df_all = pd.concat(df_dict)
        df_all = df_all.set_index('sub_id')

        # Label Transformation
        df_all["activity_id"] = df_all["activity_id"].map(self.labelToId)

        # reorder the columns as sensor1, sensor2... sensorn, sub, activity_id
        df_all = df_all[self.col_names+["sub"]+["activity_id"]]

        data_y = df_all.iloc[:,-1]
        data_x = df_all.iloc[:,:-1]

        data_x = data_x.reset_index()
        # sub_id, sensor1, sensor2... sensorn, sub, 

        return data_x, data_y