import pandas as pd
import numpy as np
import os
import pickle
from dataloaders.dataloader_base import BASE_DATA

# ========================================       SYNTHETIC_HAR_DATA             =============================
class SYNTHETIC_HAR_DATA(BASE_DATA):


    def __init__(self, args):




        self.used_cols = []
        self.col_names    =  ['acc_x']

        self.label_map = [(0, '1freq'), 
                          (1, "2freq"),
                          (2, "3freq/null"),
                          (3, "pos"),
                          (4, "neg")]

        self.drop_activities = []


        self.train_keys   = [1]
        self.vali_keys    = []
        self.test_keys    = [2]

        self.exp_mode     = args.exp_mode
        self.split_tag = "sub"

        self.LOCV_keys = [[1],[2]]
        self.all_keys = [1,2]
        self.sub_ids_of_each_sub = {}

        self.file_encoding = {} # no use

        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]

        super(SYNTHETIC_HAR_DATA, self).__init__(args)


    def load_all_the_data(self, root_path):
        print(" ----------------------- load all the data -------------------")

        file_list = os.listdir(root_path)
        df_dict = {}
        for i, file in enumerate(file_list):
            with open(os.path.join(root_path,file), 'rb') as handle:
                df = pickle.load(handle)
            activity = int(file.split(".")[0].split("_")[1])
            index = int(file.split(".")[0].split("_")[0])
            if i > int(0.8 * len(file_list)):
                sub=2
            else:
                sub =1

            sub_id = "{}_{}".format(sub,index)
            df = pd.DataFrame(df)
            df.columns = self.col_names
            df["sub_id"] = sub_id
            df["sub"] = sub
            df["activity_id"] = activity
            df_dict[sub_id] = df
            if sub not in self.sub_ids_of_each_sub.keys():
                self.sub_ids_of_each_sub[sub] = []
            self.sub_ids_of_each_sub[sub].append(sub_id)
        # all data
        df_all = pd.concat(df_dict)
        df_all = df_all.set_index('sub_id')			


        # reorder the columns as sensor1, sensor2... sensorn, sub, activity_id
        df_all = df_all[self.col_names+["sub"]+["activity_id"]]
        # label transformation
        df_all["activity_id"] = df_all["activity_id"].map(self.labelToId)
		
        data_y = df_all.iloc[:,-1]
        data_x = df_all.iloc[:,:-1]

        data_x = data_x.reset_index()

        return data_x, data_y
