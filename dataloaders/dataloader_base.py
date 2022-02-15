import pandas as pd
import numpy as np
import os
import random
import pywt
import pickle
from tqdm import tqdm
import torch
from random import sample
from dataloaders.utils import Normalizer
from sklearn.utils import class_weight
from skimage.transform import resize
# ========================================       Data loader Base class               =============================
class BASE_DATA():

    def __init__(self, args):
        """
        root_path : Root directory of the data set
        difference (bool) : Whether to calculate the first order derivative of the original data
        datanorm_type (str) : Methods of data normalization: "standardization", "minmax" , "per_sample_std", "per_sample_minmax"
        
        spectrogram (bool): Whether to convert raw data into frequency representations
            scales : Depends on the sampling frequency of the data （ UCI 数据的采样频率？？）
            wavelet : Methods of wavelet transformation

        """
        self.root_path    = args.root_path
        self.model_type   = args.model_type
        self.load_all     = args.load_all
        self.data_name    = args.data_name
        self.difference   = args.difference
        self.datanorm_type= args.datanorm_type
        self.train_vali_quote   = args.train_vali_quote


        self.drop_transition = args.drop_transition

        # the following parameters are all for slidingwindow set up 
        self.freq         = args.sampling_freq  
        self.windowsize   = args.windowsize



        self.wavename =  args.wavename
        self.freq_save_path = args.freq_save_path
        self.window_save_path = args.window_save_path


        self.data_x, self.data_y = self.load_all_the_data(self.root_path)
        # data_x : sub_id, sensor_1, sensor_2,..., sensor_n , sub
        # data_y : activity_id   index:sub_id
	
        if self.difference:
            self.data_x = self.differencing(self.data_x.set_index('sub_id').copy())
        # data_x : sub_id, sensor_1, sensor_2,..., sensor_n , sub

        #self.slidingwindows = self.get_the_sliding_index(self.data_x.copy(), self.data_y.copy())
        self.train_slidingwindows = self.get_the_sliding_index(self.data_x.copy(), self.data_y.copy(), "train")
        self.test_slidingwindows = self.get_the_sliding_index(self.data_x.copy(), self.data_y.copy(),"test")


        # compute/check the act_weights 
        # self.act_weights = self.update_classes_weight()
        #print("The orginal class weights are : ", self.act_weights)



        if self.model_type in ["freq","cross","cnn_freq", "cnn_lstm_freq", "cnn_lstm_cross"]:
            print("------------freq style -----------------")
            assert self.freq_save_path is not None
            self.genarate_spectrogram(flag = "train")
            self.genarate_spectrogram(flag = "test")

            if self.load_all:
                print("-------load all freq DATA --------")
                self.data_freq = {}
                for file in tqdm(self.train_freq_file_name):
                    with open(os.path.join(self.train_freq_path,"{}.pickle".format(file)), 'rb') as handle:
                        sample_x = pickle.load(handle)
                    self.data_freq[file] = sample_x

        if self.exp_mode in ["SOCV","FOCV"]:
            self.num_of_cv = 5
            self.index_of_cv = 0
            self.step = int(len(self.train_slidingwindows)/self.num_of_cv)
            self.window_index_list = list(np.arange(len(self.train_slidingwindows)))
            random.shuffle(self.window_index_list)
            if self.datanorm_type is not None:
                self.normalized_data_x = self.normalization(self.data_x.copy())
            else:
                self.normalized_data_x = self.data_x.copy()

        elif self.exp_mode == "LOCV":
            self.num_of_cv = len(self.LOCV_keys)
            self.index_of_cv = 0

        else:
            self.num_of_cv = 1

    def update_train_val_test_keys(self):
        if self.exp_mode in ["Given", "LOCV"]:
            if self.exp_mode == "LOCV":
                print("Leave one Out Experiment : The {} Part as the test".format(self.index_of_cv))
                # 因为每次都会变 所以LOCV模式下 需要每次都更新一下  根据index of cv
                self.test_keys =  self.LOCV_keys[self.index_of_cv]
                self.train_keys = [key for key in self.all_keys if key not in self.test_keys]
                self.index_of_cv = self.index_of_cv + 1



            # Normalization the data
            if self.datanorm_type is not None:
                train_vali_x = pd.DataFrame()
                for sub in self.train_keys:
                    temp = self.data_x[self.data_x[self.split_tag]==sub]
                    train_vali_x = pd.concat([train_vali_x,temp])

                test_x = pd.DataFrame()
                for sub in self.test_keys:
                    temp = self.data_x[self.data_x[self.split_tag]==sub]
                    test_x = pd.concat([test_x,temp])

            
                train_vali_x, test_x = self.normalization(train_vali_x, test_x)

                self.normalized_data_x = pd.concat([train_vali_x,test_x])
                self.normalized_data_x.sort_index(inplace=True)
            else:
                self.normalized_data_x = self.data_x.copy()


            # 根据test的keys  筛选出 window的第一个element有哪些
            all_test_keys = []
            if self.split_tag == "sub":
                for sub in self.test_keys:
                    all_test_keys.extend(self.sub_ids_of_each_sub[sub])
            else:
                all_test_keys = self.test_keys.copy()




            if self.exp_mode == "LOCV":
                test_file_name = os.path.join(self.window_save_path,"{}_droptrans_{}_windowsize_{}_LOCV_test_ID_{}.pickle".format(self.data_name, self.drop_transition, self.windowsize, self.index_of_cv-1))
                if os.path.exists(test_file_name):
                    with open(test_file_name, 'rb') as handle:
                        self.test_window_index = pickle.load(handle)
                else:
                    self.test_window_index = []
                    for index, window in enumerate(self.test_slidingwindows):
                        sub_id = window[0]
                        if sub_id in all_test_keys:
                            self.test_window_index.append(index)
                    with open(test_file_name, 'wb') as handle:
                        pickle.dump(self.test_window_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

            else:
                given_file_name = os.path.join(self.window_save_path,"{}_droptrans_{}_windowsize_{}_given_test.pickle".format(self.data_name, self.drop_transition, self.windowsize))
                if os.path.exists(given_file_name):
                    with open(given_file_name, 'rb') as handle:
                        self.test_window_index = pickle.load(handle)
                else:
                    self.test_window_index = []
                    for index, window in enumerate(self.test_slidingwindows):
                        sub_id = window[0]
                        if sub_id in all_test_keys:
                            self.test_window_index.append(index)
                    with open(given_file_name, 'wb') as handle:
                        pickle.dump(self.test_window_index, handle, protocol=pickle.HIGHEST_PROTOCOL)


            if self.exp_mode == "LOCV":
                train_file_name = os.path.join(self.window_save_path,"{}_droptrans_{}_windowsize_{}_LOCV_train_ID_{}.pickle".format(self.data_name, self.drop_transition, self.windowsize, self.index_of_cv-1))
                if os.path.exists(train_file_name):
                    with open(train_file_name, 'rb') as handle:
                        train_vali_window_index = pickle.load(handle)
                else:
                    train_vali_window_index = []
                    for index, window in enumerate(self.train_slidingwindows):
                        sub_id = window[0]
                        if sub_id not in all_test_keys:
                            train_vali_window_index.append(index)
                    with open(train_file_name, 'wb') as handle:
                        pickle.dump(train_vali_window_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

            else:
                given_file_name = os.path.join(self.window_save_path,"{}_droptrans_{}_windowsize_{}_given_train.pickle".format(self.data_name, self.drop_transition, self.windowsize))
                if os.path.exists(given_file_name):
                    with open(given_file_name, 'rb') as handle:
                        train_vali_window_index = pickle.load(handle)
                else:
                    train_vali_window_index = []
                    for index, window in enumerate(self.train_slidingwindows):
                        sub_id = window[0]
                        if sub_id not in all_test_keys:
                            train_vali_window_index.append(index)
                    with open(given_file_name, 'wb') as handle:
                        pickle.dump(train_vali_window_index, handle, protocol=pickle.HIGHEST_PROTOCOL)


            random.shuffle(train_vali_window_index)
            self.train_window_index = train_vali_window_index[:int(self.train_vali_quote*len(train_vali_window_index))]
            self.vali_window_index = train_vali_window_index[int(self.train_vali_quote*len(train_vali_window_index)):]

        elif self.exp_mode in ["SOCV","FOCV"]:
            print("Overlapping random Experiment : The {} Part as the test".format(self.index_of_cv))
            start = self.index_of_cv * self.step
            if self.index_of_cv < self.num_of_cv-1:
                end = (self.index_of_cv+1) * self.step
            else:
                end = len(self.train_slidingwindows)

            train_vali_index = self.window_index_list[0:start] + self.window_index_list[end:len(self.window_index_list)]
            self.test_window_index = self.window_index_list[start:end] 
            # copy shuffle
            self.train_window_index = train_vali_index[:int(self.train_vali_quote*len(train_vali_index))]
            self.vali_window_index = train_vali_index[int(self.train_vali_quote*len(train_vali_index)):]

            self.index_of_cv = self.index_of_cv + 1


        else:
            raise NotImplementedError

        # update_classes_weight
        # class_transform = {x: i for i, x in enumerate(self.no_drop_activites)}

        # y_of_all_windows  = []
        # for index in self.train_window_index:
        #     window = self.slidingwindows[index]
        #     start_index = window[1]
        #     end_index = window[2]
        #     y_of_all_windows.append(class_transform[self.data_y.iloc[start_index:end_index].mode().loc[0]])
        #act_weights = class_weight.compute_class_weight(class_weight ='balanced',classes = range(len(self.no_drop_activites)),y = y_of_all_windows)
        #self.act_weights = act_weights.round(4)
        #print("The class weights are : ", self.act_weights)
        self.act_weights = self.update_classes_weight()

    def load_all_the_data(self, root_path):
        raise NotImplementedError


    def differencing(self, df):
        # columns = [, "acc_x"..."acc_y", "sub"], index is  sub_id
        # define the name for differenced columns
        sensor_cols = df.columns[:-1]
        columns = ["diff_"+i for i in sensor_cols]

        # The original data has been divided into segments by sub_id: a segment belongs to a same user 
        # There is no continuity between different segments, so diffrecne is only done within each segment

        # Train_vali_diff
        diff_data = []
        for id in df.index.unique():
            diff_data.append(df.loc[id,sensor_cols].diff())

        diff_data = pd.concat(diff_data)
        diff_data.columns = columns
        diff_data.fillna(method ="backfill",inplace=True)
        data = pd.concat([df.iloc[:,:-1],diff_data, df.iloc[:,-1]], axis=1)

        return data.reset_index()

    def normalization(self, train_vali, test=None):
        train_vali_sensors = train_vali.iloc[:,1:-1]
        self.normalizer = Normalizer(self.datanorm_type)
        self.normalizer.fit(train_vali_sensors)
        train_vali_sensors = self.normalizer.normalize(train_vali_sensors)
        train_vali_sensors = pd.concat([train_vali.iloc[:,0],train_vali_sensors,train_vali.iloc[:,-1]], axis=1)
        if test is None:
            return train_vali_sensors
        else:
            test_sensors  = test.iloc[:,1:-1]
            test_sensors  = self.normalizer.normalize(test_sensors)
            test_sensors  =  pd.concat([test.iloc[:,0],test_sensors,test.iloc[:,-1]], axis=1)
            return train_vali_sensors, test_sensors

    def get_the_sliding_index(self, data_x, data_y , flag = "train"):
        """
        Because of the large amount of data, it is not necessary to store all the contents of the slidingwindow, 
        but only to access the index of the slidingwindow
        Each window consists of three parts: sub_ID , start_index , end_index
        The sub_ID ist used for train test split, if the subject train test split is applied
        """
        if os.path.exists(os.path.join(self.window_save_path,"{}_{}_drop_trans_{}.pickle".format(self.data_name, flag, self.drop_transition))):
            print("-----------------------Sliding file are generated -----------------------")
            with open(os.path.join(self.window_save_path,"{}_{}_drop_trans_{}.pickle".format(self.data_name, flag, self.drop_transition)), 'rb') as handle:
                window_index = pickle.load(handle)
        else:
            print("----------------------- Get the Sliding Window -------------------")

            data_y = data_y.reset_index()

            data_x["activity_id"] = data_y["activity_id"]
            if self.drop_transition:
                data_x['act_block'] = ((data_x['activity_id'].shift(1) != data_x['activity_id']) | (data_x['sub_id'].shift(1) != data_x['sub_id'])).astype(int).cumsum()
            else:
                data_x['act_block'] = (data_x['sub_id'].shift(1) != data_x['sub_id']).astype(int).cumsum()

            freq         = self.freq   
            windowsize   = self.windowsize
            if flag == "train":
                displacement = int(0.5 * self.windowsize)
            elif flag == "test":
                displacement = int(0.1 * self.windowsize)

            window_index = []
            for index in data_x.act_block.unique():

                temp_df = data_x[data_x["act_block"]==index]
                assert len(temp_df["activity_id"].unique())==1

                if temp_df["activity_id"].unique()[0] not in self.drop_activities:
                    assert len(temp_df["sub_id"].unique()) == 1
                    sub_id = temp_df["sub_id"].unique()[0]
                    start = temp_df.index[0]
                    end   = start+windowsize

                    while end <= temp_df.index[-1] + 1 :

                        window_index.append([sub_id, start, end])

                        start = start + displacement
                        end   = start + windowsize

            with open(os.path.join(self.window_save_path,"{}_{}_drop_trans_{}.pickle".format(self.data_name, flag, self.drop_transition)), 'wb') as handle:
                pickle.dump(window_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return window_index


    def genarate_spectrogram(self, flag="train"):
        save_path = os.path.join(self.freq_save_path,self.data_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        if flag == "train":
            displacement = int(0.5 * self.windowsize)
            slidingwindows = self.train_slidingwindows
            self.train_freq_path = os.path.join(save_path,"diff_{}_window_{}_step_{}_drop_trans_{}".format(self.difference, self.windowsize,displacement, self.drop_transition))
            freq_path = self.train_freq_path
        elif flag == "test":
            displacement = int(0.1 * self.windowsize)
            slidingwindows = self.test_slidingwindows
            self.test_freq_path = os.path.join(save_path,"diff_{}_window_{}_step_{}_drop_trans_{}".format(self.difference, self.windowsize, displacement, self.drop_transition))
            freq_path = self.test_freq_path




        if os.path.exists(freq_path):
            print("----------------------- file are generated -----------------------")
            if flag == "train":
                with open(os.path.join(freq_path,"freq_file_name.pickle"), 'rb') as handle:
                    self.train_freq_file_name = pickle.load(handle)
            else:
                with open(os.path.join(freq_path,"freq_file_name.pickle"), 'rb') as handle:
                    self.test_freq_file_name = pickle.load(handle)

        else:
            print("----------------------- spetrogram generating for {} -----------------------".format(flag))
            os.mkdir(freq_path)

            scales1 = np.arange(1, self.freq + 1) 

            totalscal = self.freq 
            fc = pywt.central_frequency(self.wavename)#计算小波函数的中心频率
            cparam = 2 * fc * totalscal  #常数c
            scales2 = cparam/np.arange(totalscal,0,-1) #为使转换后的频率序列是一等差序列，尺度序列必须取为这一形式（也即小波尺度）

            if self.windowsize >= 60 and "synthetic" not in self.data_name:
                l_scale = 2
            else:
                l_scale = 1

            if self.freq >=40 and "synthetic" not in self.data_name:
                f_scale = 2
            else:
                f_scale = 1

            if f_scale>1 or l_scale>1:
                resize_flag = True
            else:
                resize_flag = False
            

            freq_file_name = []

            temp_data = self.normalization(self.data_x.copy())
            for window in slidingwindows:
                sub_id = window[0]
                start_index = window[1]
                end_index = window[2]
	
                name = "{}_{}_{}".format(sub_id,start_index,end_index)
                freq_file_name.append(name)

                sample_x = temp_data.iloc[start_index:end_index,1:-1].values
                scalogram = []

                for j in range(sample_x.shape[1]):
                    if self.difference and j>= int(sample_x.shape[1]/2):
                        [cwtmatr, frequencies] = pywt.cwt(sample_x[:,j],   scales2,  self.wavename, sampling_period = 1.0/self.freq)#连续小波变换模块
                    else:
                        [cwtmatr, frequencies] = pywt.cwt(sample_x[:,j],   scales1,  self.wavename, sampling_period = 1.0/self.freq)#连续小波变换模块
                    if resize_flag:
                        cwtmatr = resize(cwtmatr, (int(self.freq/f_scale), int(self.windowsize/l_scale)), mode = 'constant')
                    scalogram.append(cwtmatr)

                scalogram = np.stack(scalogram)

                with open(os.path.join(freq_path,"{}.pickle".format(name)), 'wb') as handle:
                    pickle.dump(scalogram, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(os.path.join(freq_path,"freq_file_name.pickle"), 'wb') as handle:
                pickle.dump(freq_file_name, handle, protocol=pickle.HIGHEST_PROTOCOL)
            if flag == "train":
                self.train_freq_file_name = freq_file_name
            else:
                self.test_freq_file_name = freq_file_name

    def update_classes_weight(self):
        class_transform = {x: i for i, x in enumerate(self.no_drop_activites)}

        y_of_all_windows  = []

        for index in self.train_window_index:

            start_index = self.train_slidingwindows[index][1]
            end_index = self.train_slidingwindows[index][2]

            y_of_all_windows.append(class_transform[self.data_y.iloc[start_index:end_index].mode().loc[0]])
        y_of_all_windows = np.array(y_of_all_windows)
        target_count = np.array([np.sum(y_of_all_windows == label) for label in set(y_of_all_windows)])
        weight_target = 1.0 / target_count
        weight_samples = np.array([weight_target[t] for t in y_of_all_windows])
        weight_samples = torch.from_numpy(weight_samples)
        weight_samples = weight_samples.double()
        #act_weights = class_weight.compute_class_weight(class_weight ='balanced',classes =range(len(self.no_drop_activites)), y =y_of_all_windows)
        #act_weights = act_weights.round(4)

        print("[-] Target sampling weights:")
        print(weight_target)

        return weight_samples