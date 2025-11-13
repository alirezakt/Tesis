import os
import pickle
import random
import time
import numpy as np
import torch
from geopy.distance import geodesic

DATASET_NAME_TO_NUM = {
    'atl0801': 0,
    'atl0802': 1,
    'atl0803': 2,
    'atl0804': 3,
    'atl0805': 4,
    'atl0806': 5,
    'atl0807': 6,
}


class Trajectory_Dataloader():
    def __init__(self, args):
        DATASET_NAME_TO_NUM = {
                'atl0801': 0,
                'atl0802': 1,
                'atl0803': 2,
                'atl0804': 3,
                'atl0805': 4,
                'atl0806': 5,
                'atl0807': 6,
            }
        self.args = args
        if self.args.dataset == 'eth5':

            self.data_dirs = ['D:\\STAR\\data\\eth5\\eth\\univ', 'D:\\STAR\\data\\eth5\\eth\\hotel',
                              'D:\\STAR\\data\\eth5\\ucy\\zara\\zara01', 'D:\\STAR\\data\\eth5\\ucy\\zara\\zara02',
                              'D:\\STAR\\data\\eth5\\ucy\\univ\\students001', 'D:\\STAR\\data\\eth5\\ucy\\univ\\students003',
                              'D:\\STAR\\data\\eth5\\ucy\\univ\\uni_examples', 'D:\\STAR\\data\\eth5\\ucy\\zara\\zara03']

            # Data directory where the pre-processed pickle file resides
            self.data_dir = './data'
            skip = [6, 10, 10, 10, 10, 10, 10, 10]

            train_set = [i for i in range(len(self.data_dirs))]

            assert args.test_set in DATASET_NAME_TO_NUM.keys(), 'Unsupported dataset {}'.format(args.test_set)

            args.test_set = DATASET_NAME_TO_NUM[args.test_set]

            #if args.test_set == 4 or args.test_set == 5:
            #    self.test_set = [4, 5]
            #else:
            self.test_set = [self.args.test_set]

            for x in self.test_set:
                train_set.remove(x)

            self.train_dir = [self.data_dirs[x] for x in train_set]
            self.test_dir = [self.data_dirs[x] for x in self.test_set]
            self.trainskip = [skip[x] for x in train_set]
            self.testskip = [skip[x] for x in self.test_set]
        elif self.args.dataset == 'iffatl':

            self.data_dirs = ['D:\\STAR\\data\\iff\\atl\\20190801', 'D:\\STAR\\data\\iff\\atl\\20190802',
                              'D:\\STAR\\data\\iff\\atl\\20190803', 'D:\\STAR\\data\\iff\\atl\\20190804',
                              'D:\\STAR\\data\\iff\\atl\\20190805', 'D:\\STAR\\data\\iff\\atl\\20190806',
                              'D:\\STAR\\data\\iff\\atl\\20190807'
                              ]

            # Data directory where the pre-processed pickle file resides
            self.data_dir = 'D:\\STAR\\data'
            skip = [item for item in [args.skip] for i in range(len(self.data_dirs))]

            train_set = [i for i in range(len(self.data_dirs))]
            DATASET_NAME_TO_NUM = {
                'atl0801': 0,
                'atl0802': 1,
                'atl0803': 2,
                'atl0804': 3,
                'atl0805': 4,
                'atl0806': 5,
                'atl0807': 6,
            }
            assert args.test_set in DATASET_NAME_TO_NUM.keys(), 'Unsupported dataset {}'.format(args.test_set)

            args.test_set = DATASET_NAME_TO_NUM[args.test_set]

            self.test_set = [self.args.test_set]

            for x in self.test_set:
                train_set.remove(x)

            self.train_dir = [self.data_dirs[x] for x in train_set]
            self.test_dir = [self.data_dirs[x] for x in self.test_set]
            self.trainskip = [skip[x] for x in train_set]
            self.testskip = [skip[x] for x in self.test_set]
        elif self.args.dataset == 'kagc':

            self.data_dirs = ['D:\\STAR\\data\\kagc\\1', 'D:\\STAR\\data\\kagc\\2',
                              'D:\\STAR\\data\\kagc\\3', 'D:\\STAR\\data\\kagc\\4',
                              'D:\\STAR\\data\\kagc\\5', 'D:\\STAR\\data\\kagc\\6',
                              'D:\\STAR\\data\\kagc\\7', 'D:\\STAR\\data\\kagc\\8',
                              'D:\\STAR\\data\\kagc\\9', 'D:\\STAR\\data\\kagc\\10',
                              ]

            # Data directory where the pre-processed pickle file resides
            self.data_dir = 'D:\\STAR\\data'
            skip = [item for item in [args.skip] for i in range(len(self.data_dirs))]

            train_set = [i for i in range(len(self.data_dirs))]
            DATASET_NAME_TO_NUM = {
                '1': 0,
                '2': 1,
                '3': 2,
                '4': 3,
                '5': 4,
                '6': 5,
                '7': 6,
                '8': 7,
                '9': 8,
                '10': 9,
            }
            assert args.test_set in DATASET_NAME_TO_NUM.keys(), 'Unsupported dataset {}'.format(args.test_set)

            args.test_set = DATASET_NAME_TO_NUM[args.test_set]

            self.test_set = [self.args.test_set]

            for x in self.test_set:
                train_set.remove(x)

            self.train_dir = [self.data_dirs[x] for x in train_set]
            self.test_dir = [self.data_dirs[x] for x in self.test_set]
            self.trainskip = [skip[x] for x in train_set]
            self.testskip = [skip[x] for x in self.test_set]

        elif self.args.dataset == 'oiie':
            DATASET_NAME_TO_NUM = {
                    '1': 0,
                    '2': 1,
                    '3': 2,
                    '4': 3,
                    '5': 4,
                    '6': 5,
                    '7': 6,
                    '8': 7,
                    '9': 8,
                    '10': 9,
                }

            self.data_dirs = ['D:\\STAR\\data\\oiie\\1', 'D:\\STAR\\data\\oiie\\2',
                  'D:\\STAR\\data\\oiie\\3', 'D:\\STAR\\data\\oiie\\4',
                  'D:\\STAR\\data\\oiie\\5', 'D:\\STAR\\data\\oiie\\6',
                  'D:\\STAR\\data\\oiie\\7', 'D:\\STAR\\data\\oiie\\8',
                  'D:\\STAR\\data\\oiie\\9', 'D:\\STAR\\data\\oiie\\10',
                  ]

            # Data directory where the pre-processed pickle file resides
            skip = [item for item in [args.skip] for i in range(len(self.data_dirs))]

            train_set = [i for i in range(len(self.data_dirs))]

            assert args.test_set in DATASET_NAME_TO_NUM.keys(), 'Unsupported dataset {}'.format(args.test_set)

            args.test_set = DATASET_NAME_TO_NUM[args.test_set]

            self.test_set = [self.args.test_set]

            for x in self.test_set:
                train_set.remove(x)

            self.train_dir = [self.data_dirs[x] for x in train_set]
            self.test_dir = [self.data_dirs[x] for x in self.test_set]
            self.trainskip = [skip[x] for x in train_set]
            self.testskip = [skip[x] for x in self.test_set]

        self.train_data_file = os.path.join(self.args.save_dir, "train_trajectories.cpkl")
        self.test_data_file = os.path.join(self.args.save_dir, "test_trajectories.cpkl")
        self.train_batch_cache = os.path.join(self.args.save_dir, "train_batch_cache.cpkl")
        self.test_batch_cache = os.path.join(self.args.save_dir, "test_batch_cache.cpkl")

        print("Creating pre-processed data from raw data.")
        self.traject_preprocess('train')
        self.traject_preprocess('test')
        print("Done.")

        # Load the processed data from the pickle file
        print("Preparing data batches.")
        if not (os.path.exists(self.train_batch_cache)):
            self.frameped_dict, self.pedtraject_dict = self.load_dict(self.train_data_file)
            self.dataPreprocess('train')
        if not (os.path.exists(self.test_batch_cache)):
            self.test_frameped_dict, self.test_pedtraject_dict = self.load_dict(self.test_data_file)
            self.dataPreprocess('test')

        self.trainbatch, self.trainbatchnums, _, _ = self.load_cache(self.train_batch_cache)
        self.testbatch, self.testbatchnums, _, _ = self.load_cache(self.test_batch_cache)
        print("Done.")

        print('Total number of training batches:', self.trainbatchnums)
        print('Total number of test batches:', self.testbatchnums)

        self.reset_batch_pointer(set='train', valid=False)
        self.reset_batch_pointer(set='train', valid=True)
        self.reset_batch_pointer(set='test', valid=False)   

    def traject_preprocess(self, setname):
        '''
        Preprocesses trajectory data for a given dataset.

        Parameters:
            setname (str): The name of the dataset ('train' or 'test').

        Returns:
            None
        '''
        if setname == 'train':
            data_dirs = self.train_dir
            data_file = self.train_data_file
        else:
            data_dirs = self.test_dir
            data_file = self.test_data_file
        all_frame_data = []
        valid_frame_data = []
        numFrame_data = []

        Pedlist_data = []
        frameped_dict = []  # peds id contained in a certain frame
        pedtrajec_dict = []  # trajectories of a certain ped

        # For each dataset
        for seti, directory in enumerate(data_dirs):

            file_path = os.path.join(directory, 'true_pos_.csv')
            # Load the data from the csv file
            data = np.genfromtxt(file_path, delimiter=',')

            # Pedestrains IDs in the current dataset
            Pedlist = np.unique(data[1, :]).tolist()
            numPeds = len(Pedlist)

            # Add the list of PedIDs to the Pedlist_data
            Pedlist_data.append(Pedlist)

            # Initialize the list of numpy arrays for the current dataset
            all_frame_data.append([])
            valid_frame_data.append([])
            numFrame_data.append([])
            frameped_dict.append({})
            pedtrajec_dict.append({})

            for ind, pedi in enumerate(Pedlist):
                #if ind % 100 == 0:
                #    print(ind, len(Pedlist))

                # Extract trajectories of one person
                FrameContainPed = data[:, data[1, :] == pedi]

                # Extract frame list
                FrameList = FrameContainPed[0, :].tolist()
                if len(FrameList) < 2:
                    continue

                # Add number of frames of this trajectory
                numFrame_data[seti].append(len(FrameList))

                # Initialize the row of the numpy array
                Trajectories = []

                # For each ped in the current frame
                for fi, frame in enumerate(FrameList):
                    # Extract their x and y positions
                    current_x = FrameContainPed[3, FrameContainPed[0, :] == frame][0]  # row 4
                    current_y = FrameContainPed[2, FrameContainPed[0, :] == frame][0]  # row 3
                    current_z = 0.6
                    if self.args.dataset == 'kagc':
                        current_z = FrameContainPed[4, FrameContainPed[0, :] == frame][0]
                    elif self.args.dataset == 'oiie' and len(FrameContainPed[4, FrameContainPed[0, :] == frame]) > 0:
                        current_z = FrameContainPed[4, FrameContainPed[0, :] == frame][0] * 0.0001

                    # Add their pedID, x, y to the row of the numpy array
                    Trajectories.append([int(frame), current_x, current_y, current_z])
                    if int(frame) not in frameped_dict[seti]:
                        frameped_dict[seti][int(frame)] = []
                    frameped_dict[seti][int(frame)].append(pedi)
                pedtrajec_dict[seti][pedi] = np.array(Trajectories)

        f = open(data_file, "wb")
        pickle.dump((frameped_dict, pedtrajec_dict), f, protocol=2)
        f.close()

    def get_data_index(self, data_dict, setname, ifshuffle=True):
            '''
            Get the dataset sampling index.

            Parameters:
            - data_dict (dict): A dictionary containing the dataset.
            - setname (str): The name of the dataset (e.g., 'train', 'test', 'validation').
            - ifshuffle (bool): Whether to shuffle the data index.

            Returns:
            - data_index (numpy.ndarray): An array containing the dataset sampling index.
            '''
            set_id = []  # dataset id
            frame_id_in_set = []
            total_frame = 0
            for seti, dict in enumerate(data_dict):
                frames = sorted(dict)
                maxframe = max(frames) - self.args.seq_length
                frames = [x for x in frames if not x > maxframe]
                total_frame += len(frames)  #1444
                set_id.extend(list(seti for i in range(len(frames))))
                frame_id_in_set.extend(list(frames[i] for i in range(len(frames))))

            all_frame_id_list = list(i for i in range(total_frame))

            # data index has three rows: time frame id in dataset id#, dataset id#, reindexed time frame id in the entire dataset
            data_index = np.concatenate((np.array([frame_id_in_set], dtype=int), np.array([set_id], dtype=int), np.array([all_frame_id_list], dtype=int)), 0)

            if ifshuffle:
                random.Random().shuffle(all_frame_id_list)
            data_index = data_index[:, all_frame_id_list]

            # to make full use of the data
            if setname == 'train':
                data_index = np.append(data_index, data_index[:, :self.args.batch_size], 1)
            return data_index

    def load_dict(self, data_file):
        f = open(data_file, 'rb')
        raw_data = pickle.load(f)
        f.close()

        frameped_dict = raw_data[0]
        pedtraject_dict = raw_data[1]

        return frameped_dict, pedtraject_dict

    def load_cache(self, data_file):
        f = open(data_file, 'rb')
        raw_data = pickle.load(f)
        f.close()
        return raw_data

    def dataPreprocess(self, setname):
        '''
        Function to load the pre-processed data into the DataLoader object
        frameped_dict: This dictionary keeps track of the pedestrians present in each frame. It has the structure: {set_index: {frame_index: [pedestrian_ids]}}. //در هر فریم زمانی چه هواپیماهایی هستند
        pedtraject_dict: This dictionary contains the trajectories of individual pedestrians. It has the structure: {set_index: {pedestrian_id: trajectory_array}}. The trajectory_array includes information about the pedestrian's position (x, y coordinates) at different time steps.
        '''
        if setname == 'train':
            val_fraction = 0
            frameped_dict = self.frameped_dict
            pedtraject_dict = self.pedtraject_dict
            cachefile = self.train_batch_cache

        else:
            val_fraction = 0
            frameped_dict = self.test_frameped_dict
            pedtraject_dict = self.test_pedtraject_dict
            cachefile = self.test_batch_cache

        if setname != 'train':
            shuffle = False
        else:
            shuffle = True
        #data_index is an array with 3 row. first row is frameId in dataset, second row is dataset ID, Third row is frame index in all datasets
        data_index = self.get_data_index(frameped_dict, setname, ifshuffle=shuffle)
        # split the dataset into training and validation
        # get validation data index from the first val_fraction% of the data index
        val_index = data_index[:, :int(data_index.shape[1] * val_fraction)]
        # get training data index from the rest of the data index
        train_index = data_index[:, (int(data_index.shape[1] * val_fraction) + 1):]
        # get the training trajectories fragments from data sampling index
        trainbatch = self.get_seq_from_index_balance(frameped_dict, pedtraject_dict, train_index, setname)
        # get the validation trajectories fragments from data sampling index
        valbatch = self.get_seq_from_index_balance(frameped_dict, pedtraject_dict, val_index, setname)

        if setname == 'train':
            trainbatch = trainbatch[:30]
        else:
            trainbatch = trainbatch[:5]
      
        trainbatchnums = len(trainbatch)
        valbatchnums = len(valbatch)

        f = open(cachefile, "wb")
        pickle.dump((trainbatch, trainbatchnums, valbatch, valbatchnums), f, protocol=2)
        f.close()
    


    def get_seq_from_index_balance(self, frameped_dict, pedtraject_dict, data_index, setname):
        '''
        Query the trajectories fragments from data sampling index.
        Notes: Divide the scene if there are too many people; accumulate the scene if there are few people.
               This function takes less gpu memory.
        '''
        
        batch_data_mass = []
        batch_data = []
        Batch_id = []

        temp = self.args.batch_around_ped
        if setname == 'train':
            skip = self.trainskip  # timeframe skip number for each dataset id#
        else:
            skip = self.testskip

        ped_cnt = 0
        last_frame = 0
        batch_pednum = 0
        #data_index is an array with 3 row. first row is frameId in dataset, second row is dataset ID, Third row is frame index in all datasets
        for i in range(data_index.shape[1]):
            if i % 100 == 0:
                print(i, '/', data_index.shape[1])
            cur_frame, cur_set, _ = data_index[:, i]
            #get all pedestrain id exist at first frame
            framestart_pedi = set(frameped_dict[cur_set][cur_frame])  # all ped ids exist in the current timeframe
            try:
                #get all pedestrain id in last frame
                frameend_pedi = set(frameped_dict[cur_set][cur_frame + self.args.seq_length * skip[cur_set]])
            except:
                continue
            present_pedi = framestart_pedi | frameend_pedi  # get all ped ids exist in the startframe or endframe

            if (framestart_pedi & frameend_pedi).__len__() == 0:
                continue

            traject = ()
            IFfull = []
            for ped in present_pedi:
                cur_trajec, iffull, ifexistobs = self.find_trajectory_fragment(pedtraject_dict[cur_set][ped], cur_frame, self.args.seq_length, skip[cur_set])

                if len(cur_trajec) == 0:
                    continue
                if ifexistobs == False:
                    # Just ignore trajectories if their data don't exsist at the last obversed time step (easy for data shift)
                    continue
                if sum(cur_trajec[:, 0] > 0) < 5:
                    # filter trajectories have too few frame data (less than 5)
                    continue
                #cur_trajec[:, 1:] remove the first column from cur_trajec. The first column is the frame number. the rest of the columns are the x and y coordinates of the trajectory.
                #cur_trajec[:, 1:].reshape(-1, 1, 3) reshape the cur_trajec array into a 3d array with shape (num_rows, 1, 2). The 1 in the second dimension is the number of columns in the reshaped array. The 2 in the third dimension is the number of columns in the original array.
                cur_trajec = (cur_trajec[:, 1:].reshape(-1, 1, 3),)  # modify this if 3d
                traject = traject.__add__(cur_trajec)
                IFfull.append(iffull)

            if traject.__len__() < 1:
                continue
            if sum(IFfull) < 1:
                continue
            traject_batch = np.concatenate(traject, 1)
            batch_pednum = sum([i.shape[1] for i in batch_data]) + traject_batch.shape[1]

            cur_pednum = traject_batch.shape[1]
            ped_cnt += cur_pednum
            batch_id = (cur_set, cur_frame,)

            if cur_pednum >= self.args.batch_around_ped * 2:  # self.args.batch_around_ped:256  controls how many agents in one frame you prefer
                # if too many people in current scene
                # split the scene into two batches
                ind = traject_batch[self.args.obs_length - 1].argsort(0)
                cur_batch_data, cur_Batch_id = [], []
                Seq_batchs = [traject_batch[:, ind[:cur_pednum // 2, 0]], traject_batch[:, ind[cur_pednum // 2:, 0]]]
                for sb in Seq_batchs:
                    cur_batch_data.append(sb)
                    cur_Batch_id.append(batch_id)
                    cur_batch_data = self.massup_batch(cur_batch_data)
                    batch_data_mass.append((cur_batch_data, cur_Batch_id,))
                    cur_batch_data = []
                    cur_Batch_id = []

                last_frame = i
            elif cur_pednum >= self.args.batch_around_ped:
                # good pedestrian numbers
                cur_batch_data, cur_Batch_id = [], []
                cur_batch_data.append(traject_batch)
                cur_Batch_id.append(batch_id)
                cur_batch_data = self.massup_batch(cur_batch_data)
                batch_data_mass.append((cur_batch_data, cur_Batch_id,))

                last_frame = i
            else:  # less pedestrian numbers < batch_around_ped
                # accumulate multiple framedata into a batch
                if batch_pednum > self.args.batch_around_ped:
                    # enough people in the scene
                    batch_data.append(traject_batch)
                    Batch_id.append(batch_id)

                    batch_data = self.massup_batch(batch_data)
                    batch_data_mass.append((batch_data, Batch_id,))

                    last_frame = i
                    batch_data = []
                    Batch_id = []
                else:
                    batch_data.append(traject_batch)
                    Batch_id.append(batch_id)

        if last_frame < data_index.shape[1] - 1 and setname == 'test' and batch_pednum > 1:
            batch_data = self.massup_batch(batch_data)
            batch_data_mass.append((batch_data, Batch_id,))
        self.args.batch_around_ped = temp
        return batch_data_mass

    def find_trajectory_fragment(self, trajectory, startframe, seq_length, skip):
        '''
        Query the trajectory fragment based on the startframe. Replace where data isn't exsist with 0.
        Return an array with shape (seq_length, 4) (frame number, x, y, Z) => list of 20 trajectory fragments from startframe
        it returns return_trajec, iffull, and ifexsitobs. These could be variables that hold a trajectory and two boolean flags indicating whether the trajectory is full and whether there is an existing observation, respectively
        '''
        return_trajec = np.zeros((seq_length, 4))
        endframe = startframe + (seq_length) * skip
        #get the start trajectory (frame number, x, y)
        start_n = np.where(trajectory[:, 0] == startframe)
        #get the end trajectory (frame number, x, y)
        end_n = np.where(trajectory[:, 0] == endframe)

        iffull = False
        ifexsitobs = False

        if start_n[0].shape[0] == 0 and end_n[0].shape[0] != 0:# if start_n[0] has no elements and end_n[0] has at least one element
            start_n = 0
            end_n = end_n[0][0]
            if end_n == 0:
                return return_trajec, iffull, ifexsitobs

        elif end_n[0].shape[0] == 0 and start_n[0].shape[0] != 0:
            start_n = start_n[0][0]
            end_n = trajectory.shape[0]

        elif end_n[0].shape[0] == 0 and start_n[0].shape[0] == 0:
            start_n = 0
            end_n = trajectory.shape[0]

        else:
            end_n = end_n[0][0]
            start_n = start_n[0][0]

        if (end_n - start_n) != seq_length:  # resolve the error
            return return_trajec, iffull, ifexsitobs
        
        #extract a subset of the trajectory array from index start_n to end_n.
        candidate_seq = trajectory[start_n:end_n]
        offset_start = int((candidate_seq[0, 0] - startframe) // skip) # clip range to resolve the error
        offset_end = self.args.seq_length + int((candidate_seq[-1, 0] - endframe) // skip)
        
        try:  # resolve the valuerror
            return_trajec[offset_start:offset_end + 1, :4] = candidate_seq #replace a slice of the return_trajec array with the candidate_seq. The slice of return_trajec that is being replaced starts at offset_start and ends at offset_end + 1. The :3 in the indexing operation indicates that only the first three columns of return_trajec are being replaced. This might be because return_trajec has more columns than candidate_seq, and only the first three columns are relevant for the current operation.
        except ValueError:
            return return_trajec, iffull, ifexsitobs

        if return_trajec[self.args.obs_length - 1, 1] != 0:
            ifexsitobs = True

        if offset_end - offset_start >= seq_length - 1:
            iffull = True

        return return_trajec, iffull, ifexsitobs

    def massup_batch(self, batch_data):
        '''
        Massed up data fragements in different time window together to a batch
        '''
        num_Peds = 0
        for batch in batch_data:
            num_Peds += batch.shape[1]

        seq_list_b = np.zeros((self.args.seq_length, 0))
        nodes_batch_b = np.zeros((self.args.seq_length, 0, 3))
        nei_list_b = np.zeros((self.args.seq_length, num_Peds, num_Peds))
        #nei_weight_b = np.zeros((self.args.seq_length, num_Peds, num_Peds))
        nei_num_b = np.zeros((self.args.seq_length, num_Peds))
        num_Ped_h = 0
        batch_pednum = []
        for batch in batch_data:
            num_Ped = batch.shape[1]
            if self.args.dataset == 'eth5':
                seq_list, nei_list, nei_num = self.eth_get_social_inputs_numpy(batch)
                #nei_weight = self.eth_get_social_impact_weigh(batch)
            elif self.args.dataset == 'kagc':
                seq_list, nei_list, nei_num = self.kagc_get_social_inputs_numpy(batch)
                #nei_weight = self.kagc_get_social_impact_weigh(batch)
            else:
                seq_list, nei_list, nei_num = self.get_social_inputs_numpy(batch)
                #nei_weight = self.get_social_inputs_weights2(batch)
           
            nodes_batch_b = np.append(nodes_batch_b, batch, 1)
            seq_list_b = np.append(seq_list_b, seq_list, 1)
            nei_list_b[:, num_Ped_h:num_Ped_h + num_Ped, num_Ped_h:num_Ped_h + num_Ped] = nei_list
            #nei_weight_b[:, num_Ped_h:num_Ped_h + num_Ped, num_Ped_h:num_Ped_h + num_Ped] = nei_weight
            nei_num_b[:, num_Ped_h:num_Ped_h + num_Ped] = nei_num
            batch_pednum.append(num_Ped)
            num_Ped_h += num_Ped
        return (nodes_batch_b, seq_list_b, nei_list_b, nei_num_b, batch_pednum)

    def get_social_inputs_numpy(self, inputnodes):
        '''
        Get the sequence list (denoting where data exist) and neighboring list (denoting where neighbors exist).

        Parameters:
            inputnodes (numpy.ndarray): The input nodes representing the trajectories of pedestrians.

        Returns:
            traj_list (numpy.ndarray): The sequence list indicating the existence of data for each pedestrian at each time frame.
            nei_list (numpy.ndarray): The neighboring list indicating the neighbors of each pedestrian at each time frame.
            nei_num (numpy.ndarray): The number of neighbors for each pedestrian at each time frame.
        '''
        num_Peds = inputnodes.shape[1]

        traj_list = np.zeros((inputnodes.shape[0], num_Peds))  # which ped exist in the current timeframe, a 0-1 binary array

        # denote where data not missing
        for pedi in range(num_Peds):
            traj = inputnodes[:, pedi]  # trajectory for each ped
            traj_list[traj[:, 0] != 0, pedi] = 1  # exist location in ped

        # get relative cords, neighbor id list
        nei_list = np.zeros((inputnodes.shape[0], num_Peds, num_Peds))
        nei_num = np.zeros((inputnodes.shape[0], num_Peds))

        # nei_list[f,i,j] denote if j is i's neighbors in timeframe f
        for pedi in range(num_Peds):
            nei_list[:, pedi, :] = traj_list
            nei_list[:, pedi, pedi] = 0  # person i is not the neighbor of itself
            nei_num[:, pedi] = np.sum(nei_list[:, pedi, :], 1)
            traj_i = inputnodes[:, pedi]
            for pedj in range(num_Peds):
                traj_j = inputnodes[:, pedj]

                # if altitude (distance coversion) in 3d case
                # if traj_i[select, -1] - traj_i[select, -1] < 1000:

                # else:
                #select is a boolean array that is true if both traj_i and traj_j have data at the current time frame
                commonFrames = (traj_list[:, pedi] > 0) & (traj_list[:, pedj] > 0)

                # calculate the haversine distance between two trajectory sequences
                #it returns a numpy array with the same shape as traj_i and traj_j which contains the haversine distance between the two sequences for each time frame
                hdist = vectorized_haversine_dist(traj_i[commonFrames, 1], traj_i[commonFrames, 0], traj_j[commonFrames, 1], traj_j[commonFrames, 0])
                select_tooFar = (hdist > self.args.neighbor_thred)

                #relative_cord = traj_i[select, :] - traj_j[select, :]  # relative coords

                # invalid data index
                # change threshold here
                # change to L2 norm
                #select_tooFar = np.sqrt(relative_cord[:, 0]**2 + relative_cord[:, 1]**2) > self.args.neighbor_thred
                #select_tooFar = (abs(relative_cord[:, 0]) > self.args.neighbor_thred) | (abs(relative_cord[:, 1]) > self.args.neighbor_thred)

                #remove far away peds from neighbors
                nei_num[commonFrames, pedi] -= select_tooFar

                commonFrames[commonFrames==True] = select_tooFar
                nei_list[commonFrames, pedi, pedj] = 0

        return traj_list, nei_list, nei_num
    def get_social_inputs_weights(self, inputnodes):
        '''
        Get the sequence list (denoting where data exist) and neighboring list (denoting where neighbors exist).

        Parameters:
            inputnodes (numpy.ndarray): The input nodes representing the trajectories of pedestrians.

        Returns:
            traj_list (numpy.ndarray): The sequence list indicating the existence of data for each pedestrian at each time frame.
            nei_list (numpy.ndarray): The neighboring list indicating the neighbors of each pedestrian at each time frame.
            nei_num (numpy.ndarray): The number of neighbors for each pedestrian at each time frame.
        '''
        num_Peds = inputnodes.shape[1]

        traj_list = np.zeros((inputnodes.shape[0], num_Peds))  # which ped exist in the current timeframe, a 0-1 binary array

        # For each pedestrian in the scene set traj_list to 1 if the pedestrian has data at the current time frame
        for pedi in range(num_Peds):
            traj = inputnodes[:, pedi]  # trajectory for each ped
            traj_list[traj[:, 0] != 0, pedi] = 1  # exist location in ped

        # get relative cords, neighbor id list
        nei_list = np.zeros((inputnodes.shape[0], num_Peds, num_Peds))
        #set the number of neighbors to 0 for all pedestrians at all time frames
        nei_num = np.zeros((inputnodes.shape[0], num_Peds))

        # nei_list[f,i,j] denote if j is i's neighbors in timeframe f
        for pedi in range(num_Peds):
            nei_list[:, pedi, :] = traj_list
            nei_list[:, pedi, pedi] = 0  # person i is not the neighbor of itself
            nei_num[:, pedi] = np.sum(nei_list[:, pedi, :], 1)
            main_trajectory = inputnodes[:, pedi]
            main_trajectory_with_bearing = calculate_bearing_of_full_trajectory(main_trajectory)
            for pedj in range(num_Peds):
                if(pedi == pedj):
                    continue
                
                neibour_trajectory = inputnodes[:, pedj]
                
                neibour_trajectory_with_bearing = calculate_bearing_of_full_trajectory(neibour_trajectory)
                #select is a boolean array that is true if both main_trajectory and neibour_trajectory have data at the current time frame
                commonFrames = (traj_list[:, pedi] > 0) & (traj_list[:, pedj] > 0)

                result_neibour_impact = calculate_bearing_with_neibour_trajectories(main_trajectory_with_bearing[commonFrames], neibour_trajectory_with_bearing[commonFrames])

                select_tooFar = (result_neibour_impact[:,6] == 0)
                select_near = (result_neibour_impact[:,6] > 0)
                #remove far away peds from neighbors
                nei_num[commonFrames, pedi] -= select_tooFar

                commonFrames[commonFrames==True] = select_tooFar
                nei_list[commonFrames, pedi, pedj] = 0

                commonFrames = (traj_list[:, pedi] > 0) & (traj_list[:, pedj] > 0)
                commonFrames[commonFrames==True] = select_near
                nei_list[commonFrames, pedi, pedj] = result_neibour_impact[commonFrames,6]

        #return traj_list, nei_list, nei_num
        return nei_list
    def get_social_inputs_weights2(self, inputnodes):
        '''
        Get the sequence list (denoting where data exsist) and neighboring list (denoting where neighbors exsist).
        '''
        num_Peds = inputnodes.shape[1]

        seq_list = np.zeros((inputnodes.shape[0], num_Peds))
        # denote where data not missing

        for pedi in range(num_Peds):
            seq = inputnodes[:, pedi]
            seq_list[seq[:, 0] != 0, pedi] = 1

        # get relative cords, neighbor id list
        nei_list = np.zeros((inputnodes.shape[0], num_Peds, num_Peds))
      
        # nei_list[f,i,j] denote if j is i's neighbors in frame f
        for pedi in range(num_Peds):
            
            nei_list[:, pedi, pedi] = 0  # person i is not the neighbor of itself
            
            seqi = inputnodes[:, pedi]
            for pedj in range(num_Peds):
                if pedi == pedj:
                    continue
                seqj = inputnodes[:, pedj]
               
                distance_weight = np.zeros(seqi.shape[0])
                for i in range(seqi.shape[0]):
                    lon1, lat1, level1 = seqi[i][:3]
                    lon2, lat2, level2 = seqj[i][:3]
                    vertical_seperation = abs(level1 - level2)
                    distance = vectorized_haversine_dist(lat1, lon1, lat2, lon2)
                    if distance > self.args.neighbor_thred or vertical_seperation > 0.4:
                        distance_weight[i] = 0
                    else:
                        distance_weight[i] = 1 - distance / self.args.neighbor_thred


                nei_list[:, pedi, pedj] = distance_weight
        return nei_list
    def eth_get_social_inputs_numpy(self, inputnodes):
        '''
        Get the sequence list (denoting where data exsist) and neighboring list (denoting where neighbors exsist).
        '''
        num_Peds = inputnodes.shape[1]

        seq_list = np.zeros((inputnodes.shape[0], num_Peds))
        # denote where data not missing

        for pedi in range(num_Peds):
            seq = inputnodes[:, pedi]
            seq_list[seq[:, 0] != 0, pedi] = 1

        # get relative cords, neighbor id list
        nei_list = np.zeros((inputnodes.shape[0], num_Peds, num_Peds))
        nei_num = np.zeros((inputnodes.shape[0], num_Peds))

        # nei_list[f,i,j] denote if j is i's neighbors in frame f
        for pedi in range(num_Peds):
            nei_list[:, pedi, :] = seq_list
            nei_list[:, pedi, pedi] = 0  # person i is not the neighbor of itself
            nei_num[:, pedi] = np.sum(nei_list[:, pedi, :], 1)
            seqi = inputnodes[:, pedi]
            for pedj in range(num_Peds):
                if pedi == pedj:
                    continue
                seqj = inputnodes[:, pedj]
                select = (seq_list[:, pedi] > 0) & (seq_list[:, pedj] > 0)

                relative_cord = seqi[select, :2] - seqj[select, :2]

                # invalid data index
                select_dist = (abs(relative_cord[:, 0]) > self.args.neighbor_thred) | (
                        abs(relative_cord[:, 1]) > self.args.neighbor_thred)

                nei_num[select, pedi] -= select_dist

                select[select == True] = select_dist
                nei_list[select, pedi, pedj] = 0
        return seq_list, nei_list, nei_num
    
    def eth_get_social_impact_weigh(self, inputnodes):
        '''
        Get the sequence list (denoting where data exsist) and neighboring list (denoting where neighbors exsist).
        '''
        num_Peds = inputnodes.shape[1]

        seq_list = np.zeros((inputnodes.shape[0], num_Peds))
        # denote where data not missing

        for pedi in range(num_Peds):
            seq = inputnodes[:, pedi]
            seq_list[seq[:, 0] != 0, pedi] = 1

        # get relative cords, neighbor id list
        nei_list = np.zeros((inputnodes.shape[0], num_Peds, num_Peds))
      
        # nei_list[f,i,j] denote if j is i's neighbors in frame f
        for pedi in range(num_Peds):
            
            nei_list[:, pedi, pedi] = 0  # person i is not the neighbor of itself
            
            seqi = inputnodes[:, pedi]
            for pedj in range(num_Peds):
                if pedi == pedj:
                    continue
                seqj = inputnodes[:, pedj]
                select = (seq_list[:, pedi] > 0) & (seq_list[:, pedj] > 0)

                relative_cord = seqi[select, :2] - seqj[select, :2]
               
                distance_weight = np.sqrt((abs(relative_cord[:, 0]))**2 + (abs(relative_cord[:, 1]))**2)
                distance_weight[distance_weight > 5] = 5
                distance_weight = 1 - distance_weight / 5
                nei_list[select, pedi, pedj] = distance_weight
        return nei_list
    def kagc_get_social_inputs_numpy(self, inputnodes):
        '''
        Get the sequence list (denoting where data exsist) and neighboring list (denoting where neighbors exsist).
        '''
        num_Peds = inputnodes.shape[1]

        seq_list = np.zeros((inputnodes.shape[0], num_Peds))
        # denote where data not missing

        for pedi in range(num_Peds):
            seq = inputnodes[:, pedi]
            seq_list[seq[:, 0] != 0, pedi] = 1

        # get relative cords, neighbor id list
        nei_list = np.zeros((inputnodes.shape[0], num_Peds, num_Peds))
        nei_num = np.zeros((inputnodes.shape[0], num_Peds))

        # nei_list[f,i,j] denote if j is i's neighbors in frame f
        for pedi in range(num_Peds):
            nei_list[:, pedi, :] = seq_list
            nei_list[:, pedi, pedi] = 0  # person i is not the neighbor of itself
            nei_num[:, pedi] = np.sum(nei_list[:, pedi, :], 1)
            seqi = inputnodes[:, pedi]
            for pedj in range(num_Peds):
                if pedi == pedj:
                    continue
                seqj = inputnodes[:, pedj]
                select = (seq_list[:, pedi] > 0) & (seq_list[:, pedj] > 0)

                relative_cord = seqi[select, :3] - seqj[select, :3]

                # invalid data index
                select_dist = (abs(relative_cord[:, 0]) > self.args.neighbor_thred) | (
                        abs(relative_cord[:, 1]) > self.args.neighbor_thred)
                

                nei_num[select, pedi] -= select_dist

                select[select == True] = select_dist
                nei_list[select, pedi, pedj] = 0
        return seq_list, nei_list, nei_num
    
    def kagc_get_social_impact_weigh(self, inputnodes):
        '''
        Get the sequence list (denoting where data exsist) and neighboring list (denoting where neighbors exsist).
        '''
        num_Peds = inputnodes.shape[1]

        seq_list = np.zeros((inputnodes.shape[0], num_Peds))
        # denote where data not missing

        for pedi in range(num_Peds):
            seq = inputnodes[:, pedi]
            seq_list[seq[:, 0] != 0, pedi] = 1

        # get relative cords, neighbor id list
        nei_list = np.zeros((inputnodes.shape[0], num_Peds, num_Peds))
      
        # nei_list[f,i,j] denote if j is i's neighbors in frame f
        for pedi in range(num_Peds):
            
            nei_list[:, pedi, pedi] = 0  # person i is not the neighbor of itself
            
            seqi = inputnodes[:, pedi]
            for pedj in range(num_Peds):
                if pedi == pedj:
                    continue
                seqj = inputnodes[:, pedj]
                select = (seq_list[:, pedi] > 0) & (seq_list[:, pedj] > 0)

                relative_cord = seqi[select, :3] - seqj[select, :3]
               
                distance_weight = np.sqrt((abs(relative_cord[:, 0]))**2 + (abs(relative_cord[:, 1]))**2)
                distance_weight[distance_weight > self.args.neighbor_thred] = self.args.neighbor_thred
                distance_weight[(abs(relative_cord[:,2])) > 0.3] = self.args.neighbor_thred#if they are far vertically
                distance_weight = 1 - distance_weight / self.args.neighbor_thred
                nei_list[select, pedi, pedj] = distance_weight
        return nei_list

    def rotate_shift_batch(self, batch_data, ifrotate=True):
        '''
        Random ration and zero shifting.
        '''
        batch, seq_list, nei_list, nei_num, batch_pednum = batch_data

        # rotate batch
        if ifrotate:
            th = random.random() * np.pi
            cur_ori = batch.copy()
            batch[:, :, 0] = cur_ori[:, :, 0] * np.cos(th) - cur_ori[:, :, 1] * np.sin(th)
            batch[:, :, 1] = cur_ori[:, :, 0] * np.sin(th) + cur_ori[:, :, 1] * np.cos(th)

        # get shift value
        s = batch[self.args.obs_length - 1]

        shift_value = np.repeat(s.reshape((1, -1, 3)), self.args.seq_length, 0)

        batch_data = batch, batch - shift_value, shift_value, seq_list, nei_list, nei_num, batch_pednum
        return batch_data

    def get_train_batch(self, idx):
        batch_data, batch_id = self.trainbatch[idx]
        batch_data = self.rotate_shift_batch(batch_data, ifrotate=self.args.randomRotate)
        return batch_data, batch_id

    def get_test_batch(self, idx):
        batch_data, batch_id = self.testbatch[idx]
        batch_data = self.rotate_shift_batch(batch_data, ifrotate=False)
        return batch_data, batch_id

    def reset_batch_pointer(self, set, valid=False):
        '''
        Reset all pointers
        '''
        if set == 'train':
            if not valid:
                self.frame_pointer = 0
            else:
                self.val_frame_pointer = 0
        else:
            self.test_frame_pointer = 0

def getLossMask(outputs, node_first, seq_list, using_cuda=False):
    '''
    Get a mask to denote whether both of current and previous data exist.
    Note: It is not supposed to calculate loss for a person at time t if his data at t-1 does not exist.
    '''

    if outputs.dim() == 3:
        seq_length = outputs.shape[0]
    else:
        seq_length = outputs.shape[1]

    node_pre = node_first
    lossmask = torch.zeros(seq_length, seq_list.shape[1])

    if using_cuda:
        lossmask = lossmask.cuda()

    # For loss mask, only generate for those exist through the whole window
    for framenum in range(seq_length):
        if framenum == 0:
            lossmask[framenum] = seq_list[framenum] * node_pre
        else:
            lossmask[framenum] = seq_list[framenum] * lossmask[framenum - 1]

    return lossmask, sum(sum(lossmask))

def L2forTest(outputs, targets, obs_length, lossMask):
    '''
    Evaluation.
    '''
    seq_length = outputs.shape[0]
    error = torch.norm(outputs - targets, p=2, dim=2)
    # only calculate the pedestrian presents fully presented in the time window
    pedi_full = torch.sum(lossMask, dim=0) == seq_length
    error_full = error[obs_length - 1:, pedi_full]
    error = torch.sum(error_full)
    error_cnt = error_full.numel()
    final_error = torch.sum(error_full[-1])
    final_error_cnt = error_full[-1].numel()

    return error.item(), error_cnt, final_error.item(), final_error_cnt, error_full

def L2forTestS(outputs, targets, obs_length, lossMask, num_samples=20):
    '''
    Evaluation, stochastic version
    test
    '''
    seq_length = outputs.shape[1]
    # Calculate L2 errors between predicted outputs and target values
    error = torch.norm(outputs - targets, p=2, dim=3)

    # only calculate the pedestrian presents fully presented in the time window
    pedi_full = torch.sum(lossMask, dim=0) == seq_length
    error_full = error[:, obs_length - 1:, pedi_full]  # first dimension is MC test
    # Sum errors over time steps
    error_full_sum = torch.sum(error_full, dim=1)
    # Find the index of the minimum error for each pedestrian
    error_full_sum_min, min_index = torch.min(error_full_sum, dim=0)

    # Extract the best errors for each pedestrian
    best_error = []
    for index, value in enumerate(min_index):
        best_error.append(error_full[value, :, index])
    best_error = torch.stack(best_error)
    best_error = best_error.permute(1, 0)
    # Calculate total error and error count
    error = torch.sum(error_full_sum_min)
    error_cnt = error_full.numel() / num_samples
     # Calculate final error and final error count
    final_error = torch.sum(best_error[-1])
    final_error_cnt = error_full.shape[-1]

    return error.item(), error_cnt, final_error.item(), final_error_cnt

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('Function', method.__name__, 'time:', round((te - ts) * 1000, 1), 'ms')
        print()
        return result

    return timed

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def vectorized_haversine_dist(s_lat, s_lng, e_lat, e_lng):
    # approximate radius of earth in km
    # vectorized version to speed up the neighbor search process
    R = 6373.0

    s_lat = s_lat * np.pi / 180.0
    s_lng = np.deg2rad(s_lng)
    e_lat = np.deg2rad(e_lat)
    e_lng = np.deg2rad(e_lng)

    d = np.sin((e_lat - s_lat) / 2) ** 2 + np.cos(s_lat) * np.cos(e_lat) * np.sin((e_lng - s_lng) / 2) ** 2

    return 2 * R * np.arcsin(np.sqrt(d))

def vincenty_distance(lat1_list, lon1_list, lat2_list, lon2_list):
    distances = []
    for lat1, lon1, lat2, lon2 in zip(lat1_list, lon1_list, lat2_list, lon2_list):
        point1 = (lat1, lon1)
        point2 = (lat2, lon2)
        dis = geodesic(point1, point2).meters
        if dis > 0.0 and dis < 50.0:
            print('short distance')

        distances.append(dis)
    
    return np.array(distances)
import numpy as np

def calculate_bearing_of_full_trajectory(trajectory):
    # Ensure the input trajectory has at least two points
    if trajectory.shape[0] < 2 or trajectory.shape[1] != 3:
        raise ValueError("Invalid trajectory shape. It should be a 2D array with at least two rows and two columns (lat, lon).")

    # Initialize an array to store the bearings
    bearings = np.zeros(trajectory.shape[0])
    #check the trajectory values. if the values are zero, then replace them with the last valid non-zero value
    non_zero_indices = np.nonzero(trajectory)[0]
    zero_indices = np.where(trajectory == 0)[0]
    for i in zero_indices:
        prev_non_zero_idx = non_zero_indices[non_zero_indices < i][-1]
        trajectory[i] = trajectory[prev_non_zero_idx]
        
    last_nondegrees_bearing = 0
    # Iterate over the trajectory to calculate bearings
    for i in range(1, trajectory.shape[0]):
        # Calculate the bearing between the current point and the previous point
        lon1, lat1, level1 = trajectory[i - 1]
        lon2, lat2, level2 = trajectory[i]
        #check if lat1 and lat2 are equal and lon1 and lon2 are equal. if yes, then set bearing to last valid bearing.
        if lat1 == lat2 and lon1 == lon2:
            bearing = last_nondegrees_bearing
        else:
            bearing = np.arctan2(np.sin(np.radians(lon2 - lon1)) * np.cos(np.radians(lat2)),
                      np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) -
                      np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(lon2 - lon1)))

        # Convert bearing from radians to degrees
        last_nondegrees_bearing = bearing
        bearing = np.degrees(bearing)

        # Ensure the bearing is in the range [0, 360)
        bearing = (bearing + 360) % 360

        # Store the calculated bearing
        bearings[i] = bearing
    
    bearings[0] = bearings[1]
    #append the bearings to the trajectory
    trajectory = np.append(trajectory, bearings.reshape(-1,1), axis=1)
    return trajectory

def calculate_bearing_between_two_points(lat1, lon1, lat2, lon2):
    bearing = np.arctan2(np.sin(np.radians(lon2 - lon1)) * np.cos(np.radians(lat2)),
                      np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) -
                      np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(lon2 - lon1)))

    # Convert the bearing from radians to degrees
    bearing = np.degrees(bearing)

    # Ensure the bearing is in the range [0, 360)
    bearing = (bearing + 360) % 360

    return bearing

def calculate_bearing_with_neibour_trajectories(traj1, traj2):
    
    #check the trajectory values. if the values are zero, then replace them with the last valid non-zero value
    non_zero_indices = np.nonzero(traj1[:,:2])[0]
    zero_indices = np.where(traj1[:,:2] == 0)[0]
    for i in zero_indices:
        prev_non_zero_idx = non_zero_indices[non_zero_indices < i][-1]
        traj1[i,:2] = traj1[prev_non_zero_idx,:2]

    #check the trajectory values. if the values are zero, then replace them with the last valid non-zero value
    non_zero_indices = np.nonzero(traj2[:,:2])[0]
    zero_indices = np.where(traj2[:,:2] == 0)[0]
    for i in zero_indices:
        prev_non_zero_idx = non_zero_indices[non_zero_indices < i][-1]
        traj2[i,:2] = traj2[prev_non_zero_idx,:2]

    # Initialize an array to store the bearings
    bearings = np.zeros(traj1.shape[0])
    distances_impact = np.zeros(traj1.shape[0])
    bearing_impacts = np.zeros(traj1.shape[0])
    total_impacts = np.zeros(traj1.shape[0])
    for i in range(traj1.shape[0]):
        # Calculate the bearing between the current point and the previous point
        lon1, lat1, level1 = traj1[i][:3]
        lon2, lat2, level2 = traj2[i][:3]
        vertical_seperation = abs(level1 - level2)
        distance = vectorized_haversine_dist(lat1, lon1, lat2, lon2)

        if vertical_seperation > 0.01 or distance > 10 :
            total_impacts[i] = 0
        elif distance < 10 :
            distances_impact[i] = 1 - (distance / 10)
        elif(distance < 1): 
            total_impacts[i] = 1     

        #bearing = calculate_bearing_between_two_points(lat1, lon1, lat2, lon2)
        #bearing = traj2[i][2]       
        #diff = abs(traj1[i][2] - bearing)    
        #bearings[i] = diff
#
        #if diff < 90:
        #    bearing_impacts[i] = 1 - (diff/90)      
        #bearing_impacts[i] = moving_towards_each_other_factor(lat1, lon1, traj1[i][2], lat2, lon2, traj2[i][2])
        
        #if distance > 10:
        #    total_impacts[i] = 0
        #elif(distance < 1):#if distance is less than 1 km, then the bearing is not important  
        #    total_impacts[i] = 1            
        #else:
        #    total_impacts[i] = (bearing_impacts[i] + distances_impact[i]) / 2 # mean value
        
    traj1 = np.append(traj1, bearings.reshape(-1,1), axis=1)  
    traj1 = np.append(traj1, bearing_impacts.reshape(-1,1), axis=1)
    traj1 = np.append(traj1, distances_impact.reshape(-1,1), axis=1)
    traj1 = np.append(traj1, total_impacts.reshape(-1,1), axis=1)

    return traj1


def moving_towards_each_other_factor(lat1, lon1, heading1, lat2, lon2, heading2):
    # Calculate the bearing from object 1 to object 2 and vice versa
    bearing_1_to_2 = calculate_bearing_between_two_points(lat1, lon1, lat2, lon2)
    bearing_2_to_1 = calculate_bearing_between_two_points(lat2, lon2, lat1, lon1)
    
    # Calculate the relative bearings
    relative_bearing_1 = (heading1 - bearing_1_to_2 + 360) % 360
    relative_bearing_2 = (heading2 - bearing_2_to_1 + 360) % 360
    
    # If the relative bearings are within certain thresholds, they are moving towards each other
    threshold = 60  # degrees
    
    if relative_bearing_1 <= threshold or relative_bearing_1 >= 360 - threshold:
        if relative_bearing_2 <= threshold or relative_bearing_2 >= 360 - threshold:
            return 1.0
    
    return 0.0