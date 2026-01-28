from torch.utils.data import IterableDataset
import os, numpy as np
import os.path as osp
import h5py
from torch_geometric.data import Data
import torch
import math
import time


class FPCBase:
    def __init__(
        self, max_epochs=1, files=None, tra_len=600, time_iterval=0.01, num_steps=1
    ):

        self.num_steps = num_steps

        self.open_tra_num = 1
        self.file_handle=files
        self.shuffle_file()

        self.data_keys = (
            "pos",
            "edges",            
            "node_type",
            "pres",
            "rho",
            "v1",
            "v2",
            "Mach",            
        )
        self.out_keys = list(self.data_keys)  + ['time']

        self.tra_index = 0
        self.epcho_num=1
        self.tra_readed_index = -1

        # dataset attr
        self.tra_len = tra_len
        self.time_iterval = time_iterval

        self.opened_tra = []
        self.opened_tra_readed_index = {}
        self.opened_tra_readed_random_index = {}
        self.tra_data = {}
        self.max_epochs = max_epochs

    def open_tra(self):
        while(len(self.opened_tra) < self.open_tra_num):

            tra_index = self.datasets[self.tra_index]

            if tra_index not in self.opened_tra:
                self.opened_tra.append(tra_index)
                self.opened_tra_readed_index[tra_index] = -1
                # Adjusted to account for num_steps
                self.opened_tra_readed_random_index[tra_index] = np.random.permutation(
                    self.tra_len - self.num_steps
                )
            self.tra_index += 1

            if self.check_if_epcho_end():
                self.epcho_end()
                print('Epcho Finished')

    def check_and_close_tra(self):
        to_del = []
        for tra in self.opened_tra:
            if self.opened_tra_readed_index[tra] >= (self.tra_len - self.num_steps - 1):
                to_del.append(tra)
        for tra in to_del:
            self.opened_tra.remove(tra)
            try:
                del self.opened_tra_readed_index[tra]
                del self.opened_tra_readed_random_index[tra]
                del self.tra_data[tra]
            except Exception as e:
                print(e)

    def shuffle_file(self):
        datasets = list(self.file_handle.keys())
        np.random.shuffle(datasets)
        self.datasets = datasets

    def epcho_end(self):
        print("Epoch %d finished" % self.epcho_num)
        self.tra_index = 0
        self.shuffle_file()
        self.epcho_num = self.epcho_num + 1
        print("Epoch %d started" % self.epcho_num)

    def check_if_epcho_end(self):
        if self.tra_index >= len(self.file_handle):
            return True
        return False

    @staticmethod
    def datas_to_graph(datas):
        # ("pos", "edges", "node_type", "pres", "rho", "v1", "v2", "Mach")

        node_attr = np.hstack(
            (datas[2], datas[4][0], datas[5][0], datas[6][0], datas[3][0], datas[7])
        )
        "node_type, rho, v1, v2, pres, Mach"

        crds = torch.as_tensor(datas[0], dtype=torch.float)

        target = np.hstack((datas[4][1], datas[5][1], datas[6][1], datas[3][1]))

        node_attr = torch.as_tensor(node_attr, dtype=torch.float32)  # (num_nodes, 6)
        target    = torch.as_tensor(target,    dtype=torch.float32)  # (num_nodes, 4)
        edges_np  = datas[1]
        edges     = torch.as_tensor(edges_np.T, dtype=torch.long)    # (2, num_edges)

        g = Data(x=node_attr, edge_index=edges, y=target, pos=crds)

        #--------------------------------------------------------------------------
        # Prepare future Primitive variables (from t+1 to t+num_steps)
        # shape (num_steps+1, num_nodes, 1)
        pres_arr = datas[3]
        rho_arr  = datas[4]
        v1_arr   = datas[5]
        v2_arr   = datas[6]

        # Number of future steps = pres_arr.shape[0] - 1
        num_steps   = pres_arr.shape[0] - 1
        num_nodes   = pres_arr.shape[1]

        # Collect [rho, v1, v2, pres] at each future time i = 1..num_steps
        future_list = []
        for i in range(1, pres_arr.shape[0]):
            # (num_nodes, 1)
            step_r   = rho_arr[i]
            step_v1  = v1_arr[i]
            step_v2  = v2_arr[i]
            step_pres= pres_arr[i]

            # Concatenate along last axis → (num_nodes, 4)
            step_feats = np.concatenate(
                [step_r, step_v1, step_v2, step_pres],
                axis=1
            )
            future_list.append(step_feats)

        # Stack them into shape (num_steps, num_nodes, 4)
        future_stack = np.stack(future_list, axis=0)

        # Now reorder to (num_nodes, num_steps, 4)
        future_stack = np.transpose(future_stack, (1, 0, 2))

        # Finally reshape to (num_nodes, 4 * num_steps)
        future_prims_np = future_stack.reshape(num_nodes, 4 * num_steps)

        # Convert to torch.Tensor:
        # shape (num_nodes, 4 * num_steps)
        future_prims = torch.from_numpy(future_prims_np.astype(np.float32))
        g.future_prims = future_prims
        #--------------------------------------------------------------------------

        return g

    def __next__(self):

        self.check_and_close_tra()
        self.open_tra()

        if self.epcho_num >self.max_epochs:
            raise StopIteration

        selected_tra = np.random.choice(self.opened_tra)

        data = self.tra_data.get(selected_tra, None)
        if data is None:
            data = self.file_handle[selected_tra]
            self.tra_data[selected_tra] = data

        selected_tra_readed_index = self.opened_tra_readed_index[selected_tra]
        selected_frame = self.opened_tra_readed_random_index[selected_tra][selected_tra_readed_index+1]
        self.opened_tra_readed_index[selected_tra] += 1

        datas = []
        for k in self.data_keys:
            if k in ["pres", "rho", "v1", "v2"]:
                # Retrieve Primitive variables at t, t+1, ..., t+num_steps
                r = np.array(
                    [data[k][selected_frame + i] for i in range(self.num_steps + 1)],
                    dtype=np.float32,
                )
            else:
                r = data[k][selected_frame]
                if k in ["edges", "node_type"]:
                    r = r.astype(np.int32)
            datas.append(r)
        datas.append(np.array([self.time_iterval * selected_frame], dtype=np.float32))
        # ("pos", "edges", "node_type", "pres", "rho", "v1", "v2", "Mach")
        g = self.datas_to_graph(datas)

        return g

    def __iter__(self):
        return self


class FPC(IterableDataset):

    def __init__(
        self,
        max_epochs,
        dataset_dir,
        split="train",
        tra_len=600,
        time_iterval=0.01,
        num_steps=1,
    ):

        super().__init__()

        self.num_steps = num_steps

        # dataset_dir = osp.join(dataset_dir, split+'.h5')
        # dataset_dir = osp.join('cGnnCodes','dataset', 'data_downsampled', split + '.h5')
        dataset_dir = osp.join('dataset', 'data_downsampled', split + '.h5')
        self.max_epochs= max_epochs

        self.tra_len = tra_len
        self.time_iterval = time_iterval

        self.dataset_dir = dataset_dir
        assert os.path.isfile(dataset_dir), '%s not exist' % dataset_dir
        self.file_handle = h5py.File(dataset_dir, "r")
        print('Dataset '+  self.dataset_dir + ' Initilized')

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self.file_handle)
        else:
            per_worker = int(math.ceil(len(self.file_handle)/float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.file_handle))

        keys = list(self.file_handle.keys())
        keys = keys[iter_start:iter_end]
        files = {k: self.file_handle[k] for k in keys}
        return FPCBase(
            max_epochs=self.max_epochs,
            files=files,
            tra_len=self.tra_len,
            time_iterval=self.time_iterval,
            num_steps=self.num_steps,
        )


class FPC_ROLLOUT(IterableDataset):

    def __init__(
        self, dataset_dir, split="test", name="2D Euler", time_iterval=0.01
    ):

        # dataset_dir = osp.join(dataset_dir, split+'.h5')
        # dataset_dir = osp.join('cGnnCodes','dataset', 'data_downsampled', split + '.h5')
        dataset_dir = osp.join('dataset', 'data_downsampled', split + '.h5')
        self.dataset_dir = dataset_dir
        assert os.path.isfile(dataset_dir), '%s not exist' % dataset_dir
        self.file_handle = h5py.File(dataset_dir, "r")
        self.data_keys = (
            "pos",
            "edges",            
            "node_type",
            "pres",
            "rho",
            "v1",
            "v2",
            "Mach",            
        )
        self.time_iterval = time_iterval
        self.load_dataset()

    def load_dataset(self):
        datasets = list(self.file_handle.keys())
        self.datasets = datasets

    def change_file(self, file_index):
        file_index = self.datasets[file_index]
        self.cur_tra = self.file_handle[file_index]
        self.cur_targecity_length = self.cur_tra['pres'].shape[0]
        self.cur_tragecity_index = 0
        self.edge_index = None

    def __next__(self):
        if self.cur_tragecity_index==(self.cur_targecity_length - 1):
            raise StopIteration

        datas = []
        data = self.cur_tra
        selected_frame = self.cur_tragecity_index

        datas = []
        for k in self.data_keys:
            if k in ["pres", "rho", "v1", "v2"]:
                r = np.array((data[k][selected_frame], data[k][selected_frame+1]), dtype=np.float32)
            else:
                r = data[k][selected_frame]
                if k in ["edges", "node_type"]:
                    r = r.astype(np.int32)
            datas.append(r)
        datas.append(np.array([self.time_iterval * selected_frame], dtype=np.float32))

        self.cur_tragecity_index += 1
        g = FPCBase.datas_to_graph(datas)

        return g

    def __iter__(self):
        return self
