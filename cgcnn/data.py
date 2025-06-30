from __future__ import print_function, division

import csv
import functools
import json
import os
import random
import warnings

import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_val_test_loader(dataset, collate_fn, batch_size=64, train_ratio=None, val_ratio=0.1, 
                                test_ratio=0.1, return_test=False, num_workers=1, pin_memory=False):
    """
    划分数据集为训练/验证/测试集，并创建对应的DataLoader
    参数说明：
    dataset: 完整数据集
    collate_fn: 数据批处理函数
    batch_size: 批大小
    train_ratio/val_ratio/test_ratio: 各分集比例
    return_test: 是否返回测试集
    num_workers/pin_memory: DataLoader参数
    """
    total_size = len(dataset)
    indices = list(range(total_size))
    random.shuffle(indices)  # 内部进行shuffle更安全

    # 计算各分集的实际大小
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = int(test_ratio * total_size)

    # 创建采样器./train/val/test
    train_end = train_size
    val_start = total_size - val_size - test_size
    val_end = val_start + val_size

    train_sampler = SubsetRandomSampler(indices[:train_end])
    val_sampler = SubsetRandomSampler(indices[val_start:val_end])

    # 创建DataLoader
    loader_args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'collate_fn': collate_fn,
        'pin_memory': pin_memory
    }
    train_loader = DataLoader(dataset, sampler=train_sampler, **loader_args)
    val_loader = DataLoader(dataset, sampler=val_sampler, **loader_args)

    if return_test:
        test_sampler = SubsetRandomSampler(indices[val_end:val_end+test_size])
        test_loader = DataLoader(dataset, sampler=test_sampler, **loader_args)
        return train_loader, val_loader, test_loader
    return train_loader, val_loader


def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)  （原子个数，原子特征长度）
      晶体1： [[1.0, 2.0], [3.0, 4.0]]（2 个原子，每个原子有 2 维特征）
      晶体2：[[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]

      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)   （原子个数，原子近邻个数，近邻特征长度）
      晶体1：  [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]（每个原子有 2 个邻居，每个邻居有 2 维特征）
      晶体2：  [[[0.9, 1.0], [1.1, 1.2]], [[1.3, 1.4], [1.5, 1.6]], [[1.7, 1.8], [1.9, 2.0]]] 

      nbr_fea_idx: torch.LongTensor shape (n_i, M)  （原子个数，原子近邻个数）
      晶体1：  [[0, 1], [0, 1]] （邻居索引）
      晶体2：  [[0, 1], [1, 2], [2, 0]]

      target: torch.Tensor shape (1, )
      晶体1：  [5.0]
      晶体2：  [10.0]

      cif_id: str or int
      晶体1： "id1"
      晶体2： "id2"

    Returns
    -------
    N = sum(n_i); N0 = sum(i)
    N是所有晶体原子的总数，N0是晶体总数

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)  （这一批次晶体的的原子总个数，原子长度）
        [[1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0]]

    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)  （这一批次的原子总个数，每个原子的领居个数，领居长度） 邻居特征长度是高斯的中心点个数
      Bond features of each atom's M neighbors

      [[[0.1, 0.2], [0.3, 0.4]],
        [[0.5, 0.6], [0.7, 0.8]],
        [[0.9, 1.0], [1.1, 1.2]],
        [[1.3, 1.4], [1.5, 1.6]],
        [[1.7, 1.8], [1.9, 2.0]]]

    batch_nbr_fea_idx: torch.LongTensor shape (N, M)  （这一批次的原子总个数，邻居个数）
      Indices of M neighbors of each atom

      [[0, 1],
        [1, 0],
        [2, 3],
        [3, 4],
        [4, 2]]

    crystal_atom_idx: list of torch.LongTensor of length N0
      Mapping from the crystal idx to atom idx
    [tensor([0, 1]), tensor([2, 3, 4])] 这是一个堆叠张量

    target: torch.Tensor shape (N0, 1) （晶体数量，1）
      Target value for prediction
      [[5.0],
        [10.0]]

    batch_cif_ids: list
    ["id1", "id2"]
    """
    '''
        初始状态base_index = 0
        晶体1：n_i = 2
        nbr_fea_idx + base_index → [[0, 1], [0, 1]] + 0 → [[0, 1], [0, 1]]
        new_idx = [0, 1]
        base_index += 2 → base_index = 2
        处理晶体 2：
        n_i = 3
        nbr_fea_idx + base_index → [[0, 1], [1, 2], [2, 0]] + 2 → [[2, 3], [3, 4], [4, 2]]
        new_idx = [2, 3, 4]
        base_index += 3 → base_index = 5

        最终结果
        batch_nbr_fea_idx：
        [[0, 1],
        [0, 1],
        [2, 3],
        [3, 4],
        [4, 2]]

        crystal_atom_idx：
        [tensor([0, 1]), tensor([2, 3, 4])]
        '''

    
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target,batch_cif_ids = [], [], []

    base_idx = 0  #记录这一批次的原子总个数

    for i, ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)\
            in enumerate(dataset_list):

        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)

        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)  
        crystal_atom_idx.append(new_idx)
        base_idx += n_i

    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx),\
        torch.stack(batch_target, dim=0),\
        batch_cif_ids


class GaussianDistance():
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """

        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer():
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)  #原子类别
        self._embedding = {}  #字典，存储原子类型到特征向量的映射

    def get_atom_fea(self, atom_type):
        return self._embedding[atom_type]    # 获取原子特征向量

    def load_state_dict(self, state_dict):  # 加载原子嵌入
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())

        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CIFData(Dataset):
    """
    The CIFData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of CIF files. The dataset should have the following
    directory structure:

    root_dir
    ├── id_prop.csv
    ├── atom_init.json
    ├── id0.cif
    ├── id1.cif
    ├── ...

    id_prop.csv: a CSV file with two columns. The first column recodes a
    unique ID for each crystal, and the second column recodes the value of
    target property.

    atom_init.json: a JSON file that stores the initialization vector for each
    element.

    ID.cif: a CIF file that recodes the crystal structure, where ID is the
    unique ID for the crystal.

    Parameters
    ----------

    root_dir: str
        The path to the root directory of the dataset
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    random_seed: int
        Random seed for shuffling the dataset

    Returns
    -------

    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (1, )
    cif_id: str or int
    """
    def __init__(self, root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 random_seed=123):
        self.root_dir ,self.max_num_nbr, self.radius =root_dir, max_num_nbr, radius

        assert os.path.exists(root_dir), 'root_dir does not exist!'

        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]

        random.seed(random_seed)
        random.shuffle(self.id_prop_data)

        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'

        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data[idx]
        crystal = Structure.from_file(os.path.join(self.root_dir, cif_id+'.cif'))

        # 构建原子特征矩阵 (原子数, 原子特征映射长度)
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number) for i in range(len(crystal))])
        atom_fea = torch.Tensor(atom_fea)


        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs] #nbrs中的nbr的第二个值是距离，依据距离排序，近-远

        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(cif_id))

                #取出所有邻居的索引，不足的用0补足
                '''
                nbr的属性，坐标，距离，索引
                nbr = [
                    ((0.1, 0.2, 0.3), 1.2, 0),
                    ((0.4, 0.5, 0.6), 1.5, 1),
                    ((0.7, 0.8, 0.9), 1.8, 2)
                ]
                list(map(lambda x: x[2], nbr)) → [0, 1, 2]
                [0, 1, 2] + [0] * 2 → [0, 1, 2, 0, 0]

                补充邻居距离（nbr_fea）
                我们同样需要构建一个长度为5的距离列表：
                获取当前邻居的距离：list(map(lambda x: x[1], nbr)) → [1.2, 1.5, 1.8]
                设置虚拟距离为 self.radius + 1.（假设 self.radius 为8 → 虚拟距离为9.0）
                补充虚拟距离：[1.2, 1.5, 1.8] + [9.0] * 2 → [1.2, 1.5, 1.8, 9.0, 9.0]

                nbr_fea 是通过以下步骤生成的：
                从晶体结构中提取每个原子的邻居信息（包括距离和邻居索引）。
                如果邻居数量不足，则使用虚拟邻居补全（补零和虚拟距离）。
                使用 GaussianDistance 对距离进行高斯拓展，将每个距离值映射到一个高维向量。
                '''
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +  [0] * (self.max_num_nbr - len(nbr))) 
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +  [self.radius + 1.] * (self.max_num_nbr -len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],nbr[:self.max_num_nbr])))

        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)

        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        target = torch.Tensor([float(target)])

        return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id
