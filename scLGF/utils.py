import torch
import opt
import random
import numpy as np
import h5py
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale, minmax_scale
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
# from sklearn import metrics
# from munkres import Munkres
import scanpy as sc

from scipy.optimize import linear_sum_assignment


def setup():
    print("setting:")
    setup_seed(opt.args.seed)
    if opt.args.name == 'zeisel':
        opt.args.n_clusters = 9
        opt.args.n_input = 2000
        opt.args.lambda_value = 10
        opt.args.gamma_value = 1e3

    elif opt.args.name == 'Quake_10x_Limb_Muscle':
        opt.args.n_clusters = 6
        opt.args.n_input = 2000
        opt.args.lambda_value = 0.1
        opt.args.gamma_value = 0.5
        opt.args.lr = 1e-5

    elif opt.args.name == 'Quake_Smart-seq2_Limb_Muscle':
        opt.args.n_clusters=6
        opt.args.n_input = 2000
        opt.args.lambda_value = 0.1
        opt.args.gamma_value = 0.5
        opt.args.lr = 1e-4

    elif opt.args.name == 'Romanov':
        opt.args.n_clusters = 7
        opt.args.n_input = 2000
        opt.args.lambda_value = 0.1
        opt.args.gamma_value = 0.1
        opt.args.lr = 1e-5

    elif opt.args.name == 'Muraro':
        opt.args.n_clusters = 9
        opt.args.n_input = 2000
        opt.args.lambda_value = 0.1
        opt.args.gamma_value = 0.1
        opt.args.lr = 1e-4

    elif opt.args.name == 'Young':
        opt.args.n_clusters = 11
        opt.args.n_input = 2000
        opt.args.lambda_value = 0.1
        opt.args.gamma_value = 0.1
        opt.args.lr = 1e-5

    elif opt.args.name == 'Pollen':
        opt.args.n_clusters = 11
        opt.args.n_input = 500
        opt.args.lambda_value = 0.1
        opt.args.gamma_value = 0.1
        opt.args.lr = 1e-5

    elif opt.args.name == 'Quake_Smart-seq2_Lung':
        opt.args.n_clusters = 11
        opt.args.n_input = 2000
        opt.args.lambda_value = 0.1
        opt.args.gamma_value = 0.1
        opt.args.lr = 1e-5
    elif opt.args.name == 'Quake_Smart-seq2_Heart':
        opt.args.n_clusters = 8
        opt.args.n_input = 2000
        opt.args.lambda_value = 10
        opt.args.gamma_value = 0.1
        opt.args.lr = 1e-5

    elif opt.args.name == 'Quake_10x_Bladder':
        opt.args.n_clusters = 4
        opt.args.n_input = 500
        opt.args.lambda_value = 100
        opt.args.gamma_value = 0.1
        opt.args.lr = 1e-5


    else:
        print("error!")
        print("please add the new dataset's parameters")
        print("------------------------------")
        print("dataset       : ")
        print("device        : ")
        print("random seed   : ")
        print("clusters      : ")
        print("alpha value   : ")
        print("lambda value  : ")
        print("gamma value   : ")
        print("learning rate : ")
        print("------------------------------")
        exit(0)

    opt.args.device = torch.device("cuda" if opt.args.cuda else "cpu")
    print("------------------------------")
    print("dataset       : {}".format(opt.args.name))
    print("device        : {}".format(opt.args.device))
    print("random seed   : {}".format(opt.args.seed))
    print("clusters      : {}".format(opt.args.n_clusters))
    print("lambda value  : {}".format(opt.args.lambda_value))
    print("gamma value   : {:.0e}".format(opt.args.gamma_value))
    print("learning rate : {:.0e}".format(opt.args.lr))
    print("------------------------------")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class load_data_origin_data(Dataset):
    def __init__(self, dataset, load_type, take_log=False, scaling=False):
        def load_txt():
            self.x = np.loadtxt('data/{}.txt'.format(dataset), dtype=float)
            self.y = np.loadtxt('data/{}_label.txt'.format(dataset), dtype=int)

        def load_h5():
            adata,adata_raw = loadH5AnnData('{}'.format(dataset))
            self.x = adata.X
            self.y = adata.obs['cell_labels'].values
            self.x1 = adata_raw.X
            self.y1 = adata_raw.obs['cell_labels'].values

        def load_csv():
            pre_process_paras = {'take_log': take_log, 'scaling': scaling}
            self.pre_process_paras = pre_process_paras
            print(pre_process_paras)
            dataset_list, dataset_list_raw = pre_processing_single(dataset, pre_process_paras, type='csv')
            self.x = dataset_list[0]['gene_exp'].transpose().astype(np.float32)
            self.x1 = dataset_list_raw[0]['gene_exp'].transpose().astype(np.float32)
            # self.x = x.T
            print(self.x.shape)
            self.y = dataset_list[0]['cell_labels'].astype(np.int32)
            self.y1 = dataset_list_raw[0]['cell_labels'].astype(np.int32)
            self.cluster_label = dataset_list[0]['cluster_labels'].astype(np.int32)

        if load_type == "csv":
            load_csv()
        elif load_type == "h5":
            load_h5()
        elif load_type == "txt":
            load_txt()

def read_csv(filename, take_log):
    """ Read TPM data of a dataset saved in csv format
    Format of the csv:
    first row: sample labels
    second row: cell labels
    third row: cluster labels from Seurat
    first column: gene symbols
    Args:
        filename: name of the csv file
        take_log: whether do log-transformation on input data
    Returns:
        dataset: a dict with keys 'gene_exp', 'gene_sym', 'sample_labels', 'cell_labels', 'cluster_labels'
    """
    dataset = {}
    df = pd.read_csv(filename, header=None)
    dat = df[df.columns[1:]].values
    dataset['sample_labels'] = dat[0, :].astype(int)
    dataset['cell_labels'] = dat[1, :].astype(int)
    dataset['cluster_labels'] = dat[2, :].astype(int)
    gene_sym = df[df.columns[0]].tolist()[3:]
    gene_exp = dat[3:, :]


    if take_log:
            gene_exp = np.log2(gene_exp + 1)
    dataset['gene_exp'] = gene_exp
    dataset['gene_sym'] = gene_sym
    return dataset

def read_txt(filename, take_log):
    dataset = {}
    # df = pd.read_table(filename, header=None)
    # dat = df[df.columns[1:]].values
    # dataset['cell_labels'] = dat[8, 1:]
    # gene_sym = df[df.columns[0]].tolist()[11:]
    # gene_exp = dat[11:, 1:].astype(np.float32)
    # if take_log:
    #     gene_exp = np.log2(gene_exp + 1)
    # dataset['gene_exp'] = gene_exp
    # dataset['gene_sym'] = gene_sym
    # dataset['cell_labels'] = convert_strclass_to_numclass(dataset['cell_labels'])
    #
    # save_csv(gene_exp, gene_sym,  dataset['cell_labels'])

    return dataset

def pre_processing_single(dataset_file_list, pre_process_paras, type=opt.args.load_type):
    """ pre-processing of multiple datasets
    Args:
        dataset_file_list: list of filenames of datasets
        pre_process_paras: dict, parameters for pre-processing
    Returns:
        dataset_list: list of datasets
    """
    # parameters
    take_log = pre_process_paras['take_log']
    scaling = pre_process_paras['scaling']
    dataset_list = []
    dataset_list_raw = []
    data_file = dataset_file_list

    if type == 'csv':
        dataset = read_csv(data_file, take_log)
    elif type == 'txt':
        dataset = read_txt(data_file, take_log)


    dataset['gene_exp'] = dataset['gene_exp'].astype(np.float)
    dataset_list_raw.append(dataset)

    if scaling:  # scale to [0,1]
        minmax_scale(dataset['gene_exp'], feature_range=(0, 1), axis=1, copy=False)


    dataset_list.append(dataset)
    return dataset_list, dataset_list_raw

def select(dataset):
    data = dataset.x
    sc.pp.highly_variable_genes(data, min_mean=0.0125, max_mean=3, min_disp=0.5)
    sc.pl.highly_variable_genes(data)
    data.raw = data
    data = data[:, data.var['highly_variable']]
    return dataset

def load_graph(count,k=opt.args.k, pca=50, mode="connectivity"):

    if pca:
        countp = dopca(count, dim=pca)
    else:
        countp = count
    A = kneighbors_graph(countp, k, mode=mode, metric="euclidean", include_self=True)
    adj = A.toarray()
    adj_n = norm_adj(adj)
    return adj, adj_n

def dopca(X, dim=10):
    pcaten = PCA(n_components=dim)
    X_10 = pcaten.fit_transform(X)
    return X_10


    import os
    print("delete file: ", path)
    os.remove(path)

    return adj


def norm_adj(A):
    normalized_D = degree_power(A, -0.5)
    output = normalized_D.dot(A).dot(normalized_D)
    return output


def degree_power(A, k):
    degrees = np.power(np.array(A.sum(1)), k).flatten()
    degrees[np.isinf(degrees)] = 0.
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D

def numpy_to_torch(a, sparse=False):
    """
    numpy array to torch tensor
    :param a: the numpy array
    :param sparse: is sparse tensor or not
    :return: torch tensor
    """
    if sparse:
        a = torch.sparse.Tensor(a)
        a = a.to_sparse()
    else:
        a = torch.FloatTensor(a)
    return a


def torch_to_numpy(t):
    """
    torch tensor to numpy array
    :param t: the torch tensor
    :return: numpy array
    """
    return t.numpy()

class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))

def pretrain(model, dataset, A_norm):
    print("Pretraining...")
    model.pretrain(LoadDataset(dataset.x), A_norm)




def load_pretrain_parameter(model):
    """
    load pretrained parameters
    Args:
        model: Dual Correlation Reduction Network
    Returns: model
    """
    pretrained_dict = torch.load('model_pretrain/{}_pretrain.pkl'.format(opt.args.name), map_location='cpu')
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def eva(y_true, y_pred, epoch=0):
    acc= cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    # print('Epoch_{}'.format(epoch), ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
    #       ', f1 {:.4f}'.format(f1))
    print('Epoch_{}'.format(epoch), ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))
    return acc, nmi, ari

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array((ind[0], ind[1])).T

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def clustering(Z, y):
    """
    clustering based on embedding
    Args:
        Z: the input embedding
        y: the ground truth

    Returns: acc, nmi, ari, f1, clustering centers
    """
    model = KMeans(n_clusters=opt.args.n_clusters, n_init=20)
    cluster_id = model.fit_predict(Z.data.cpu().numpy())
    acc, nmi, ari= eva(y, cluster_id, show_details=opt.args.show_training_details)
    return acc, nmi, ari, model.cluster_centers_


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def read_dataset(adata, transpose=False, test_split=False, copy=False):

    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError

    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error

    # if adata.X.size < 50e6: # check if adata.X is integer only if array is small
    #     if sp.sparse.issparse(adata.X):
    #         assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
    #     else:
    #         assert np.all(adata.X.astype(int) == adata.X), norm_error

    if transpose: adata = adata.transpose()
    from sklearn.model_selection import train_test_split
    if test_split:
        train_idx, test_idx = train_test_split(np.arange(adata.n_obs), test_size=0.1, random_state=42)
        spl = pd.Series(['train'] * adata.n_obs)
        spl.iloc[test_idx] = 'test'
        adata.obs['DCA_split'] = spl.values
    else:
        adata.obs['DCA_split'] = 'train'

    adata.obs['DCA_split'] = adata.obs['DCA_split'].astype('category')
    print('### Autoencoder: Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))

    return adata

def loadH5AnnData(path):
    x, y = prepro(path)

    print("Cell number:", x.shape[0])
    print("Gene number", x.shape[1])
    x = np.ceil(x).astype(np.int)
    cluster_number = len(np.unique(y))
    print("Cluster number:", cluster_number)
    opt.args.n_clusters = cluster_number
    adata = sc.AnnData(x)
    adata.obs['cell_labels'] = y

    adata_nom = normalize(adata, copy=True, highly_genes=opt.args.highly_genes, size_factors=True, normalize_input=True, logtrans_input=True)
    return adata_nom,adata

def prepro(filename):
    data_path = filename
    mat, obs, var, uns = read_data(data_path, sparsify=False, skip_exprs=False)
    if isinstance(mat, np.ndarray):
        X = np.array(mat)
    else:
        X = np.array(mat.toarray())
    cell_name = np.array(obs["cell_type1"])
    cell_type, cell_label = np.unique(cell_name, return_inverse=True)
    return X, cell_label

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def read_clean(data):
    assert isinstance(data, np.ndarray)
    if data.dtype.type is np.bytes_:
        data = decode(data)
    if data.size == 1:
        data = data.flat[0]
    return data

def dict_from_group(group):
    assert isinstance(group, h5py.Group)
    d = dotdict()
    for key in group:
        if isinstance(group[key], h5py.Group):
            value = dict_from_group(group[key])
        else:
            value = read_clean(group[key][...])
        d[key] = value
    return d

def read_data(filename, sparsify = False, skip_exprs = False):
    with h5py.File(filename, "r") as f:
        obs = pd.DataFrame(dict_from_group(f["obs"]), index = decode(f["obs_names"][...]))
        var = pd.DataFrame(dict_from_group(f["var"]), index = decode(f["var_names"][...]))
        uns = dict_from_group(f["uns"])
        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                mat = sp.csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                               exprs_handle["indptr"][...]), shape = exprs_handle["shape"][...])
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = sp.csr_matrix(mat)
        else:
            mat = sp.csr_matrix((obs.shape[0], var.shape[0]))
    return mat, obs, var, uns

def empty_safe(fn, dtype):
    def _fn(x):
        if x.size:
            return fn(x)
        return x.astype(dtype)
    return _fn

def normalize(adata, copy=True, highly_genes = None, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError
    # norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    # assert 'n_count' not in adata.obs, norm_error
    # if adata.X.size < 50e6: # check if adata.X is integer only if array is small
    #     if sp.sparse.issparse(adata.X):
    #         assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
    #     else:
    #         assert np.all(adata.X.astype(int) == adata.X), norm_error

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0
    if logtrans_input:
        sc.pp.log1p(adata)
    if highly_genes != None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = highly_genes, subset=True)
    if normalize_input:
        sc.pp.scale(adata)
    return adata

decode = empty_safe(np.vectorize(lambda _x: _x.decode("utf-8")), str)

def save(cluster_lables):
    test = pd.DataFrame(columns=[opt.args.name], data=cluster_lables)
    test.to_csv('scDFCN.csv', index=False, sep=',')