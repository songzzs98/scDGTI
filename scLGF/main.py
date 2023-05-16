from __future__ import print_function, division
from utils import *
import opt
from  scLGF import scLGF
from train import train, acc_reuslt, nmi_result, f1_result, ari_result
if __name__ == "__main__":
    #
    print(opt.args.name)
    setup()

    file_path = "data/" + opt.args.name + "." + opt.args.load_type
    dataset = load_data_origin_data(file_path, opt.args.load_type, scaling=True)
    x = dataset.x
    y = dataset.y
    adata = sc.AnnData(x)
    adata.obs['cell_labels'] = y

    x1 = dataset.x1
    y1 = dataset.y1
    adata_raw = sc.AnnData(x1)
    adata_raw.obs['cell_labels'] = y1


    A, A_norm = load_graph(x)

    x = numpy_to_torch(x).to(opt.args.device)
    A_norm = numpy_to_torch(A_norm, sparse=True).to(opt.args.device)
    model = scLGF(n_node=x.shape[0]).to(opt.args.device)
    print(model)


    pretrain(model,LoadDataset(x), A_norm)
    adata_embedding,cluster_lables, acc_reuslt,nmi_result,ari_result = train(model, LoadDataset(x), x, y, A, A_norm)
    print("NMI: {:.4f}".format(nmi_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]))
    print("ARI: {:.4f}".format(ari_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]))
    print("Epoch:", np.where(acc_reuslt == np.max(acc_reuslt))[0][0])