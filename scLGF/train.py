import tqdm

import opt
from utils import *
from torch.optim import Adam
import torch.nn.functional as F
import scanpy as sc

acc_reuslt = []
nmi_result = []
ari_result = []
f1_result = []
tol=1e-3


def train(model, dataset, x, y, A, A_norm):
    optimizer = Adam(model.parameters(), lr=opt.args.lr)
    original_acc = opt.args.acc

    print("Training on {}…".format(opt.args.name))
    with torch.no_grad():
        x_hat, z_hat, adj_hat, z_ae, z_igae, _, _, _, z_tilde = model(x, A_norm)
    kmeans = KMeans(n_clusters=opt.args.n_clusters, n_init=20)
    cluster_id = kmeans.fit_predict(z_tilde.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(opt.args.device)
    eva(y, cluster_id, 'Initialization')

    y_pred_last = cluster_id
    num = x.shape[0]



    for epoch in range(opt.args.epoch):
        x_hat, z_hat, adj_hat, z_ae, z_sgae, q, q1, q2, z_tilde = model(x, A_norm)

        tmp_q = q.data
        p = target_distribution(tmp_q)

        loss_ae = F.mse_loss(x_hat, x)
        loss_w = F.mse_loss(z_hat, torch.spmm(A_norm, x))
        loss_a = F.mse_loss(adj_hat, A_norm.to_dense())
        loss_sgae =loss_w +opt.args.gamma_value * loss_a
        loss_kl = F.kl_div((q.log() + q1.log() + q2.log()) / 3, p, reduction='batchmean')
        loss = loss_ae + loss_sgae + opt.args.lambda_value *loss_kl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('{} loss: {}'.format(epoch, loss))

        kmeans = KMeans(n_clusters=opt.args.n_clusters, n_init=20).fit(z_tilde.data.cpu().numpy())
        labels = kmeans.labels_

        acc, nmi, ari= eva(y, labels, epoch)
        acc_reuslt.append(acc)
        nmi_result.append(nmi)
        ari_result.append(ari)
        print("epoch:",epoch," nmi:",nmi," ari:",ari)

        delta_label = np.sum(labels != y_pred_last).astype(np.float32) / num
        if epoch > 0 and delta_label < tol:
            print('delta_label ', delta_label, '< tol ', tol)
            print("Reach tolerance threshold. Stopping training.")
            break

    adata = sc.AnnData(z_tilde.cpu().detach().numpy())
    cluster_lables=kmeans.labels_

    return adata,cluster_lables, acc_reuslt,nmi_result,ari_result


