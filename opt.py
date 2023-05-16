import argparse

parser = argparse.ArgumentParser(description='scDFCN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# setting
#Quake_10x_Limb_Muscle,Quake_Smart-seq2_Limb_Muscle,zeisel,Muraro,Romanov
parser.add_argument('--name', type=str, default="Muraro")
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--seed', type=int, default=5)
parser.add_argument('--data_file', default='data/Macosko.h5')
parser.add_argument('--load_type', type=str, default='h5')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--highly_genes',type=int,default=2000)
parser.add_argument('--k',type=int,default=15)


# parser.add_argument('--alpha_value', type=float, default=0.2)
# parser.add_argument('--lambda_value', type=float, default=0)
# parser.add_argument('--gamma_value', type=float, default=1e3)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--n_z', type=int, default=32)
parser.add_argument('--epoch', type=int, default=300)
parser.add_argument('--show_training_details', type=bool, default=True)
#pre
parser.add_argument('--pre_batch_size', type=int, default=256)
parser.add_argument('--pre_epoch_ae', type=int, default=100)
parser.add_argument('--pre_epoch_gae', type=int, default=100)
parser.add_argument('--pre_epoch', type=int, default=100)
parser.add_argument('--pre_lr', type=float, default=1e-6)
parser.add_argument('--noise_value', type=float, default=1)


# AE structure parameter from DFCN
parser.add_argument('--ae_n_enc_1', type=int, default=128)
parser.add_argument('--ae_n_enc_2', type=int, default=256)
parser.add_argument('--ae_n_enc_3', type=int, default=512)
parser.add_argument('--ae_n_dec_1', type=int, default=512)
parser.add_argument('--ae_n_dec_2', type=int, default=256)
parser.add_argument('--ae_n_dec_3', type=int, default=128)

# IGAE structure parameter from DFCN
parser.add_argument('--gae_n_enc_1', type=int, default=128)
parser.add_argument('--gae_n_enc_2', type=int, default=256)
parser.add_argument('--gae_n_enc_3', type=int, default=32)
parser.add_argument('--gae_n_dec_1', type=int, default=32)
parser.add_argument('--gae_n_dec_2', type=int, default=256)
parser.add_argument('--gae_n_dec_3', type=int, default=128)

# clustering performance: acc, nmi, ari, f1
parser.add_argument('--acc', type=float, default=0)
parser.add_argument('--nmi', type=float, default=0)
parser.add_argument('--ari', type=float, default=0)
parser.add_argument('--f1', type=float, default=0)

args = parser.parse_args()
