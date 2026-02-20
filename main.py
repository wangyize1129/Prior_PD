import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
#from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score,precision_recall_fscore_support,f1_score,accuracy_score,precision_score,recall_score,balanced_accuracy_score
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict

# from models.dropout import LinearScheduler
from aeon.datasets import load_classification

from utils import *

from torch.cuda.amp import GradScaler, autocast
                                         
import torch.nn.functional as F
from mydataload import loadorean, ParkinsonDataset
from multiscale_diffusion_adapter import Application
import random
# from sample_method import rd_torch,dpp
import random
from timm.optim.adamp import AdamP
from lookhead import Lookahead
import warnings

from models.timemil_cfca import TimeMIL_cfca

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

                                  
def train(trainloader, milnet, criterion, optimizer, epoch,args):
    milnet.train()
    total_loss = 0

    for batch_id, (feats, label) in enumerate(trainloader):
        bag_feats = feats.to(device)
        bag_label = label.to(device)
        optimizer.zero_grad()
        if epoch < args.epoch_des:
            bag_prediction = milnet(bag_feats, warmup=True)
        else:
            bag_prediction = milnet(bag_feats, warmup=False)
        bag_loss = criterion(bag_prediction, bag_label)
        loss = bag_loss
        sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f  total loss: %.4f' % \
                        (batch_id, len(trainloader), bag_loss.item(), loss.item()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(milnet.parameters(), 2.0)
        optimizer.step()
        total_loss = total_loss + bag_loss
    return total_loss / len(trainloader)



def test(testloader, milnet, criterion, args):
    milnet.eval()
    total_loss = 0
    test_labels = []
    test_predictions = []
    
    with torch.no_grad():
        for batch_id, (feats, label) in enumerate(testloader):
            bag_feats = feats.to(device)
            bag_label = label.to(device)
            bag_prediction = milnet(bag_feats)  # b*class
            bag_loss = criterion(bag_prediction, bag_label)
            loss = bag_loss
            total_loss = total_loss + loss.item()
            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (batch_id, len(testloader), loss.item()))
            test_labels.extend([label.cpu().numpy()])
            test_predictions.extend([torch.sigmoid(bag_prediction).cpu().numpy()])
    
    test_labels = np.vstack(test_labels)
    test_predictions = np.vstack(test_predictions)

    test_predictions_prob = np.exp(test_predictions)/np.sum(np.exp(test_predictions),axis=1,keepdims=True)
    
    
    test_predictions = np.argmax(test_predictions,axis=1)
    test_labels = np.argmax(test_labels,axis=1)
    avg_score = accuracy_score(test_labels,test_predictions)
    balanced_avg_score = balanced_accuracy_score(test_labels,test_predictions)
    f1_marco = f1_score(test_labels,test_predictions,average='macro')
    f1_micro = f1_score(test_labels,test_predictions,average='micro')
    
    p_marco = precision_score(test_labels,test_predictions,average='macro')
    p_micro = precision_score(test_labels,test_predictions,average='micro')
    
    r_marco = recall_score(test_labels,test_predictions,average='macro')
    r_micro = recall_score(test_labels,test_predictions,average='micro')
    
    
    r_marco = recall_score(test_labels,test_predictions,average='macro')
    r_micro = recall_score(test_labels,test_predictions,average='micro')
    
    if args.num_classes ==2: 
        # print(test_labels.shape)
        
        roc_auc_ovo_marco=   roc_auc_score(test_labels,test_predictions_prob[:,1],average='macro')
        roc_auc_ovo_micro=  0.# roc_auc_score(test_labels,test_predictions_prob,average='micro',multi_class='ovo')
    
        roc_auc_ovr_marco=   roc_auc_score(test_labels,test_predictions_prob[:,1],average='macro')
        roc_auc_ovr_micro=  0.# roc_auc_score(test_labels,test_predictions_prob,average='micro',multi_class='ovr')
    
    
    else:
        roc_auc_ovo_marco=   roc_auc_score(test_labels,test_predictions_prob,average='macro',multi_class='ovo')
        roc_auc_ovo_micro=  0.# roc_auc_score(test_labels,test_predictions_prob,average='micro',multi_class='ovo')

        roc_auc_ovr_marco=   roc_auc_score(test_labels,test_predictions_prob,average='macro',multi_class='ovr')
        roc_auc_ovr_micro=  0.# 
    

    
    results = [avg_score,balanced_avg_score,f1_marco,f1_micro, p_marco,p_micro,r_marco,r_micro,roc_auc_ovo_marco,roc_auc_ovo_micro,roc_auc_ovr_marco,roc_auc_ovr_micro]
    
    
    return total_loss / len(testloader), results



def main():
    parser = argparse.ArgumentParser(description='time classification by TimeMIL')
    parser.add_argument('--dataset', default="Parkinson", type=str, help='dataset ')
    parser.add_argument('--num_classes', default=3, type=int, help='Number of output classes [2]')
    parser.add_argument('--num_workers', default=0, type=int, help='number of workers used in dataloader [4]')
    parser.add_argument('--feats_size', default=768, type=int, help='Dimension of the feature size [512] resnet-50 1024')
    parser.add_argument('--lr', default=0.004980622158522066, type=float, help='1e-3 Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=120, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=0.00010024529090703385, type=float, help='Weight decay 1e-4]')
    parser.add_argument('--dropout_patch', default=0.7748341185661919, type=float, help='Patch dropout rate [0] 0.5')
    parser.add_argument('--dropout_node', default=0.33369622455087644, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--optimizer', default='adamw', type=str, help='adamw sgd')
    parser.add_argument('--save_dir', default='./savemodel/', type=str, help='the directory used to save all the output')
    parser.add_argument('--epoch_des', default=10, type=int, help='turn on warmup')
   
    parser.add_argument('--embed', default=192, type=int, help='Number of embedding')
    
    parser.add_argument('--batchsize', default=64, type=int, help='batchsize')

    parser.add_argument('--model', default='cfca', type=str,
                        help='Model type: timemil (original) or cfca (Time-Channel Factorized Attention)')
    parser.add_argument('--channel_reduction', default=4, type=int, help='Channel attention reduction ratio')
    parser.add_argument('--channel_pool', default='mean_max', type=str, choices=['mean', 'max', 'mean_max'],
                        help='Channel attention pooling type')
    

    parser.add_argument('--use_multiscale_diffusion', action='store_true', help='Use FFT filter bank multi-scale diffusion augmentation (fixed 3 bands: 0.5-3Hz, 3-7Hz, 7-12Hz)')
    parser.add_argument('--diffusion_parkinson_mult', default=2, type=int, help='Parkinson samples multiplier')
    parser.add_argument('--diffusion_healthy_mult', default=7, type=int, help='Healthy samples multiplier')
    parser.add_argument('--diffusion_other_mult', default=5, type=int, help='Other samples multiplier')
    parser.add_argument('--diffusion_epochs', default=30, type=int, help='Diffusion model training epochs')
    parser.add_argument('--diffusion_steps', default=100, type=int, help='Number of diffusion steps (more=better quality but slower)')
    parser.add_argument('--diffusion_batch_size', default=32, type=int, help='Batch size for diffusion training')
    parser.add_argument('--diffusion_lr', default=2.1930485556643678e-05, type=float, help='Learning rate for diffusion training')
    parser.add_argument('--diffusion_hidden_dim', default=64, type=int, help='Hidden dimension of diffusion model')
    parser.add_argument('--lambda1', default=0.04093813608598782, type=float, help='Frequency prior loss weight (L_freq, for 3-7Hz tremor in Parkinson only)')
    parser.add_argument('--lambda2', default=0, type=float, help='Reconstruction consistency loss weight (L_recon)')
    parser.add_argument('--augment_dir', default='/tmp/Parkinson/multiscale_augment', type=str, 
                        help='Directory to save/load augmented data')
    
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    
    args.save_dir = args.save_dir+'InceptBackbone'
    
    maybe_mkdir_p(join(args.save_dir, f'{args.dataset}'))
    args.save_dir = make_dirs(join(args.save_dir, f'{args.dataset}'))
    maybe_mkdir_p(args.save_dir)

    # <------------- set up logging ------------->
    logging_path = os.path.join(args.save_dir, 'Train_log.log')
    logger = get_logger(logging_path)

    # <------------- save hyperparams ------------->
    option = vars(args)
    file_name = os.path.join(args.save_dir, 'option.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(option.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')

    # criterion = nn.CrossEntropyLoss(label_smoothing=0.0)#0.01
    criterion = nn.BCEWithLogitsLoss() # one-vs-rest binary MIL

    
    trainset = ParkinsonDataset(args, split='train', seed=args.seed)
    testset = ParkinsonDataset(args, split='test', seed=args.seed)

    if args.use_multiscale_diffusion:
        trainset = Application(
            trainset,
            args,
            parkinson_multiplier=args.diffusion_parkinson_mult,
            healthy_multiplier=args.diffusion_healthy_mult,
            other_multiplier=args.diffusion_other_mult,
            training_epochs=args.diffusion_epochs,
            num_diffusion_steps=args.diffusion_steps,
            batch_size=args.diffusion_batch_size,
            lr=args.diffusion_lr,
            hidden_dim=args.diffusion_hidden_dim,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            augment_dir=args.augment_dir
        )

    seq_len, num_classes, L_in = trainset.max_len, trainset.num_class, trainset.feat_in

    args.feats_size = L_in
    args.num_classes = num_classes

    # <------------- define MIL network ------------->
    milnet = TimeMIL_cfca(
        args.feats_size,
        mDim=args.embed,
        n_classes=num_classes,
        dropout=args.dropout_node,
        max_seq_len=seq_len,
        channel_reduction=args.channel_reduction,
        channel_pool=args.channel_pool
    ).to(device)
    
    if  args.optimizer == 'adamw':
    # optimizer = torch.optim.AdamW(milnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.AdamW(milnet.parameters(), lr=args.lr,  weight_decay=args.weight_decay)
        optimizer =Lookahead(optimizer)    
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(milnet.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        # optimizer =Lookahead(optimizer) 
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer =Lookahead(optimizer) 
    
    elif args.optimizer == 'adamp':
        optimizer = AdamP(milnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer =Lookahead(optimizer) 
    

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)

    trainloader = DataLoader(trainset, args.batchsize, shuffle=True, num_workers=args.num_workers, drop_last=False, pin_memory=True)
    # if args.batchsize==1:
    #     testloader = DataLoader(testset, args.batchsize, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True)
    # else:
    testloader = DataLoader(testset, 128, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True)

    best_score = 0
    save_path = join(args.save_dir, 'weights')
    os.makedirs(save_path, exist_ok=True)
    
    os.makedirs(join(args.save_dir,'lesion'), exist_ok=True)
    results_best = None
    for epoch in range(1, args.num_epochs + 1):

        train_loss_bag = train(trainloader, milnet, criterion, optimizer, epoch,args) # iterate all bags

        test_loss_bag,results= test(testloader, milnet, criterion, args)
      
        [avg_score,balanced_avg_score,f1_marco,f1_micro,p_marco,p_micro,r_marco,r_micro,roc_auc_ovo_marco,roc_auc_ovo_micro,roc_auc_ovr_marco,roc_auc_ovr_micro] = results

    
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        logger.info('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, accuracy: %.4f, bal. average score: %.4f, f1 marco: %.4f   f1 mirco: %.4f  p marco: %.4f   p mirco: %.4f r marco: %.4f   r mirco: %.4f  roc_auc ovo marco: %.4f   roc_auc ovo mirco: %.4f  roc_auc ovr marco: %.4f   roc_auc ovr mirco: %.4f, lr: %.6f' % 
                  (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score,balanced_avg_score,f1_marco,f1_micro, p_marco,p_micro,r_marco,r_micro,roc_auc_ovo_marco,roc_auc_ovo_micro,roc_auc_ovr_marco,roc_auc_ovr_micro, current_lr))

        current_score = avg_score
        if current_score >= best_score:
            
            results_best = results
            
            best_score = current_score
            print(current_score)
            save_name = os.path.join(save_path, 'best_model.pth')
            torch.save(milnet.state_dict(), save_name)
            logger.info('Best model saved at: ' + save_name)


    [avg_score,balanced_avg_score,f1_marco,f1_micro,p_marco,p_micro,r_marco,r_micro,roc_auc_ovo_marco,roc_auc_ovo_micro,roc_auc_ovr_marco,roc_auc_ovr_micro] = results_best
    logger.info('\r Best  Results: accuracy: %.4f, bal. average score: %.4f, f1 marco: %.4f   f1 mirco: %.4f  p marco: %.4f   p mirco: %.4f r marco: %.4f   r mirco: %.4f  roc_auc ovo marco: %.4f   roc_auc ovo mirco: %.4f  roc_auc ovr marco: %.4f   roc_auc ovr mirco: %.4f' % 
                  ( avg_score,balanced_avg_score,f1_marco,f1_micro, p_marco,p_micro,r_marco,r_micro,roc_auc_ovo_marco,roc_auc_ovo_micro,roc_auc_ovr_marco,roc_auc_ovr_micro )) 


if __name__ == '__main__':
    main()
