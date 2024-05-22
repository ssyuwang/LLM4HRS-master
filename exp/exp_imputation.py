from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.utils import normalization_grud as normalization
from utils.utils import renormalization_np as renormalization
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from tqdm import tqdm

warnings.filterwarnings('ignore')

class Exp_Imputation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Imputation, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight)
        return model_optim

    def _select_criterion(self):
        criterion = nn.L1Loss()
        #criterion = nn.MSELoss()

        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y,batch_mask,eval_mask) in tqdm(enumerate(vali_loader)):
                batch_x = batch_x.float().to(self.device)

                batch_mask = batch_mask.to(self.device)
                eval_mask = eval_mask.to(self.device)

                # random mask
                B, T, N = batch_x.shape
                batch_x = batch_x.float().to(self.device)
                batch_x_copy = batch_x
  
                outputs = self.model(batch_x, batch_mask)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_x_copy.detach().cpu()
                eval_mask = eval_mask.detach().cpu()
                
                loss = criterion(pred[eval_mask ==1], true[eval_mask ==1])

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
      
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y,batch_mask,eval_mask) in tqdm(enumerate(train_loader)):
                iter_count += 1
                model_optim.zero_grad()

                B, T, N = batch_x.shape

                batch_x = batch_x.float().to(self.device)
                batch_mask = batch_mask.to(self.device)
                eval_mask = eval_mask.to(self.device)

                batch_x_copy = batch_x

                outputs= self.model(batch_x, batch_mask)

                
               # intact_missing_mask =  intact_missing_mask.float().to(self.device)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
        
                loss1 = criterion(outputs[eval_mask ], batch_x_copy[eval_mask ])
                loss =  loss1 

               
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        missing_masks = []
        indicating_masks = []
        eval_masks = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y,batch_mask,eval_mask) in tqdm(enumerate(test_loader)):
                batch_x = batch_x.float().to(self.device)

                batch_mask = batch_mask.to(self.device)
                eval_mask = eval_mask.to(self.device)
                batch_x_copy = batch_x
                
                B, T, N = batch_x.shape
                
                outputs = self.model(batch_x, batch_mask)

                # eval
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                outputs = outputs.detach().cpu().numpy()
                pred = outputs
                true = batch_x_copy.detach().cpu().numpy()
                eval_mask = eval_mask.detach().cpu().numpy()
                
                preds.append(pred)
                trues.append(true)
                eval_masks.append(eval_mask)
               
               
                if i % 20 == 0:
                    filled = true[0, :, -1].copy()
                    #filled = filled * eval_masks[0, :, -1] + \
                    #         pred[0, :, -1] * (1 - eval_masks[0, :, -1])
                    visual(true[0, :, -1], filled, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        eval_masks = np.concatenate(eval_masks, 0)
        
        print('test shape:', preds.shape, trues.shape)

        raw = np.loadtxt(os.path.join(self.args.root_path, self.args.data_path), dtype=float,delimiter=",")  
        data, norm_parameters = normalization(raw)

        trues1 = renormalization(trues.reshape(-1,N),norm_parameters)
        trues1 = trues1.reshape(-1,T,N)
        preds1 = renormalization(preds.reshape(-1,N),norm_parameters)
        preds1 = preds1.reshape(-1,T,N)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae1, mse1, rmse1, mape1, mspe1 = metric(preds1[eval_masks  == 1], trues1[eval_masks  == 1])
        mae, mse, rmse, mape, mspe = metric(preds[eval_masks  == 1], trues[eval_masks  == 1])
        #mse = cal_mse(preds,trues,indicating_masks)
        #mae = cal_mae(preds,trues,indicating_masks)
        print('mse:{}, mae:{},rmse:{}'.format(mse, mae,rmse))
        f = open("result_imputation.txt", 'a')
        f.write(setting + "  \n")
        f.write('rmse1:{},mae1:{},mape1:{},rmse:{}, mae:{}, maps:{}'.format(rmse1,mae1,mape1,rmse, mae,mape))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds1)
        np.save(folder_path + 'true.npy', trues1)
        return
