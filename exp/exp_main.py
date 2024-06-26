from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, DLinear_FITS, FITS, FITS_DLinear, FITS_p, FITS_100
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

from utils.metrics import SE
from models.Stat_models import Naive_repeat

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'DLinear_FITS': DLinear_FITS,
            'Naive_repeat': Naive_repeat,  # Add this line
            'FITS': FITS,
            'FITS_DLinear': FITS_DLinear,
            'FITS_p': FITS_p,
            'FITS_100': FITS_100,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'DLinear' or 'DLinear_FITS' or 'FITS' or 'FITS_DLinear' or 'FITS_p' or 'FITS_100' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'DLinear' or 'DLinear_FITS' or 'FITS' or 'FITS_DLinear' or 'FITS_p' or 'FITS_100' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

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

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'DLinear' or 'DLinear_FITS' or 'FITS' or 'FITS_DLinear' or 'FITS_p' or 'FITS_100' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'DLinear' or 'DLinear_FITS' or 'FITS' or 'FITS_DLinear' or 'FITS_p' or 'FITS_100' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; time left: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.1f} seconds")
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
        naive_model = Naive_repeat(self.args).to(self.device)
            
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print(setting)
        self.model.eval()
        naive_model.eval()
        
        preds = []
        naive_preds = []
        trues = []
        gts = []
        pds = []
        naive_pds = []

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                naive_output = naive_model(batch_x)

                if 'DLinear' or 'DLinear_FITS' or 'FITS' or 'FITS_DLinear' or 'FITS_p' or 'FITS_100' in self.args.model:
                    outputs = self.model(batch_x)
                else:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                naive_output = naive_output[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu().numpy()
                naive_pred = naive_output.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                naive_preds.append(naive_pred)
                trues.append(true)

                if i % 1 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    naive_pd = np.concatenate((input[0, :, -1], naive_pred[0, :, -1]), axis=0)
                    gts.append(gt)
                    pds.append(pd)
                    naive_pds.append(naive_pd)

        preds = np.concatenate(preds, axis=0)
        naive_preds = np.concatenate(naive_preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        gts = np.array(gts)
        pds = np.array(pds)
        naive_pds = np.array(naive_pds)

        np.save(os.path.join(folder_path, 'gt.npy'), gts)
        np.save(os.path.join(folder_path, 'pd.npy'), pds)
        np.save(os.path.join(folder_path, 'naive_pd.npy'), naive_pds)

        print(f"Shapes: pred {preds.shape}, true {trues.shape}, naive_pred {naive_preds.shape}")
        print(f"Shapes: gt {gts.shape}, pd {pds.shape}, naive_pd {naive_pds.shape}")
        # print("Saved gt.npy and pd.npy files")
       
        def MAE(output, target):
            return torch.mean(torch.abs(output - target))

        criterion_mse = nn.MSELoss()
        criterion_mae = MAE

        def compute_metrics(preds, trues, criterion_mse, criterion_mae):
            mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
            se = SE(preds, trues)
            mse_torch = criterion_mse(torch.tensor(preds), torch.tensor(trues)).item()
            mae_torch = criterion_mae(torch.tensor(preds), torch.tensor(trues)).item()
            se_torch = torch.mean((torch.tensor(preds)[:, -1, :] - torch.tensor(trues)[:, -1, :]) ** 2).item()
            
            # Calculate relative RMSE and RMAE
            rmse = np.sqrt(mse)
            rmse_torch = np.sqrt(mse_torch)
            relative_rmse = rmse / np.mean(np.abs(trues))
            relative_rmse_torch = rmse_torch / np.mean(np.abs(trues))
            relative_mae = mae / np.mean(np.abs(trues))
            relative_mae_torch = mae_torch / np.mean(np.abs(trues))
            
            return {
                'mse': mse, 'mae': mae, 'se': se, 'relative_rmse': relative_rmse, 'relative_mae': relative_mae,
                'mse_torch': mse_torch, 'mae_torch': mae_torch, 'se_torch': se_torch, 
                'relative_rmse_torch': relative_rmse_torch, 'relative_mae_torch': relative_mae_torch
            }

        def log_metrics(results, filename='final_metrics.txt'):
            print("Final Metrics:")
            
            # Print results
            for model_name, metrics in results.items():
                print(f"{model_name:<12} MSE: {metrics['mse']:<10.6f} MAE: {metrics['mae']:<10.6f} SE: {metrics['se']:<10.6f} "
                    f"RRMSE: {metrics['relative_rmse']:<10.6f} RMAE: {metrics['relative_mae']:<10.6f} (numpy)")
                print(f"{model_name:<12} MSE: {metrics['mse_torch']:<10.6f} MAE: {metrics['mae_torch']:<10.6f} SE: {metrics['se_torch']:<10.6f} "
                    f"RRMSE: {metrics['relative_rmse_torch']:<10.6f} RMAE: {metrics['relative_mae_torch']:<10.6f} (torch)")
                print()

            # Write results to file
            with open(filename, 'w') as f:
                f.write("Final Metrics:\n")
                for model_name, metrics in results.items():
                    f.write(f"{model_name:<12} MSE: {metrics['mse']:<10.3f} MAE: {metrics['mae']:<10.3f} SE: {metrics['se']:<10.3f} "
                            f"RRMSE: {metrics['relative_rmse_torch']:<10.2%} | RRMSE: {metrics['relative_rmse']:<10.3f} RMAE: {metrics['relative_mae']:<10.3f} (numpy)\n")
                    f.write(f"{model_name:<12} MSE: {metrics['mse_torch']:<10.3f} MAE: {metrics['mae_torch']:<10.3f} SE: {metrics['se_torch']:<10.3f} "
                            f"RRMSE: {metrics['relative_rmse_torch']:<10.2%} | RRMSE: {metrics['relative_rmse_torch']:<10.3f} RMAE: {metrics['relative_mae_torch']:<10.3f} (torch)\n")
                    f.write("\n")

        # Usage remains the same
        results = {}
        model_name = self.args.model
        results[f'{model_name}'] = compute_metrics(preds, trues, criterion_mse, criterion_mae)
        results['Repeat'] = compute_metrics(naive_preds, trues, criterion_mse, criterion_mae)

        log_metrics(results)
        print("")
        print(self.model)
        
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'DLinear' or 'DLinear_FITS' or 'FITS' or 'FITS_DLinear' or 'FITS_p' or 'FITS_100' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'DLinear' or 'DLinear_FITS' or 'FITS' or 'FITS_DLinear' or 'FITS_p' or 'FITS_100' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
