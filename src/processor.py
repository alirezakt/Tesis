from datetime import datetime
import torch
import torch.nn as nn
from .bstar import STAR
from .lstm_star import LSTM_STAR
from .bstar_AttentionGM import STAR_GM
from .srlstm import SR_LSTM
from .utils import *
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import pickle


class processor(object):
    """
    The processor class is responsible for training and testing the STAR model for air traffic prediction.

    Args:
        args (object): An object containing the arguments for the processor.

    Attributes:
        args (object): An object containing the arguments for the processor.
        dataloader (Trajectory_Dataloader): An instance of the Trajectory_Dataloader class for loading trajectory data.
        net (STAR): An instance of the STAR class representing the neural network model.
        optimizer (torch.optim.Adam): The optimizer used for training the model.
        criterion (nn.MSELoss): The loss function used for training the model.
        best_ade (float): The best average displacement error achieved during training.
        best_fde (float): The best final displacement error achieved during training.
        best_epoch (int): The epoch number at which the best final displacement error was achieved.

    Methods:
        save_model(epoch): Saves the model at a given epoch.
        load_model(): Loads a saved model if specified.
        set_optimizer(): Sets the optimizer for training the model.
        test(): Performs testing on the trained model.
        train(): Performs training on the model.
        train_epoch(epoch, best_test=False): Performs one epoch of training.
        test_epoch(): Performs one epoch of testing.
    """
    def __init__(self, args):

        torch.autograd.set_detect_anomaly(True)
        torch._dynamo.config.capture_scalar_outputs = True
        torch. cuda. set_device(0)

        self.args = args

        self.dataloader = Trajectory_Dataloader(args)
        self.net = LSTM_STAR(args, dropout_prob=self.args.gwo_dropout_rate, nlayers=self.args.gwo_transformer_layers, nhead=self.args.gwo_attention_head, bilstm_hidden_size=self.args.gwo_lstm_layers)
        #self.net = SR_LSTM(args)
        self.set_optimizer()
        milestones = [self.args.num_epochs // 2]
        gamma = self.args.gwo_scheduler_gamma
        self.scheduler = MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(device)
        
        #self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', verbose=True)

        if self.args.using_cuda:
            #self.net = torch.compile(self.net.cuda())
            print('using cuda')
        else:
            #self.net = torch.compile(self.net.cpu())
            print('using cpu')

        if not os.path.isdir(self.args.model_dir):
            os.mkdir(self.args.model_dir)

        self.net_file = open(os.path.join(self.args.model_dir, 'net.txt'), 'a+')
        self.net_file.write(str(self.net))
        self.net_file.close()
        self.log_file_curve = open(os.path.join(self.args.model_dir, 'log_curve.txt'), 'a+')

        self.best_ade = 100
        self.best_fde = 100
        self.best_epoch = -1

    def save_model(self, epoch):

        model_path = self.args.save_dir + '/' + self.args.train_model + '/' + self.args.train_model + '_' + \
                     str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, model_path)

    def load_model(self):

        if self.args.load_model is not None:
            self.args.model_save_path = self.args.save_dir + '/' + self.args.train_model + '/' + self.args.train_model + '_' + \
                                        str(self.args.load_model) + '.tar'
            print(self.args.model_save_path)
            if os.path.isfile(self.args.model_save_path):
                print('Loading checkpoint')
                checkpoint = torch.load(self.args.model_save_path)
                model_epoch = checkpoint['epoch']
                self.net.load_state_dict(checkpoint['state_dict'])
                print('Loaded checkpoint at epoch', model_epoch)

    def set_optimizer(self):

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.learning_rate, weight_decay=self.args.gwo_weight_decay)
        self.criterion = nn.MSELoss(reduction='none')

    def test(self):

        print('Testing begin')
        self.load_model()
        self.net.eval()
        test_error, test_final_error, output_all = self.test_epoch()
        print('Set: {}, epoch: {},test_error: {} test_final_error: {}'.format(self.args.test_set, self.args.load_model,
                                                                                       test_error, test_final_error))

    def train(self):
        print('Training begin at {}' .format(datetime.now().strftime("%H:%M")))
        test_error, test_final_error = 0, 0
        for epoch in range(self.args.num_epochs):

            self.net.train()
            train_loss = self.train_epoch(epoch, best_test=False)

            if epoch >= self.args.start_test:
                self.net.eval()
                test_error, test_final_error, output_all = self.test_epoch()
                self.best_ade = test_error if test_final_error < self.best_fde else self.best_ade

                # save the output from best test results
                if test_final_error < self.best_fde:
                    f = open(os.path.join(self.args.save_dir, 'all_test_output_{}.cpkl'.format(epoch)), "wb")
                    pickle.dump(output_all, f, protocol=2)
                    f.close()

                self.best_epoch = epoch if test_final_error < self.best_fde else self.best_epoch
                self.best_fde = test_final_error if test_final_error < self.best_fde else self.best_fde
                self.save_model(epoch)

            formatted_loss = f"{train_loss:.7f}"
            formatted_test_error = f"{test_error:.4f}"
            formatted_test_final_error = f"{test_final_error:.4f}"
            self.log_file_curve.write(
                datetime.now().strftime("%H:%M") + ' ' + str(epoch) + ',' + str(formatted_loss) + ',' + str(formatted_test_error) + ',' + 
                str(formatted_test_final_error) + ',' + str(self.optimizer.param_groups[0]['lr']) + '\n')

            #if epoch % 2 == 0:
            #self.log_file_curve.close()
            self.log_file_curve.flush()
            #self.log_file_curve = open(os.path.join(self.args.model_dir, 'log_curve.txt'), 'a+')

            if epoch >= self.args.start_test:
                print('{}  epoch {}, train_loss={:.7f}, lr={:.5f}, ADE={:.4f}, FDE={:.4f}, Best_ADE={:.4f}, Best_FDE={:.4f} at Epoch {}'
                        .format(datetime.now().strftime("%H:%M"), epoch, train_loss, self.optimizer.param_groups[0]['lr'], test_error, test_final_error, self.best_ade, self.best_fde,
                                self.best_epoch))
            else:
                print('{}  epoch {}, train_loss={:.7f}, lr={:.5f}'
                        .format(datetime.now().strftime("%H:%M"), epoch, train_loss, self.optimizer.param_groups[0]['lr']))
       
        total_params = sum(p.numel() for p in self.net.parameters())
        print("Number of Parameters in the Sample Model: {:,}".format(total_params))
        print('Training finished at {}' .format(datetime.now().strftime("%H:%M")))

    def train_epoch(self, epoch, best_test=False):

        self.dataloader.reset_batch_pointer(set='train', valid=False)
        loss_epoch = 0        

        for batch in range(self.dataloader.trainbatchnums):
            start = time.time()
            inputs, batch_id = self.dataloader.get_train_batch(batch)
            inputs = tuple([torch.Tensor(i) for i in inputs])
            inputs = tuple([i.cuda() for i in inputs])

            loss = torch.zeros(1).cuda()
            batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs
            inputs_forward = batch_abs[:-1], batch_norm[:-1], shift_value[:-1], seq_list[:-1], nei_list[:-1], nei_num[:-1], batch_pednum

            self.optimizer.zero_grad()

            outputs = self.net.forward(inputs_forward, iftest=False)

            lossmask, num = getLossMask(outputs, seq_list[0], seq_list[1:], using_cuda=self.args.using_cuda)
            a = self.criterion(outputs, batch_norm[1:, :, :3])
            loss_o = torch.sum(a, dim=2)

            loss += (torch.sum(loss_o * lossmask / num))
            loss_epoch += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.gwo_clip)

            self.optimizer.step()

            end = time.time()

            if batch % self.args.show_step == 0 and self.args.ifshow_detail:
                print('train-{}/{} (epoch {}), train_loss = {:.5f}, time/batch = {:.5f} '.format(batch,
                                                                                                 self.dataloader.trainbatchnums,
                                                                                                 epoch,
                                                                                                 loss.item(),
                                                                                                 end - start))
        
        train_loss_epoch = loss_epoch / self.dataloader.trainbatchnums
        self.scheduler.step()
        return train_loss_epoch

# Decorator to ensure that no gradients are computed during this function
    @torch.no_grad()
    def test_epoch(self):
        self.dataloader.reset_batch_pointer(set='test')
        error_epoch, final_error_epoch = 0, 0,
        error_cnt_epoch, final_error_cnt_epoch = 1e-5, 1e-5 #values 1e-5 is shorthand for 0.00001 in scientific notation
        # Lists to store mean and variance of the model's predictions
        all_output_mean, all_output_var = [], []

        #tqdm is a Python library that provides a progress bar for loops and other iterable computations
        for batch in tqdm(range(self.dataloader.testbatchnums)):

            inputs, batch_id = self.dataloader.get_test_batch(batch)
            inputs = tuple([torch.Tensor(i) for i in inputs])

            # Move inputs to GPU if using CUDA
            if self.args.using_cuda:
                inputs = tuple([i.cuda() for i in inputs])
                
            # Unpack inputs
            #batch_abs: Absolute coordinates of pedestrians. It likely has the shape [sequence_length, num_pedestrians, 2] where sequence_length is the length of the trajectory sequence, num_pedestrians is the number of pedestrians, and 2 corresponds to the x and y coordinates.
            #batch_norm: Normalized coordinates of pedestrians. Similar to batch_abs, it represents the normalized coordinates of pedestrians in the trajectory sequence.
            #shift_value: Shift values representing the changes in pedestrian positions over time. It is used during testing for predicting future trajectory points.
            #seq_list: A list indicating the length of the trajectory sequence for each pedestrian. It's a tensor of shape [sequence_length].
            #nei_list: A tensor representing the neighborhood information of pedestrians. It likely has the shape [sequence_length, num_pedestrians, max_neighbors] where max_neighbors is the maximum number of neighboring pedestrians considered.
            #nei_num: A tensor representing the number of neighbors for each pedestrian in each time step. It has the shape [sequence_length, num_pedestrians].
            #batch_pednum: A list containing the number of pedestrians in different scenes for a batch.
            batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs
            # Prepare inputs for the model
            inputs_forward = batch_abs[:-1], batch_norm[:-1], shift_value[:-1], seq_list[:-1], nei_list[:-1], nei_num[:-1], batch_pednum
            # List to store predictions for multiple samples (Monte Carlo dropout)
            all_output = []
            # Perform multiple forward passes with the model for uncertainty estimation
            for i in range(self.args.sample_num):
                outputs_infer = self.net.forward(inputs_forward, iftest=True)
                all_output.append(outputs_infer)

            self.optimizer.zero_grad()
            # Stack the predictions along a new dimension
            all_output = torch.stack(all_output)  # output
            # Calculate loss mask and the number of valid predictions
            lossmask, num = getLossMask(all_output, seq_list[0], seq_list[1:], using_cuda=self.args.using_cuda)
            # Calculate errors for evaluation
            error, error_cnt, final_error, final_error_cnt = L2forTestS(all_output, batch_norm[1:, :, :3],
                                                                        self.args.obs_length, lossmask)
            # Append mean and variance of predictions to lists
            all_output_mean.append(torch.mean(all_output, 0))
            all_output_var.append(torch.var(all_output, 0))
            temp = torch.mean(all_output, 0).cpu()
            # Update error metrics
            error_epoch += error
            error_cnt_epoch += error_cnt
            final_error_epoch += final_error
            final_error_cnt_epoch += final_error_cnt

        # Create a dictionary to store all the test outputs (mean and variance)
        output_all = dict(zip(['test_mean', 'test_var'], [all_output_mean, all_output_var]))
        # Return normalized errors and the output dictionary
        return error_epoch / error_cnt_epoch, final_error_epoch / final_error_cnt_epoch, output_all
