import math
import sys
import time
import numpy as np
import torch
import operator
import os
from torch.autograd import Variable

sys.path.append('../')
from utils.tools import AverageMeter, Early_Stopping
from utils.ProgressBar import ProgressBar
from utils.logger import Logger
from utils.statistics import Statistics
from utils.messages import Messages
from metrics.metrics import compute_accuracy, compute_confusion_matrix, extract_stats_from_confm, compute_mIoU
from tensorboardX import SummaryWriter

class SimpleTrainer(object):
    def __init__(self, cf, model):
        self.cf = cf
        self.model = model
        self.logger_stats = Logger(cf.log_file_stats)
        self.stats = Statistics()
        self.msg = Messages()
        self.validator = self.validation(self.logger_stats, self.model, cf, self.stats, self.msg)
        self.trainer = self.train(self.logger_stats, self.model, cf, self.validator, self.stats, self.msg)
        self.predictor = self.predict(self.logger_stats, self.model, cf)

    class train(object):
        def __init__(self, logger_stats, model, cf, validator, stats, msg):
            # Initialize training variables
            self.logger_stats = logger_stats
            self.model = model
            self.cf = cf
            self.validator = validator
            self.logger_stats.write('\n- Starting train <--- \n')
            self.curr_epoch = 1 if self.model.best_stats.epoch==0 else self.model.best_stats.epoch
            self.stop = False
            self.stats = stats
            self.best_acc = 0
            self.msg = msg
            self.loss = None
            self.outputs = None
            self.labels = None
            self.writer = SummaryWriter(os.path.join(cf.tensorboard_path,'train'))

        def start(self, train_loader, train_set, valid_set=None,
                  valid_loader=None):
            self.train_num_batches = math.ceil(train_set.num_images / float(self.cf.train_batch_size))
            self.val_num_batches = 0 if valid_set is None else math.ceil(valid_set.num_images / \
                                                                    float(self.cf.valid_batch_size))
            # Define early stopping control
            if self.cf.early_stopping:
                early_Stopping = Early_Stopping(self.cf)
            else:
                early_Stopping = None

            prev_msg = '\nTotal estimated training time...\n'
            self.global_bar = ProgressBar((self.cf.epochs+1-self.curr_epoch)*(self.train_num_batches+self.val_num_batches), lenBar=20)
            self.global_bar.set_prev_msg(prev_msg)


            # Train process
            for epoch in range(self.curr_epoch, self.cf.epochs + 1):
                # Shuffle train data
                train_set.update_indexes()

                # Initialize logger
                epoch_time = time.time()
                self.logger_stats.write('\t ------ Epoch: ' + str(epoch) + ' ------ \n')

                # Initialize epoch progress bar
                self.msg.accum_str = '\n\nEpoch %d/%d estimated time...\n' % \
                                     (epoch, self.cf.epochs)
                epoch_bar = ProgressBar(self.train_num_batches, lenBar=20)
                epoch_bar.update(show=False)

                # Initialize stats
                self.stats.epoch = epoch
                self.train_loss = AverageMeter()
                self.confm_list = np.zeros((self.cf.num_classes, self.cf.num_classes))

                # Train epoch
                self.training_loop(epoch, train_loader, epoch_bar)

                # Save stats
                self.stats.train.conf_m = self.confm_list
                self.compute_stats(np.asarray(self.confm_list), self.train_loss)
                self.save_stats_epoch(epoch)
                self.logger_stats.write_stat(self.stats.train, epoch, os.path.join(self.cf.train_json_path,
                                                                                'train_epoch_' + str(epoch) + '.json'))

                # Validate epoch
                self.validate_epoch(valid_set, valid_loader, early_Stopping, epoch, self.global_bar)

                # Update scheduler
                if self.model.scheduler is not None:
                    self.model.scheduler.step(self.stats.val.loss)

                # Saving model if score improvement
                new_best = self.model.save(self.stats)
                if new_best:
                    self.logger_stats.write_best_stats(self.stats, epoch, self.cf.best_json_file)

                # Update display values
                self.update_messages(epoch, epoch_time, new_best)

                if self.stop:
                    return

            # Save model without training
            if self.cf.epochs == 0:
                self.model.save_model()

        def training_loop(self, epoch, train_loader, epoch_bar):
            # Train epoch
            for i, data in enumerate(train_loader):
                # Read Data
                inputs, labels = data

                N, w, h, c = inputs.size()
                inputs = Variable(inputs).cuda()
                self.inputs = inputs
                self.labels = Variable(labels).cuda()

                # Predict model
                self.model.optimizer.zero_grad()
                self.outputs = self.model.net(inputs)
                predictions = self.outputs.data.max(1)[1].cpu().numpy()

                # Compute gradients
                self.compute_gradients()

                # Compute batch stats
                self.train_loss.update(float(self.loss.cpu().item()), N)
                confm = compute_confusion_matrix(predictions, self.labels.cpu().data.numpy(), self.cf.num_classes,
                                                 self.cf.void_class)
                self.confm_list = map(operator.add, self.confm_list, confm)

                if self.cf.normalize_loss:
                    self.stats.train.loss = self.train_loss.avg
                else:
                    self.stats.train.loss = self.train_loss.avg

                if not self.cf.debug:
                    # Save stats
                    self.save_stats_batch((epoch - 1) * self.train_num_batches + i)

                    # Update epoch messages
                    self.update_epoch_messages(epoch_bar, self.global_bar, self.train_num_batches, epoch, i)

        def save_stats_epoch(self, epoch):
            # Save logger
            if epoch is not None:
                # Epoch loss tensorboard
                self.writer.add_scalar('losses/epoch', self.stats.train.loss, epoch)
                self.writer.add_scalar('metrics/accuracy', 100.*self.stats.train.acc, epoch)

        def save_stats_batch(self, batch):
            # Save logger
            if batch is not None:
                self.writer.add_scalar('losses/batch', self.stats.train.loss, batch)

        def compute_gradients(self):
            self.loss = self.model.loss(self.outputs, self.labels)
            self.loss.backward()
            self.model.optimizer.step()

        def compute_stats(self, confm_list, train_loss):
            TP_list, TN_list, FP_list, FN_list = extract_stats_from_confm(confm_list)
            mean_accuracy = compute_accuracy(TP_list, TN_list, FP_list, FN_list)
            self.stats.train.acc = np.nanmean(mean_accuracy)
            self.stats.train.loss = float(train_loss.avg.cpu().data)

        def validate_epoch(self,valid_set, valid_loader, early_Stopping, epoch, global_bar):

            if valid_set is not None and valid_loader is not None:
                # Set model in validation mode
                self.model.net.eval()

                self.validator.start(valid_set, valid_loader, 'Epoch Validation', epoch, global_bar=global_bar)

                # Early stopping checking
                if self.cf.early_stopping:
                    early_Stopping.check(self.stats.train.loss, self.stats.val.loss, self.stats.val.mIoU,
                                         self.stats.val.acc)
                    if early_Stopping.stop == True:
                        self.stop=True
                # Set model in training mode
                self.model.net.train()


        def update_messages(self, epoch, epoch_time):
            # Update logger
            epoch_time = time.time() - epoch_time
            self.logger_stats.write('\t Epoch step finished: %ds \n' % (epoch_time))

            # Compute best stats
            self.msg.msg_stats_last = '\nLast epoch: acc = %.2f, loss = %.5f\n' % (100 * self.stats.val.acc,
                                                                                   self.stats.val.loss)
            if self.best_acc < self.stats.val.acc:
                self.msg.msg_stats_best = 'Best case: epoch = %d, acc = %.2f, loss = %.5f\n' % (
                epoch, 100 * self.stats.val.acc, self.stats.val.loss)

                msg_confm = self.stats.val.get_confm_str()
                self.logger_stats.write(msg_confm)
                self.msg.msg_stats_best = self.msg.msg_stats_best + msg_confm

                self.best_acc = self.stats.val.acc

        def update_epoch_messages(self, epoch_bar, global_bar, train_num_batches, epoch, batch):
            # Update progress bar
            epoch_bar.set_msg('loss = %.5f' % self.stats.train.loss)
            self.msg.last_str = epoch_bar.get_message(step=True)
            global_bar.set_msg(self.msg.accum_str + self.msg.last_str + self.msg.msg_stats_last + \
                               self.msg.msg_stats_best)
            global_bar.update()

            # writer.add_scalar('train_loss', train_loss.avg, curr_iter)

            # Display progress
            curr_iter = (epoch - 1) * train_num_batches + batch + 1
            if (batch + 1) % math.ceil(train_num_batches / 20.) == 0:
                self.logger_stats.write('[Global iteration %d], [iter %d / %d], [train loss %.5f] \n' % (
                    curr_iter, batch + 1, train_num_batches, self.stats.train.loss))

    class validation(object):
        def __init__(self, logger_stats, model, cf, stats, msg):
            # Initialize validation variables
            self.logger_stats = logger_stats
            self.model = model
            self.cf = cf
            self.stats = stats
            self.msg = msg
            self.writer = SummaryWriter(os.path.join(cf.tensorboard_path, 'validation'))

        def start(self, valid_set, valid_loader, mode='Validation', epoch=None, global_bar=None, save_folder=None):
            confm_list = np.zeros((self.cf.num_classes,self.cf.num_classes))

            self.val_loss = AverageMeter()

            # Initialize epoch progress bar
            val_num_batches = math.ceil(valid_set.num_images / float(self.cf.valid_batch_size))
            prev_msg = '\n' + mode + ' estimated time...\n'
            bar = ProgressBar(val_num_batches, lenBar=20)
            bar.set_prev_msg(prev_msg)
            bar.update(show=False)

            # Validate model
            if self.cf.problem_type == 'detection':
                self.validation_loop(epoch, valid_loader, valid_set, bar, global_bar, save_folder)
            else:
                self.validation_loop(epoch, valid_loader, valid_set, bar, global_bar, confm_list)

            # Compute stats
            self.compute_stats(np.asarray(self.stats.val.conf_m), self.val_loss)

            # Save stats
            self.save_stats(epoch)
            if mode == 'Epoch Validation':
                self.logger_stats.write_stat(self.stats.train, epoch,
                                            os.path.join(self.cf.train_json_path,'valid_epoch_' + str(epoch) + '.json'))
            elif mode == 'Validation':
                self.logger_stats.write_stat(self.stats.val, epoch, self.cf.val_json_file)
            elif mode == 'Test':
                self.logger_stats.write_stat(self.stats.test, epoch, self.cf.test_json_file)

        def validation_loop(self, epoch, valid_loader, valid_set, bar, global_bar, confm_list):
            for vi, data in enumerate(valid_loader):
                # Read data
                inputs, gts = data
                n_images, w, h, c = inputs.size()
                inputs = Variable(inputs).cuda()
                gts = Variable(gts).cuda()

                # Predict model
                with torch.no_grad():
                    outputs = self.model.net(inputs)
                    predictions = outputs.data.max(1)[1].cpu().numpy()

                    # Compute batch stats
                    self.val_loss.update(float(self.model.loss(outputs, gts).cpu().item() / n_images), n_images)
                    confm = compute_confusion_matrix(predictions, gts.cpu().data.numpy(), self.cf.num_classes,
                                                     self.cf.void_class)
                    confm_list = map(operator.add, confm_list, confm)

                # Save epoch stats
                self.stats.val.conf_m = confm_list
                if not self.cf.normalize_loss:
                    self.stats.val.loss = self.val_loss.avg
                else:
                    self.stats.val.loss = self.val_loss.avg

                # Save predictions and generate overlaping
                self.update_tensorboard(inputs.cpu(), gts.cpu(),
                                        predictions, epoch, range(vi * self.cf.valid_batch_size,
                                                                  vi * self.cf.valid_batch_size +
                                                                  np.shape(predictions)[0]),
                                        valid_set.num_images)

                # Update messages
                self.update_msg(bar, global_bar)

        def update_tensorboard(self,inputs,gts,predictions,epoch,indexes,val_len):
            pass


        def update_msg(self, bar, global_bar):
            if global_bar==None:
                # Update progress bar
                bar.update()
            else:
                self.msg.eval_str = '\n' + bar.get_message(step=True)
                global_bar.set_msg(self.msg.accum_str + self.msg.last_str + self.msg.msg_stats_last + \
                                   self.msg.msg_stats_best + self.msg.eval_str)
                global_bar.update()

        def compute_stats(self, confm_list, val_loss):
            TP_list, TN_list, FP_list, FN_list = extract_stats_from_confm(confm_list)
            mean_accuracy = compute_accuracy(TP_list, TN_list, FP_list, FN_list)
            self.stats.val.acc = np.nanmean(mean_accuracy)
            self.stats.val.loss = val_loss.avg

        def save_stats(self, epoch):
            # Save logger
            if epoch is not None:
                self.logger_stats.write('----------------- Epoch scores summary ------------------------- \n')
                self.logger_stats.write('[epoch %d], [val loss %.5f], [acc %.2f] \n' % (
                    epoch, self.stats.val.loss, 100*self.stats.val.acc))
                self.logger_stats.write('---------------------------------------------------------------- \n')
            else:
                self.logger_stats.write('----------------- Scores summary -------------------- \n')
                self.logger_stats.write('[val loss %.5f], [acc %.2f] \n' % (
                    self.stats.val.loss, 100 * self.stats.val.acc))
                self.logger_stats.write('---------------------------------------------------------------- \n')


    class predict(object):
        def __init__(self, logger_stats, model, cf):
            self.logger_stats = logger_stats
            self.model = model
            self.cf = cf

        def start(self, dataloader):
            self.model.net.eval()

            for vi, data in enumerate(dataloader):
                inputs, img_name, img_shape = data

                inputs = Variable(inputs).cuda()
                with torch.no_grad():
                    outputs = self.model.net(inputs)
                    predictions = outputs.data.max(1)[1].cpu().numpy()

                    self.write_results(predictions, img_name, img_shape)

                # self.logger_stats.write('%d / %d \n' % (vi + 1, len(dataloader)))
                print('%d / %d' % (vi + 1, len(dataloader)))
        def write_results(self,predictions, img_name):
            pass
