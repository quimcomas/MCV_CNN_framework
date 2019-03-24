import sys
import subprocess
import time
import numpy as np
import os
import cv2 as cv
import operator
import math
import torch
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F

sys.path.append('../')
from simple_trainer_manager import SimpleTrainer
from metrics.metrics import compute_accuracy, compute_confusion_matrix, extract_stats_from_confm, compute_mIoU
from utils.plot import Compute_plot
from metrics.object_detection import Compute_kitti_AP

class Detection_Manager(SimpleTrainer):
    def __init__(self, cf, model):
        super(Detection_Manager, self).__init__(cf, model)

    class train(SimpleTrainer.train):
        def __init__(self, logger_stats, model, cf, validator, stats, msg):
            super(Detection_Manager.train, self).__init__(logger_stats, model, cf, validator, stats, msg)
            if self.cf.resume_experiment:
                self.msg.msg_stats_best = 'Best case [%s]: epoch = %d, mIoU = %.2f, acc= %.2f, loss = %.5f\n' % (
                    self.cf.save_condition, self.model.net.best_stats.epoch, 100 * self.model.net.best_stats.val.mIoU,
                    100 * self.model.net.best_stats.val.acc, self.model.net.best_stats.val.loss)

        def training_loop(self, epoch, train_loader, epoch_bar):
            # Train epoch
            for i, data in enumerate(train_loader):
                # Read Data
                inputs, loc_targets, cls_targets = data
                N, w, h, c = inputs.size()
                self.inputs = Variable(inputs).cuda()
                self.loc_targets = Variable(loc_targets).cuda()
                self.cls_targets = Variable(cls_targets).cuda()
                # Predict model
                self.model.optimizer.zero_grad()
                self.loc_preds, self.cls_preds = self.model.net(self.inputs)
                # Compute gradients
                self.compute_gradients()

                # Compute batch stats
                self.train_loss.update(float(self.loss[0].cpu().data[0]), N)
                # confm = compute_confusion_matrix(predictions, self.labels.cpu().data.numpy(), self.cf.num_classes,
                #                                  self.cf.void_class)
                # self.confm_list = map(operator.add, self.confm_list, confm)

                if self.cf.normalize_loss:
                    self.stats.train.loss = self.train_loss.avg
                else:
                    self.stats.train.loss = self.train_loss.avg

                if not self.cf.debug:
                    # Save stats
                    self.save_stats_batch((epoch - 1) * self.train_num_batches + i)

                    # Update epoch messages
                    self.update_epoch_messages(epoch_bar, self.global_bar, self.train_num_batches, epoch, i)

            # Save model without training
            if self.cf.epochs == 0:
                self.model.save_model()

        def validate_epoch(self, valid_set, valid_loader, early_Stopping, epoch, global_bar):

            if valid_set is not None and valid_loader is not None:
                # Set model in validation mode
                self.model.net.eval()
                self.model.net.training = False
                self.validator.start(valid_set, valid_loader, 'Epoch Validation', epoch,
                                     global_bar=global_bar, save_folder=self.cf.temp_folder)

                # Early stopping checking
                if self.cf.early_stopping:
                    early_Stopping.check(self.stats.train.loss, self.stats.val.AP)
                    if early_Stopping.stop == True:
                        self.stop = True

                # Set model in training mode
                self.model.net.training = True
                self.model.net.train()

        def compute_gradients(self):
            self.loss = self.model.loss(self.loc_preds, self.loc_targets, self.cls_preds, self.cls_targets)
            self.loss[0].backward()
            self.model.optimizer.step()

        def update_messages(self, epoch, epoch_time, new_best):
            # Update logger
            epoch_time = time.time() - epoch_time
            self.logger_stats.write('\t Epoch step finished: %ds \n' % (epoch_time))

            # Compute best stats
            self.msg.msg_stats_last = '\nLast epoch: loss = %.5f, mean_AP = %.2f' % (self.stats.val.loss, self.stats.val.AP)
            for label in range(len(self.cf.labels)):
                self.msg.msg_stats_last += ', AP_%s %.2f' % (self.cf.labels[label], self.stats.val.AP_perclass[label])
            if new_best:
                self.msg.msg_stats_best = '\n Best case [%s]: epoch = %d, avg_loss = %.5f' % (
                                          self.cf.save_condition,epoch, self.stats.val.loss)
                for label in range(len(self.cf.labels)):
                    self.msg.msg_stats_best += ', AP_%s %.2f' % (
                    self.cf.labels[label], self.stats.val.AP_perclass[label])
                # msg_confm = self.stats.val.get_confm_str()
                # self.logger_stats.write(msg_confm)
                # self.msg.msg_stats_best = self.msg.msg_stats_best + '\nConfusion matrix:\n' + msg_confm

        def update_epoch_messages(self, epoch_bar, global_bar, train_num_batches, epoch, batch):
            # Update progress bar
            epoch_bar.set_msg('loc_loss = %.5f, cls_loss = %.5f, batch_loss = %.5f, avg_loss = %.5f' %
                              (self.loss[1], self.loss[2], float(self.loss[0].cpu().data[0]), self.stats.train.loss))
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

        def compute_stats(self, confm_list, train_loss):
            # TP_list, TN_list, FP_list, FN_list = extract_stats_from_confm(confm_list)
            # mean_IoU = compute_mIoU(TP_list, FP_list, FN_list)
            # mean_accuracy = compute_accuracy_segmentation(TP_list, FN_list)
            # self.stats.train.acc = np.nanmean(mean_accuracy)
            # self.stats.train.mIoU_perclass = mean_IoU
            # self.stats.train.mIoU = np.nanmean(mean_IoU)
            if train_loss is not None:
                self.stats.val.loss = train_loss.avg

        def save_stats_epoch(self, epoch):
            # Save logger
            if epoch is not None:
                # Epoch loss tensorboard
                self.writer.add_scalar('losses/epoch', self.stats.train.loss, epoch)
                self.writer.add_scalar('metrics/accuracy', 100.*self.stats.train.acc, epoch)
                self.writer.add_scalar('metrics/mIoU', 100.*self.stats.train.mIoU, epoch)
                # conf_mat_img = confm_metrics2image(self.stats.train.get_confm_norm(), self.cf.labels)
                # self.writer.add_image('metrics/conf_matrix', conf_mat_img, epoch)

    class validation(SimpleTrainer.validation):
        def __init__(self, logger_stats, model, cf, stats, msg):
            super(Detection_Manager.validation, self).__init__(logger_stats, model, cf, stats, msg)

        def validation_loop(self, epoch, valid_loader, valid_set, bar, global_bar, save_folder=None):
            if epoch is None:
                if not os.path.exists(os.path.join(self.cf.predict_path_output,'data')):
                    os.makedirs(os.path.join(self.cf.predict_path_output,'data'))
            else:
                if not os.path.exists(os.path.join(save_folder,'data')):
                    os.makedirs(os.path.join(save_folder,'data'))
                val_epoch_images = open(os.path.join(save_folder, "val_epoch_images.txt"), 'w')
            for vi, data in enumerate(valid_loader):
                # Read data
                inputs, loc_targets, cls_targets, gt_path, size = data
                n_images, _, _, _ = inputs.size()
                img_name = gt_path[0].split('/')[-1].split('.')[0]
                inputs = Variable(inputs).cuda()
                loc_targets = Variable(loc_targets).cuda()
                cls_targets = Variable(cls_targets).cuda()
                if epoch is None:
                    f = open(os.path.join(self.cf.predict_path_output, 'data', img_name + ".txt"), 'w')
                else:
                    f = open(os.path.join(save_folder , 'data', img_name + ".txt"), 'w')
                # Predict model
                with torch.no_grad():
                    loc_preds, cls_preds = self.model.net(inputs)
                    # self.val_loss.update(float(self.model.loss(loc_preds, loc_targets, cls_preds,
                    #                                            cls_targets).cpu().data[0] / n_images), n_images)
                    box_preds, label_preds, score_preds = self.model.box_coder.decode(
                        loc_preds.cpu().data.squeeze(),
                        F.softmax(cls_preds.squeeze(), dim=1).cpu().data,
                        score_thresh=0.25)
                    if len(box_preds) > 0:
                        box_preds = box_preds.cpu().numpy()
                        label_preds = label_preds.cpu().numpy()
                        score_preds = score_preds.cpu().numpy()
                        # print(len(box_preds))
                        # print(label_preds.size)
                        # print(score_preds.size)
                        for det in range(len(box_preds)):
                            if score_preds[det] >= 0:
                                box_preds[det] = box_preds[det] / [size[0], size[1], size[0], size[1]]
                                f.write(self.cf.labels[label_preds[det]] + ' ' + str(0) + ' ' + str(0) + ' ' + '-10' + ' ' \
                                        + str(box_preds[det][0]) + ' ' + str(box_preds[det][1]) + ' ' \
                                        + str(box_preds[det][2]) + ' ' + str(box_preds[det][3]) + ' ' \
                                        + '-1 -1 -1 -1000 -1000 -1000 ' \
                                        + '-1000 ' \
                                        + str(score_preds[det]) + '\n'
                                        )
                    f.close()
                    if epoch is not None:
                        val_epoch_images.write(gt_path[0] + '\n')
                    # Compute batch stats
                    # self.val_loss.update(float(self.model.loss(outputs, gts).cpu().data[0] / n_images), n_images)
                    # confm = compute_confusion_matrix(predictions, gts.cpu().data.numpy(), self.cf.num_classes,
                    #                                  self.cf.void_class)
                    # confm_list = map(operator.add, confm_list, confm)
		
                # Save epoch stats
                # self.stats.val.conf_m = confm_list
                if not self.cf.normalize_loss:
                    self.stats.val.loss = self.val_loss.avg
                else:
                    self.stats.val.loss = self.val_loss.avg

                # Save predictions and generate overlaping
                # self.update_tensorboard(inputs.cpu(), gts.cpu(),
                #                         predictions, epoch, range(vi * self.cf.valid_batch_size,
                #                                                   vi * self.cf.valid_batch_size +
                #                                                   np.shape(predictions)[0]),
                #                         valid_set.num_images)

                # Update messages
                self.update_msg(bar, global_bar)
            # Calculate stats for detection
            # extern kitti evaluation call
            if epoch is not None:
                val_epoch_images.close()
                evaluation_bash_comand = "./devkit_kitti_txt/cpp/evaluate_object_txt %s %s %s" % (
                    self.cf.temp_folder, os.path.join(self.cf.temp_folder,"val_epoch_images.txt"), self.cf.temp_folder)
            else:
                evaluation_bash_comand = "./devkit_kitti_txt/cpp/evaluate_object_txt %s %s %s" % (
                    self.cf.predict_path_output, self.cf.valid_gt_txt, self.cf.predict_path_output)
            process = subprocess.Popen(evaluation_bash_comand, shell=True)
            process.communicate()


        def compute_stats(self, confm_list, val_loss):
            for label in self.cf.labels:
                scores = Compute_kitti_AP(os.path.join(self.cf.temp_folder,"stats_%s_detection.txt" %(label.lower())))
                # print scores
                self.stats.val.AP_perclass.append(scores[1])
            self.stats.val.AP = np.mean(np.asarray(self.stats.val.AP_perclass, dtype=np.float32))
            if val_loss is not None:
                self.stats.val.loss = val_loss.avg

        def save_stats(self, epoch):
            # Save logger
            if epoch is not None:
                # add log
                self.logger_stats.write('----------------- Epoch scores summary ------------------------- \n')
                self.logger_stats.write('[epoch %d], [val loss %.5f]' % (
                    epoch, self.stats.val.loss))
                for label in range(len(self.cf.labels)):
                    self.logger_stats.write(', [AP_%s %.2f]' % (self.cf.labels[label], self.stats.val.AP_perclass[label]))
                    # add scores to tensorboard
                    self.writer.add_scalar('metrics/AP_%s'%(self.cf.labels[label]),
                                           100.*self.stats.val.AP_perclass[label], epoch)
                self.logger_stats.write('\n---------------------------------------------------------------- \n')
                # add scores to tensorboard
                self.writer.add_scalar('losses/epoch',  self.stats.val.loss, epoch)

            else:
                self.logger_stats.write('----------------- Scores summary -------------------- \n')
                self.logger_stats.write('[val loss %.5f]' % (
                    self.stats.val.loss))
                for label in range(len(self.cf.labels)):
                    self.logger_stats.write(', [AP_%s %.2f]' % (self.cf.labels[label], self.stats.val.AP_perclass[label]))
                self.logger_stats.write('\n---------------------------------------------------------------- \n')

        def update_msg(self, bar, global_bar):

            # self.compute_stats(np.asarray(self.stats.val.conf_m), None)
            # bar.set_msg(', mIoU: %.02f' % (100.*np.nanmean(self.stats.val.mIoU)))

            if global_bar==None:
                # Update progress bar
                bar.update()
            else:
                self.msg.eval_str = '\n' + bar.get_message(step=True)
                global_bar.set_msg(self.msg.accum_str + self.msg.last_str + self.msg.msg_stats_last + self.msg.msg_stats_best + self.msg.eval_str)
                global_bar.update()

        def update_tensorboard(self,inputs,gts,predictions,epoch,indexes,val_len):
            if epoch is not None and self.cf.color_map is not None:
                save_img(self.writer, inputs, gts, predictions, epoch, indexes, self.cf.predict_to_save, val_len,
                        self.cf.color_map, self.cf.labels, self.cf.void_class, n_legend_rows=3)

    class predict(SimpleTrainer.predict):
        def __init__(self, logger_stats, model, cf):
            super(Detection_Manager.predict, self).__init__(logger_stats, model, cf)

        def write_results(self, predictions, img_name, img_shape):
            path = os.path.join(self.cf.predict_path_output, img_name[0])
            predictions = predictions[0]
            predictions = Image.fromarray(predictions.astype(np.uint8))
            if self.cf.resize_image_test is not None:
                predictions = predictions.resize((self.cf.original_size[1],
                                                  self.cf.original_size[0]), resample=Image.BILINEAR)
            predictions = np.array(predictions)
            cv.imwrite(path, predictions)
