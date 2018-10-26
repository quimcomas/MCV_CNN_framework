import sys
import time
import numpy as np
import os
import cv2 as cv
import operator
import torch
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F

sys.path.append('../')
from simple_trainer_manager import SimpleTrainer
from metrics.metrics import compute_accuracy, compute_confusion_matrix, extract_stats_from_confm, compute_mIoU

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
                self.train_loss.update(float(self.loss.cpu().data[0]), N)
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
                self.validator.start(valid_set, valid_loader, 'Epoch Validation', epoch, global_bar=global_bar)

                # Early stopping checking
                if self.cf.early_stopping:
                    early_Stopping.check(self.stats.train.loss, self.stats.val.loss, self.stats.val.mIoU,
                                         self.stats.val.acc)
                    if early_Stopping.stop == True:
                        self.stop = True

                # Set model in training mode
                self.model.net.training = True
                self.model.net.train()

        def compute_gradients(self):
            self.loss = self.model.loss(self.loc_preds, self.loc_targets, self.cls_preds, self.cls_targets)
            self.loss.backward()
            self.model.optimizer.step()

        def update_messages(self, epoch, epoch_time, new_best):
            # Update logger
            epoch_time = time.time() - epoch_time
            self.logger_stats.write('\t Epoch step finished: %ds \n' % (epoch_time))

            # Compute best stats
            self.msg.msg_stats_last = '\nLast epoch: mIoU = %.2f, acc= %.2f, loss = %.5f\n' % (
            100 * self.stats.val.mIoU, 100 * self.stats.val.acc, self.stats.val.loss)
            if new_best:
                self.msg.msg_stats_best = 'Best case [%s]: epoch = %d, mIoU = %.2f, acc= %.2f, loss = %.5f\n' % (
                                          self.cf.save_condition,epoch, 100 * self.stats.val.mIoU,
                                          100 * self.stats.val.acc, self.stats.val.loss)
                msg_confm = self.stats.val.get_confm_str()
                self.logger_stats.write(msg_confm)
                self.msg.msg_stats_best = self.msg.msg_stats_best + '\nConfusion matrix:\n' + msg_confm

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

        def validation_loop(self, epoch, valid_loader, valid_set, bar, global_bar):
            for vi, data in enumerate(valid_loader):
                # Read data
                inputs, loc_targets, cls_targets = data
                n_images, w, h, c = inputs.size()
                inputs = Variable(inputs).cuda()
                print(inputs.size())
                # Predict model
                with torch.no_grad():
                    loc_preds, cls_preds = self.model.net(inputs)
                    print(loc_preds.size())
                    print(cls_preds.size())
                    # predictions = outputs.data.max(1)[1].cpu().numpy()
                    box_preds, label_preds, score_preds = self.model.box_coder.decode(
                        loc_preds.cpu().data.squeeze(),
                        F.softmax(cls_preds.squeeze(), dim=1).cpu().data,
                        score_thresh=0.5)
                    if len(box_preds) > 0:
                        box_preds = box_preds.cpu().numpy()
                        label_preds = label_preds.cpu().numpy()
                        score_preds = score_preds.cpu().numpy()
                        print(box_preds.size)
                        print(label_preds.size)
                        print(score_preds.size)
                    exit(-1)
                    # Compute batch stats
                    self.val_loss.update(float(self.model.loss(outputs, gts).cpu().data[0] / n_images), n_images)
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

        def compute_stats(self, confm_list, val_loss):
            TP_list, TN_list, FP_list, FN_list = extract_stats_from_confm(confm_list)
            mean_IoU = compute_mIoU(TP_list, FP_list, FN_list)
            mean_accuracy = compute_accuracy_segmentation(TP_list, FN_list)
            self.stats.val.acc = np.nanmean(mean_accuracy)
            self.stats.val.mIoU_perclass = mean_IoU
            self.stats.val.mIoU = np.nanmean(mean_IoU)
            if val_loss is not None:
                self.stats.val.loss = val_loss.avg

        def save_stats(self, epoch):
            # Save logger
            if epoch is not None:
                # add log
                self.logger_stats.write('----------------- Epoch scores summary ------------------------- \n')
                self.logger_stats.write('[epoch %d], [val loss %.5f], [acc %.2f], [mean_IoU %.2f] \n' % (
                    epoch, self.stats.val.loss, 100*self.stats.val.acc, 100*self.stats.val.mIoU))
                self.logger_stats.write('---------------------------------------------------------------- \n')

                # add scores to tensorboard
                self.writer.add_scalar('losses/epoch',  self.stats.val.loss, epoch)
                self.writer.add_scalar('metrics/accuracy', 100.*self.stats.val.acc, epoch)
                self.writer.add_scalar('metrics/mIoU', 100.*self.stats.val.mIoU, epoch)
                conf_mat_img = confm_metrics2image(self.stats.val.get_confm_norm(), self.cf.labels)
                self.writer.add_image('metrics/conf_matrix', conf_mat_img, epoch)
            else:
                self.logger_stats.write('----------------- Scores summary -------------------- \n')
                self.logger_stats.write('[val loss %.5f], [acc %.2f], [mean_IoU %.2f]\n' % (
                    self.stats.val.loss, 100 * self.stats.val.acc, 100 * self.stats.val.mIoU))
                self.logger_stats.write('---------------------------------------------------------------- \n')

        def update_msg(self, bar, global_bar):

            self.compute_stats(np.asarray(self.stats.val.conf_m), None)
            bar.set_msg(', mIoU: %.02f' % (100.*np.nanmean(self.stats.val.mIoU)))

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
