import json

# Save the printf to a log file
class Logger(object):
    def __init__(self, log_file):
        self.file = log_file
        log = open(log_file, 'a')
        log.close()

    def write(self, message):
        with open(self.file, 'a') as log:
            log.write(message)

    def create_dict(self, stats, epoch):
        if epoch is None:
            epoch = 0
        mIoU_class_list = stats.mIoU_perclass if stats.mIoU_perclass==[] else stats.mIoU_perclass.tolist()
        acc_class_list = stats.acc_perclass if stats.acc_perclass == [] else stats.acc_perclass.tolist()
        precision_class_list = stats.precision_perclass if stats.precision_perclass == [] else stats.precision_perclass.tolist()
        recall_class_list = stats.recall_perclass if stats.recall_perclass == [] else stats.recall_perclass.tolist()
        f1score_class_list = stats.f1score_perclass if stats.f1score_perclass == [] else stats.f1score_perclass.tolist()
        conf_m = [[0 if conf_row.sum()==0 else el/conf_row.sum() for el in conf_row] for conf_row in stats.conf_m]
        stats_dic = {
                     'epoch': epoch, 'loss': stats.loss,
                     'mIoU': stats.mIoU, 'acc': stats.acc,
                     'precision': stats.precision,'recall': stats.recall, 'f1score': stats.f1score,
                     'conf_m': conf_m,'mIoU_perclass': mIoU_class_list,'acc_perclass': acc_class_list,
                     'precision_perclass': precision_class_list,'recall_perclass': recall_class_list,
                     'f1score_perclass': f1score_class_list
                     }
        return stats_dic

    def write_stat(self, stats, epoch, file):
        with open(file, "w") as json_file:
            stats_dict = self.create_dict(stats, epoch)
            json.dump(stats_dict, json_file, indent=1)

    def write_best_stats(self, stats, epoch, file):
        with open(file, "w") as json_file:
            train_stats_dict = self.create_dict(stats.train, epoch)
            val_stats_dict = self.create_dict(stats.val, epoch)
            stats_list = [train_stats_dict,val_stats_dict]
            json.dump(stats_list, json_file, indent=1)