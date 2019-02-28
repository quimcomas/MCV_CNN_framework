# System imports
import os
import sys
from glob import glob

# glob imports
import numpy as np

# matlplot imports
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def comute_mAP_splitted(path='/data/104-1/Experiments/Faster_RCNN/models', dataset='kitti', net='res101',
                        classes=['car', 'pedestrian'], scores=['easy', 'moderate', 'hard'], nThresholds=41,
                        subeval=None):
    nClasses = len(classes)
    nScores = len(scores)

    if subeval is not None:
        print('evaluation file: %s' % (subeval))
        path_prefix = os.path.join(path, 'test', dataset, subeval, net)
    else:
        path_prefix = os.path.join(path, 'test', dataset, net)
    epochs = glob(os.path.join(path_prefix, 'model_iter*/plot'))

    epochs = np.sort(np.asarray([int(epoch.split('/')[-2].split('_')[1].split('r')[1]) for epoch in epochs]))

    if len(epochs) == 0:
        return []

    nEpochs = max(epochs)

    matrixScores = -1 * np.ones((nEpochs, nClasses, nScores, nThresholds))

    for epoch in epochs:
        for indx_class, eval_class in enumerate(classes):
            if os.path.exists(
                    os.path.join(path_prefix, 'model_iter' + str(epoch), 'plot', eval_class + '_detection.txt')):
                filename = os.path.join(path_prefix, 'model_iter' + str(epoch), 'plot',
                                        eval_class + '_detection.txt')
                with open(filename, 'r') as f:
                    vLines = f.readlines()
                    vLines = np.asarray([[float(number) for number in line.rstrip().split(' ')] for line in vLines])
                    vLines = np.transpose(vLines)
                matrixScores[epoch - 1, indx_class, 0, :] = np.transpose(vLines[2, :])
                matrixScores[epoch - 1, indx_class, 1, :] = np.transpose(vLines[1, :])
                matrixScores[epoch - 1, indx_class, 2, :] = np.transpose(vLines[3, :])

    mean_scores = np.mean(matrixScores, axis=3)
    return mean_scores

def plot_mIoU(vmIoU, legends, classes, scores, name_prefix):
    scores = []
    for indx_score, value_score in enumerate(scores):
        accum_dataY = None
        accum_dataX = None
        print value_score  # easy, moderate, hard
        for indx_class, value_class in enumerate(classes):
            dataY = [np.asarray(mIoU)[:, indx_class, indx_score] if len(mIoU) > 0 else [] for mIoU in vmIoU]
            best_data = [max(data) if len(mIoU) > 0 else 0 for data in dataY]
            lens_data = [len(data) for data in dataY]
            dataX = [range(1, len(data) + 1) for data in dataY]
            filename = name_prefix + '_plot_' + value_class + '_' + value_score + '.png'
            title = name_prefix.split('/')[-1] + '_' + value_class + '_' + value_score
            # plot_multiple_data(dataY, dataX, filename, title, legends)
            if accum_dataY is None and accum_dataX is None:
                accum_dataY = dataY
                accum_dataX = dataX
            else:
                accum_dataY = [[el1 + el2 for (el1, el2) in zip(vX, ac_vX)] for (vX, ac_vX) in zip(dataY, accum_dataY)]
                accum_dataX = [[el1 + el2 for (el1, el2) in zip(vX, ac_vX)] for (vX, ac_vX) in zip(dataX, accum_dataX)]

        if accum_dataX is not None and accum_dataY is not None:
            filename = name_prefix + '_plot_mean_' + value_score + '.png'
            title = name_prefix.split('/')[-1] + '_mean_' + value_score
            accum_dataX = [[el / len(classes) for el in v_x] for v_x in accum_dataX]
            accum_dataY = [[el / len(classes) for el in v_x] for v_x in accum_dataY]
            # plot_multiple_data(accum_dataY, accum_dataX, filename, title, legends)

            if indx_score == 0:
                best_indx = [np.argmax(np.asarray(data)) for data in accum_dataY]
                best_epoch = [x[indx] for (x, indx) in zip(accum_dataX, best_indx)]
            best_value = [y[indx] for (y, indx) in zip(accum_dataY, best_indx)]
            # print legends  # Model name
            print best_epoch  # best epoch model
            # print best_value # best mean mAP  over each class

            for indx_class, value_class in enumerate(classes):
                dataY = [np.asarray(mIoU)[:, indx_class, indx_score] if len(mIoU) > 0 else [] for mIoU in vmIoU]
                value = [y[indx] for (y, indx) in zip(dataY, best_indx)]
                print value_class  # car, pedestrian, cyclist
                print value  # best mAP per class
                scores.append([value_class, value])
    return scores


def plot_multiple_data(dataY, dataX, filename, title, legends):
    plt.ioff()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    minY = 100
    maxY = 0
    for idx, _ in enumerate(dataY):
        x = np.asarray([data for data in dataX[idx] if dataY[idx][data - 1] != -1])
        y = np.asarray([data for data in dataY[idx] if data != -1])
        maxY = max(max(y + 0.05), maxY)
        minY = min(min(y), minY)
        lines = ax.plot(x, 100 * y)
    ax.set_title(title)

    lgd = ax.legend(legends, loc='center right', bbox_to_anchor=(2.5, 0.5))

    ax.set_xlabel('Iteration')
    ax.set_ylabel('IoU')
    ax.set_ylim([minY * 100.0, maxY * 100.0])
    # ax.set_ylim([1,100])

    plt.grid()

    # fig.draw()
    fig.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()


def Compute_plot(cf, subeval=None):
    path = os.path.join(cf.exp_folder)#'/data/104-1/Experiments/Detectron/Models'
    path_out = os.path.join(cf.exp_folder,'..','plots')#'/data/104-1/Experiments/Detectron/plots'
    # Nets
    vSourceDB = [cf.dataset]
    vNets = [cf.model_type]

    vModels = [cf.model_name]
    if subeval is not None:
        print subeval

    title_prefix = 'baselines_nets_synthia_random_generator'

    classes = cf.labels
    scores = ['moderate', 'easy', 'hard']

    nThresholds = 41

    path_out = os.path.join(path_out, title_prefix)
    # if not os.path.exists(path_out):
    #     os.makedirs(path_out)

    vmIoU = []
    legends = []
    for model in vModels:
        model_path = os.path.join(path, model + '_' + cf.dataset)
        for net in vNets:
            for sourceDB in vSourceDB:
                mean_scores = comute_mAP_splitted(path=model_path, dataset=sourceDB, net=net, classes=classes,
                                                  scores=scores, nThresholds=nThresholds, subeval=subeval)
                if mean_scores != []:
                    vmIoU.append(mean_scores)
                    legends.append(os.path.join(model, net, sourceDB))
    scores = plot_mIoU(vmIoU, legends, classes, scores, os.path.join(path_out, title_prefix))
    return scores

