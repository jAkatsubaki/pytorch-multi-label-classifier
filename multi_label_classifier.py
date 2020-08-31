import os
import sys
import time
import copy
import random
import logging
import numpy as np
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from collections import OrderedDict, defaultdict

from data.loader import MultiLabelDataLoader
from models.model import load_model, save_model, modify_last_layer_lr
from options.options import Options
from util import util 
from util.webvisualizer import WebVisualizer


def forward_batch(model, criterion, inputs, targets, opt, phase):
    if opt.cuda:
        inputs = inputs.cuda(opt.devices[0], non_blocking=True)

    if phase in ["Train"]:
        inputs_var = Variable(inputs, requires_grad=True)
        #logging.info("Switch to Train Mode")
        model.train()
    elif phase in ["Validate", "Test"]:
        inputs_var = Variable(inputs, volatile=True)
        #logging.info("Switch to Test Mode")
        model.eval()
        
    # forward
    if opt.cuda:
        if len(opt.devices) > 1:
            output = nn.parallel.data_parallel(model, inputs_var, opt.devices)
        else:
            output = model(inputs_var)
    else:
        output = model(inputs_var)
    
    # calculate loss 
    target_vars = list()
    for index in range(len(targets)):
        if opt.cuda:
            targets[index] = targets[index].cuda(opt.devices[0], non_blocking=True)
        target_vars.append(Variable(targets[index]))
    loss_list = list()
    loss = Variable(torch.FloatTensor(1)).zero_()
    if opt.cuda:
        loss = loss.cuda(opt.devices[0])
    for index in range(len(targets)):
        sub_loss = criterion(output[index], target_vars[index])
        loss_list.append(sub_loss.data)
        loss += sub_loss
    
    return output, loss, loss_list


def forward_dataset(model, criterion, data_loader, opt, labels=None):
    sum_batch = 0 
    accuracy = list()
    avg_loss = list()
    comatrix = None
    for i, data in enumerate(data_loader):
        if opt.mode == "Train":
            if random.random() > opt.validate_ratio:
                continue
        if opt.mode == "Test":
            logging.info("test %s/%s image" %(i, len(data_loader)))
        sum_batch += 1
        inputs, targets = data
        output, loss, loss_list = forward_batch(model, criterion, inputs, targets, opt, "Validate")
        batch_accuracy, batch_predictions = calc_accuracy(output, targets, opt.score_thres, opt.top_k)

        if opt.mode == "Test":
            save_test_path = os.path.join(opt.test_dir, "plot")
            util.mkdir(save_test_path)
            if labels != None:
                targets_list = list(map(lambda x: x.numpy()[0], targets))
                targets_str = ",".join([labels[ii]["__name__"] for ii, t in enumerate(targets_list) if t == 1])
                predict_list = list(map(lambda x: x[0], batch_predictions))
                predict_str =",".join([labels[ii]["__name__"] for ii, t in enumerate(predict_list) if t == 1])

                if comatrix == None:
                    comatrix = [[0 for _ in range(len(targets))] for _ in range(len(targets))]

                for idx_t, t in enumerate(targets_list):
                    if t == 0:
                        continue
                    for idx_p, p in enumerate(predict_list):
                        if p == 1:
                            comatrix[idx_t][idx_p] += 1
                    
                plt_title = f"Correct: {targets_str}, Predict: {predict_str}"
                plt.imshow(inputs[0].permute(1,2,0))
                plt.title(plt_title)
                plt.savefig(os.path.join(save_test_path, f"test{i:09d}.png"), dpi=200, bbox_inches="tight", pad_inches=0.1)
                plt.clf()

        # accumulate accuracy
        if len(accuracy) == 0:
            accuracy = copy.deepcopy(batch_accuracy)
            for index, item in enumerate(batch_accuracy):
                for k,v in item.items():
                    accuracy[index][k]["ratio"] = v["ratio"]
        else:
            for index, item in enumerate(batch_accuracy):
                for k,v in item.items():
                    accuracy[index][k]["ratio"] += v["ratio"]
        # accumulate loss
        if len(avg_loss) == 0:
            avg_loss = copy.deepcopy(loss_list) 
        else:
            for index, loss in enumerate(loss_list):
                avg_loss[index] += loss
    # average on batches
    for index, item in enumerate(accuracy):
        for k,v in item.items():
            accuracy[index][k]["ratio"] /= float(sum_batch)
    for index in range(len(avg_loss)):
        avg_loss[index] /= float(sum_batch)
    
    plt.matshow(comatrix)
    plt.colorbar()
    plt.xlabel("prediction")
    plt.ylabel("truth")
    plt.savefig(os.path.join(opt.test_dir, "acc_matrix"))
    return accuracy, avg_loss


def calc_accuracy(outputs, targets, score_thres, top_k=(1,)):
    max_k = max(top_k)
    accuracy = []
    predictions = []
    thres_list = eval(score_thres)
    if isinstance(thres_list, float) or isinstance(thres_list, int) :
        thres_list = [eval(score_thres)]*len(targets)

    for i in range(len(targets)):
        # i-th class y_true value 
        target = targets[i]
        # i-th class y_pred value
        output = outputs[i].data
        batch_size = output.size(0)
        curr_k = min(max_k, output.size(1))
        # predict value with index 0 (false) and 1 (true)
        top_value, index = output.cpu().topk(curr_k, 1)
        index = index.t()
        top_value = top_value.t()
        correct = index.eq(target.cpu().view(1,-1).expand_as(index))
        mask = (top_value>=thres_list[i])
        correct = correct*mask
        predictions.append(index.numpy()[0].tolist())
        res = defaultdict(dict)
        for k in top_k:
            k = min(k, output.size(1))
            correct_k = correct[:k].view(-1).float().sum(0)
            res[k]["s"] = batch_size
            res[k]["r"] = correct_k
            res[k]["ratio"] = float(correct_k)/batch_size
        accuracy.append(res)
    return accuracy, predictions


def train(model, criterion, train_set, val_set, opt, labels=None):
    # define web visualizer using visdom
    webvis = WebVisualizer(opt)
    
    # modify learning rate of last layer
    finetune_params = modify_last_layer_lr(model.named_parameters(), 
                                            opt.lr, opt.lr_mult_w, opt.lr_mult_b)
    # define optimizer
    optimizer = optim.SGD(finetune_params, 
                          opt.lr, 
                          momentum=opt.momentum, 
                          weight_decay=opt.weight_decay)
    # define laerning rate scheluer
    scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                          step_size=opt.lr_decay_in_epoch,
                                          gamma=opt.gamma)
    if labels is not None:
        rid2name, id2rid = labels
    
    # record forward and backward times 
    train_batch_num = len(train_set)
    total_batch_iter = 0
    logging.info("####################Train Model###################")
    for epoch in range(opt.sum_epoch):
        epoch_start_t = time.time()
        epoch_batch_iter = 0
        logging.info('Begin of epoch %d' %(epoch))
        for i, data in enumerate(train_set):
            logging.info(f"Batch {i}")
            iter_start_t = time.time()
            # train 
            inputs, targets = data
            output, loss, loss_list = forward_batch(model, criterion, inputs, targets, opt, "Train")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
           
            webvis.reset()
            epoch_batch_iter += 1
            total_batch_iter += 1

            # display train loss and accuracy
            if total_batch_iter % opt.display_train_freq == 0:
                # accuracy
                batch_accuracy = calc_accuracy(output, targets, opt.score_thres, opt.top_k)[0]
                util.print_loss(loss_list, "Train", epoch, total_batch_iter)
                util.print_accuracy(batch_accuracy, "Train", epoch, total_batch_iter)
                if opt.display_id > 0:
                    x_axis = epoch + float(epoch_batch_iter)/train_batch_num
                    # TODO support accuracy visualization of multiple top_k
                    plot_accuracy = [batch_accuracy[i][opt.top_k[0]] for i in range(len(batch_accuracy)) ]
                    accuracy_list = [item["ratio"] for item in plot_accuracy]
                    webvis.plot_points(x_axis, loss_list, "Loss", "Train")
                    webvis.plot_points(x_axis, accuracy_list, "Accuracy", "Train")
            
            # display train data
            # TODO: OutOfIndex at target_ids = ...
            if total_batch_iter % opt.display_data_freq == 0:
                image_list = list()
                show_image_num = int(np.ceil(opt.display_image_ratio * inputs.size()[0]))
                for index in range(show_image_num): 
                    input_im = util.tensor2im(inputs[index], opt.mean, opt.std)
                    class_label = "Image_" + str(index)
                    if labels is not None:
                        target_ids = [targets[i][index] for i in range(opt.class_num)]
                        rids = [id2rid[j][k] for j,k in enumerate(target_ids)]
                        class_label += "_"
                        class_label += "#".join([rid2name[j][k] for j,k in enumerate(rids)])
                    image_list.append((class_label, input_im))
                image_dict = OrderedDict(image_list)
                save_result = total_batch_iter % opt.update_html_freq
                webvis.plot_images(image_dict, opt.display_id + 2*opt.class_num, epoch, save_result)
            
            # validate and display validate loss and accuracy
            if len(val_set) > 0  and total_batch_iter % opt.display_validate_freq == 0:
                val_accuracy, val_loss = validate(model, criterion, val_set, opt)
                x_axis = epoch + float(epoch_batch_iter)/train_batch_num
                accuracy_list = [val_accuracy[i][opt.top_k[0]]["ratio"] for i in range(len(val_accuracy))]
                util.print_loss(val_loss, "Validate", epoch, total_batch_iter)
                util.print_accuracy(val_accuracy, "Validate", epoch, total_batch_iter)
                if opt.display_id > 0:
                    webvis.plot_points(x_axis, val_loss, "Loss", "Validate")
                    webvis.plot_points(x_axis, accuracy_list, "Accuracy", "Validate")

            # save snapshot 
            if total_batch_iter % opt.save_batch_iter_freq == 0:
                logging.info("saving the latest model (epoch %d, total_batch_iter %d)" %(epoch, total_batch_iter))
                save_model(model, opt, epoch)
                # TODO snapshot loss and accuracy
            
        logging.info('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.sum_epoch, time.time() - epoch_start_t))
        
        if (epoch+1) % opt.save_epoch_freq == 0:
            logging.info('saving the model at the end of epoch %d, iters %d' %(epoch+1, total_batch_iter))
            save_model(model, opt, epoch+1) 

        # adjust learning rate 
        scheduler.step()
        lr = optimizer.param_groups[0]['lr'] 
        logging.info('learning rate = %.7f epoch = %d' %(lr,epoch)) 
    logging.info("--------Optimization Done--------")


def validate(model, criterion, val_set, opt):
    return forward_dataset(model, criterion, val_set, opt)


def test(model, criterion, test_set, opt, rid2name):
    logging.info("####################Test Model###################")
    test_accuracy, test_loss = forward_dataset(model, criterion, test_set, opt, labels=rid2name)
    logging.info("data_dir:   " + opt.data_dir + "/TestSet/")
    logging.info("score_thres:"+  str(opt.score_thres))
    for index, item in enumerate(test_accuracy):
        logging.info("Attribute %d:" %(index))
        for top_k, value in item.items():
            logging.info("----Accuracy of Top%d: %f" %(top_k, value["ratio"])) 
    logging.info("#################Finished Testing################")


def main():
    # parse options 
    op = Options()
    opt = op.parse()

    # initialize train or test working dir
    trainer_dir = "trainer_" + opt.name
    opt.model_dir = os.path.join(opt.dir, trainer_dir, "Train") 
    opt.data_dir = os.path.join(opt.dir, trainer_dir, "Data") 
    opt.test_dir = os.path.join(opt.dir, trainer_dir, "Test") 
    
    if not os.path.exists(opt.data_dir):
        os.makedirs(opt.data_dir)
    if opt.mode == "Train":
        if not os.path.exists(opt.model_dir):        
            os.makedirs(opt.model_dir)
        log_dir = opt.model_dir 
        log_path = log_dir + "/train.log"
    if opt.mode == "Test":
        if not os.path.exists(opt.test_dir):
            os.makedirs(opt.test_dir)
        log_dir = opt.test_dir
        log_path = log_dir + "/test.log"

    # save options to disk
    util.opt2file(opt, log_dir+"/opt.txt")
    
    # log setting 
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    fh = logging.FileHandler(log_path, 'a')
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logging.getLogger().addHandler(fh)
    logging.getLogger().addHandler(ch)
    log_level = logging.INFO
    logging.getLogger().setLevel(log_level)
    
    # load train or test data
    data_loader = MultiLabelDataLoader(opt)
    if opt.mode == "Train":
        train_set = data_loader.GetTrainSet()
        val_set = data_loader.GetValSet()
    elif opt.mode == "Test":
        test_set = data_loader.GetTestSet()

    num_classes = data_loader.GetNumClasses()
    rid2name = data_loader.GetRID2Name()
    id2rid = data_loader.GetID2RID()
    opt.class_num = len(num_classes)

    # load model
    model = load_model(opt, num_classes)

    # define loss function
    criterion = nn.CrossEntropyLoss(weight=opt.loss_weight) 
    
    # use cuda
    if opt.cuda:
        model = model.cuda(opt.devices[0])
        criterion = criterion.cuda(opt.devices[0])
        cudnn.benchmark = True
    
    # Train model
    if opt.mode == "Train":
        train(model, criterion, train_set, val_set, opt, (rid2name, id2rid))
    # Test model
    elif opt.mode == "Test":
        test(model, criterion, test_set, opt, rid2name)


if __name__ == "__main__":
    main()
