import time
import torch
from options.train_options_brats import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

import os
import sys
import tensorboardX
import shutil
import numpy
import pdb
import skimage.io
import platform

if platform.system() == 'Windows':
  sys.path.append(r"E:\我的坚果云\sourcecode\python\util")
else:
  sys.path.append("/home/chenxu/我的坚果云/sourcecode/python/util")
import common_metrics
import common_brats

if __name__ == '__main__':
    opts = TrainOptions().parse()   # get training options

    if len(opts.gpu_ids) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opts.gpu_ids[0])
        device = torch.device("cuda")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        device = torch.device("cpu")

    if not os.path.exists(opts.log_dir):
        os.makedirs(opts.log_dir)

    dataset_s = common_brats.Dataset(opts.dataroot, modality="t2", n_slices=opts.input_nc, valid=True)
    dataset_t = common_brats.Dataset(opts.dataroot, modality="t1", n_slices=opts.input_nc, valid=True)
    dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=opts.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    dataloader_t = torch.utils.data.DataLoader(dataset_t, batch_size=opts.batch_size, shuffle=True, pin_memory=True, drop_last=True)

    if opts.do_validation:
        val_data_t, val_data_s = common_brats.load_test_data(opts.dataroot, "val")

    model = create_model(opts)      # create a model given opt.model and other options
    #print('The number of training images = %d' % dataset_size)

    #visualizer = Visualizer(opts)   # create a visualizer that display/save images and plots
    #opts.visualizer = visualizer

    best_psnr = 0
    optimize_time = 0.1
    for it in range(opts.max_epochs):
        for batch_id, (data_s, data_t) in enumerate(zip(dataloader_s, dataloader_t)):
            patch_s = data_s["image"].to(device)
            patch_t = data_t["image"].to(device)

            data = {
                "A": patch_s,
                "B": patch_t,
                "A_paths": "",
            }
            if it == 0 and batch_id == 0:
                model.data_dependent_initialize(data)
                model.setup(opts)  # regular setup: load and print networks; create schedulers
                model.parallelize()

            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            if len(opts.gpu_ids) > 0:
                torch.cuda.synchronize()

        if it % opts.display_freq == 0 and opts.do_validation:   # display images on visdom and save images to a HTML file
            msg = "Iter: %d" % (it + 1)

            model.netG_A.eval()
            model.netG_B.eval()

            val_st_psnr = numpy.zeros((val_data_s.shape[0], 1), numpy.float32)
            val_ts_psnr = numpy.zeros((val_data_t.shape[0], 1), numpy.float32)
            val_st_list = []
            val_ts_list = []
            with torch.no_grad():
                for i in range(val_data_s.shape[0]):
                    val_st = numpy.zeros(val_data_s.shape[1:], numpy.float32)
                    val_ts = numpy.zeros(val_data_t.shape[1:], numpy.float32)
                    used = numpy.zeros(val_data_s.shape[1:], numpy.float32)
                    for j in range(val_data_s.shape[1] - opts.input_nc + 1):
                        val_patch_s = torch.tensor(val_data_s[i:i + 1, j:j + opts.input_nc, :, :], device=device)
                        val_patch_t = torch.tensor(val_data_t[i:i + 1, j:j + opts.input_nc, :, :], device=device)

                        ret_st = model.netG_A(val_patch_s)
                        ret_ts = model.netG_B(val_patch_t)

                        val_st[j:j + opts.input_nc, :, :] += ret_st.cpu().detach().numpy()[0]
                        val_ts[j:j + opts.input_nc, :, :] += ret_ts.cpu().detach().numpy()[0]
                        used[j:j + opts.input_nc, :, :] += 1

                    assert used.min() > 0
                    val_st /= used
                    val_ts /= used

                    st_psnr = common_metrics.psnr(val_st, val_data_t[i])
                    ts_psnr = common_metrics.psnr(val_ts, val_data_s[i])

                    val_st_psnr[i] = st_psnr
                    val_ts_psnr[i] = ts_psnr
                    val_st_list.append(val_st)
                    val_ts_list.append(val_ts)

            model.netG_A.train()
            model.netG_B.train()

            msg += "  val_st_psnr:%f/%f  val_ts_psnr:%f/%f" % \
                   (val_st_psnr.mean(), val_st_psnr.std(), val_ts_psnr.mean(), val_ts_psnr.std())
            gen_images_test = numpy.concatenate([val_data_s[0], val_st_list[0], val_ts_list[0], val_data_t[0]], 2)
            gen_images_test = numpy.expand_dims(gen_images_test, 0).astype(numpy.float32)
            gen_images_test = common_brats.generate_display_image(gen_images_test, is_seg=False)

            if opts.log_dir:
                try:
                    skimage.io.imsave(os.path.join(opts.log_dir, "gen_images_test.jpg"), gen_images_test)
                except:
                    pass

            if val_ts_psnr.mean() > best_psnr:
                best_psnr = val_ts_psnr.mean()

                if best_psnr > opts.psnr_threshold:
                    model.save_networks("best")

            msg += "  best_ts_psnr:%f" % best_psnr
            print(msg)

            model.update_learning_rate()

    model.save_networks("final")
