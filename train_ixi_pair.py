import time
import torch
from options.train_options_ixi import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from skimage.metrics import structural_similarity as SSIM
import random

import os
import sys
import tensorboardX
import shutil
import numpy
import pdb
import skimage.io
import platform

if platform.system() == 'Windows':
    NUM_WORKERS = 0
    sys.path.append(r"E:\我的坚果云\sourcecode\python\util")
else:
    NUM_WORKERS = 4
    sys.path.append("/home/chenxu/我的坚果云/sourcecode/python/util")
import common_metrics
import common_ixi

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

    dataset_s = common_ixi.Dataset(opts.dataroot, modality="t2", phase="val", n_slices=opts.input_nc, debug=opts.debug)
    dataset_t = common_ixi.Dataset(opts.dataroot, modality="pd", phase="val", n_slices=opts.input_nc, debug=opts.debug)

    if opts.random:
        seed = random.randint(0, 1000000)
        gen1 = torch.Generator()
        gen1.manual_seed(seed)
        seed = random.randint(0, 1000000)
        gen2 = torch.Generator()
        gen2.manual_seed(seed)
    else:
        seed = random.randint(0, 1000000)
        gen1 = torch.Generator()
        gen2 = torch.Generator()
        gen1.manual_seed(seed)
        gen2.manual_seed(seed)

    dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=opts.batch_size, shuffle=True, pin_memory=True,
                                               drop_last=True, num_workers=NUM_WORKERS, generator=gen1)
    dataloader_t = torch.utils.data.DataLoader(dataset_t, batch_size=opts.batch_size, shuffle=True, pin_memory=True,
                                               drop_last=True, num_workers=NUM_WORKERS, generator=gen2)

    val_data_s, val_data_t = common_ixi.load_test_data(opts.dataroot, "test")

    model = create_model(opts)      # create a model given opt.model and other options
    #print('The number of training images = %d' % dataset_size)

    #visualizer = Visualizer(opts)   # create a visualizer that display/save images and plots
    #opts.visualizer = visualizer

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

        if it % opts.display_freq == 0:   # display images on visdom and save images to a HTML file
            msg = "Iter: %d" % (it + 1)

            model.netG_A.eval()
            model.netG_B.eval()

            val_st_psnr = numpy.zeros((val_data_s.shape[0], ), numpy.float32)
            val_ts_psnr = numpy.zeros((val_data_t.shape[0], ), numpy.float32)
            val_st_ssim = numpy.zeros((val_data_s.shape[0], ), numpy.float32)
            val_ts_ssim = numpy.zeros((val_data_t.shape[0], ), numpy.float32)
            val_st_mae = numpy.zeros((val_data_s.shape[0], ), numpy.float32)
            val_ts_mae = numpy.zeros((val_data_t.shape[0], ), numpy.float32)
            val_st_gmsd = numpy.zeros((val_data_s.shape[0], ), numpy.float32)
            val_ts_gmsd = numpy.zeros((val_data_t.shape[0], ), numpy.float32)
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
                    st_ssim = SSIM(val_st, val_data_t[i], data_range=2.)
                    ts_ssim = SSIM(val_ts, val_data_s[i], data_range=2.)
                    st_mae = abs(val_st - val_data_t[i]).mean()
                    ts_mae = abs(val_ts - val_data_s[i]).mean()
                    st_gmsd = common_metrics.GMSD_3D(val_st, val_data_t[i])
                    ts_gmsd = common_metrics.GMSD_3D(val_ts, val_data_s[i])

                    val_st_psnr[i] = st_psnr
                    val_ts_psnr[i] = ts_psnr
                    val_st_ssim[i] = st_ssim
                    val_ts_ssim[i] = ts_ssim
                    val_st_mae[i] = st_mae
                    val_ts_mae[i] = ts_mae
                    val_st_gmsd[i] = st_gmsd
                    val_ts_gmsd[i] = ts_gmsd

            model.netG_A.train()
            model.netG_B.train()

            msg += ("  val_st_psnr:%f/%f  val_st_ssim:%f/%f  val_st_mae:%f/%f  val_st_gmsd:%f/%f"
                    "  val_ts_psnr:%f/%f  val_ts_ssim:%f/%f  val_ts_mae:%f/%f  val_ts_gmsd:%f/%f") % \
                   (val_st_psnr.mean(), val_st_psnr.std(), val_st_ssim.mean(), val_st_ssim.std(),
                    val_st_mae.mean(), val_st_mae.std(), val_st_gmsd.mean(), val_st_gmsd.std(),
                    val_ts_psnr.mean(), val_ts_psnr.std(), val_ts_ssim.mean(), val_ts_ssim.std(),
                    val_ts_mae.mean(), val_ts_mae.std(), val_ts_gmsd.mean(), val_ts_gmsd.std())
            gen_images_test = numpy.concatenate([val_data_s[0], val_st_list[0], val_ts_list[0], val_data_t[0]], 2)
            gen_images_test = numpy.expand_dims(gen_images_test, 0).astype(numpy.float32)
            gen_images_test = common_ixi.generate_display_image(gen_images_test, is_seg=False)

            if opts.log_dir:
                skimage.io.imsave(os.path.join(opts.log_dir, "gen_images_test.jpg"), gen_images_test)
                if it >= opts.max_epochs - 10:
                    numpy.save(os.path.join(opts.log_dir, "ts_psnr_%s.npy" % it), val_ts_psnr)
                    numpy.save(os.path.join(opts.log_dir, "ts_ssim_%s.npy" % it), val_ts_ssim)
                    numpy.save(os.path.join(opts.log_dir, "ts_mae_%s.npy" % it), val_ts_mae)
                    numpy.save(os.path.join(opts.log_dir, "ts_gmsd_%s.npy" % it), val_ts_gmsd)

            print(msg)

            model.update_learning_rate()

    model.save_networks("final")
