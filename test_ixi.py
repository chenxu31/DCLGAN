"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util
import torch

import sys
import numpy
import pdb
import skimage.io
from skimage.metrics import structural_similarity as SSIM
import platform

if platform.system() == 'Windows':
  sys.path.append(r"E:\我的坚果云\sourcecode\python\util")
else:
  sys.path.append("/home/chenxu/我的坚果云/sourcecode/python/util")
import common_metrics
import common_ixi

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1s

    if len(opt.gpu_ids) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_ids[0])
        device = torch.device("cuda")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        device = torch.device("cpu")

    if opt.results_dir and not os.path.exists(opt.results_dir):
        os.makedirs(opt.results_dir)

    test_data_s, test_data_t = common_ixi.load_test_data(opt.dataroot, "test")

    model = create_model(opt)

    test_st_psnr = numpy.zeros((test_data_s.shape[0], 1), numpy.float32)
    test_ts_psnr = numpy.zeros((test_data_t.shape[0], 1), numpy.float32)
    test_st_ssim = numpy.zeros((test_data_s.shape[0], 1), numpy.float32)
    test_ts_ssim = numpy.zeros((test_data_t.shape[0], 1), numpy.float32)
    test_st_mae = numpy.zeros((test_data_s.shape[0], 1), numpy.float32)
    test_ts_mae = numpy.zeros((test_data_t.shape[0], 1), numpy.float32)
    test_st_list = []
    test_ts_list = []
    with torch.no_grad():
        for i in range(len(test_data_s)):
            test_st = numpy.zeros(test_data_s[i].shape, numpy.float32)
            test_ts = numpy.zeros(test_data_t[i].shape, numpy.float32)
            used = numpy.zeros(test_data_s[i].shape, numpy.float32)
            for j in range(test_data_s[i].shape[0] - opt.input_nc + 1):
                test_patch_s = torch.tensor(numpy.expand_dims(test_data_s[i][j:j + opt.input_nc, :, :], 0), device=device)
                test_patch_t = torch.tensor(numpy.expand_dims(test_data_t[i][j:j + opt.input_nc, :, :], 0), device=device)

                if i == 0 and j == 0:
                    data = {
                        "A": test_patch_s,
                        "B": test_patch_t,
                        "A_paths": "",
                    }
                    model.data_dependent_initialize(data)
                    model.setup(opt)  # regular setup: load and print networks; create schedulers
                    model.parallelize()
                    if opt.eval:
                        model.eval()

                ret_st = model.netG_A(test_patch_s)
                ret_ts = model.netG_B(test_patch_t)

                test_st[j:j + opt.input_nc, :, :] += ret_st.cpu().detach().numpy()[0]
                test_ts[j:j + opt.input_nc, :, :] += ret_ts.cpu().detach().numpy()[0]
                used[j:j + opt.input_nc, :, :] += 1

            assert used.min() > 0
            test_st /= used
            test_ts /= used

            if opt.results_dir:
                common_ixi.save_nii(test_ts, os.path.join(opt.results_dir, "syn_%d.nii.gz" % i))

            st_psnr = common_metrics.psnr(test_st, test_data_t[i])
            ts_psnr = common_metrics.psnr(test_ts, test_data_s[i])
            st_ssim = SSIM(test_st, test_data_t[i], range=2.)
            ts_ssim = SSIM(test_ts, test_data_s[i], range=2.)
            st_mae = abs(test_st - test_data_t[i]).mean()
            ts_mae = abs(test_ts - test_data_s[i]).mean()

            test_st_psnr[i] = st_psnr
            test_ts_psnr[i] = ts_psnr
            test_st_ssim[i] = st_ssim
            test_ts_ssim[i] = ts_ssim
            test_st_mae[i] = st_mae
            test_ts_mae[i] = ts_mae
            test_st_list.append(test_st)
            test_ts_list.append(test_ts)

    msg = "test_st_psnr:%f/%f  test_st_ssim:%f/%f  test_st_mae:%f/%f  test_ts_psnr:%f/%f  test_ts_ssim:%f/%f  test_ts_mae:%f/%f" % \
          (test_st_psnr.mean(), test_st_psnr.std(), test_st_ssim.mean(), test_st_ssim.std(), test_st_mae.mean(), test_st_mae.std(),
           test_ts_psnr.mean(), test_ts_psnr.std(), test_ts_ssim.mean(), test_ts_ssim.std(), test_ts_mae.mean(), test_ts_mae.std())
    print(msg)

    if opt.results_dir:
        with open(os.path.join(opt.results_dir, "result.txt"), "w") as f:
            f.write(msg)

        numpy.save(os.path.join(opt.results_dir, "st_psnr.npy"), test_st_psnr)
        numpy.save(os.path.join(opt.results_dir, "ts_psnr.npy"), test_ts_psnr)
        numpy.save(os.path.join(opt.results_dir, "st_ssim.npy"), test_st_ssim)
        numpy.save(os.path.join(opt.results_dir, "ts_ssim.npy"), test_ts_ssim)
        numpy.save(os.path.join(opt.results_dir, "st_mae.npy"), test_st_mae)
        numpy.save(os.path.join(opt.results_dir, "ts_mae.npy"), test_ts_mae)
