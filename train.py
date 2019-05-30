import os
import yaml  # YAML file parser
import time
import shutil  # high level operations on files or collections of files
import torch
import random  # implements pseudo-random number generators for various distributions.
import argparse  # parser for command-line options, arguments, and sub-commands
import numpy as np

from torch.utils import data  # create Dataset, Dataloader objects
from tqdm import tqdm  # progress bar for python and CLI

from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function
from ptsemseg.loader import get_loader
from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer

from tensorboardX import SummaryWriter


def train(cfg, writer, logger):
	# --------------
	# Setup seeds #
	# --------------

	# set this seed you want to have reproducible results when using random generation on CPU
	torch.manual_seed(cfg.get("seed", 1337))
	# set this seed if you want to have reproducible results when using random generation on GPU
	torch.cuda.manual_seed(cfg.get("seed", 1337))
	# if you set the np.random.seed(a_fixed_number) every time you call the numpy's other random function, the result
	# will be the same.
	np.random.seed(cfg.get("seed", 1337))
	random.seed(cfg.get("seed", 1337))

	# Setup device
	# set device to GPU (cuda) if it is available, otherwise use CPU.
	# CUDA (Compute Unified Device Architecture) is a parallel computing platform and programming model created
	# by NVIDIA and implemented by the graphics processing units (GPUs) that they produce
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Setup Augmentations
	# dict.get() returns the value for the key "augmentations" if the key is in the dict, else default.
	augmentations = cfg["training"].get("augmentations", None)
	data_aug = get_composed_augmentations(augmentations)

	# Setup Dataloader
	# create torch.nn.data.Dataset instance
	data_loader = get_loader(cfg["data"]["dataset"])
	data_path = cfg["data"]["path"]

	t_loader = data_loader(
		data_path,
		is_transform=True,
		split=cfg["data"]["train_split"],
		img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
		augmentations=data_aug,
	)

	v_loader = data_loader(
		data_path,
		is_transform=True,
		split=cfg["data"]["val_split"],
		img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
	)

	n_classes = t_loader.n_classes
	trainloader = data.DataLoader(
		t_loader,
		batch_size=cfg["training"]["batch_size"],
		num_workers=cfg["training"]["n_workers"],
		shuffle=True,
	)

	valloader = data.DataLoader(
		v_loader, batch_size=cfg["training"]["batch_size"], num_workers=cfg["training"]["n_workers"]
	)

	# Setup Metrics
	running_metrics_val = runningScore(n_classes)

	# Setup Model
	model = get_model(cfg["model"], n_classes).to(device)

	model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

	# Setup optimizer, lr_scheduler and loss function
	optimizer_cls = get_optimizer(cfg)
	optimizer_params = {k: v for k, v in cfg["training"]["optimizer"].items() if k != "name"}

	optimizer = optimizer_cls(model.parameters(), **optimizer_params)
	logger.info("Using optimizer {}".format(optimizer))

	scheduler = get_scheduler(optimizer, cfg["training"]["lr_schedule"])

	loss_fn = get_loss_function(cfg)
	logger.info("Using loss {}".format(loss_fn))

	start_iter = 0
	if cfg["training"]["resume"] is not None:
		if os.path.isfile(cfg["training"]["resume"]):
			logger.info(
				"Loading model and optimizer from checkpoint '{}'".format(cfg["training"]["resume"])
			)
			checkpoint = torch.load(cfg["training"]["resume"])
			model.load_state_dict(checkpoint["model_state"])
			optimizer.load_state_dict(checkpoint["optimizer_state"])
			scheduler.load_state_dict(checkpoint["scheduler_state"])
			start_iter = checkpoint["epoch"]
			logger.info(
				"Loaded checkpoint '{}' (iter {})".format(
					cfg["training"]["resume"], checkpoint["epoch"]
				)
			)
		else:
			logger.info("No checkpoint found at '{}'".format(cfg["training"]["resume"]))

	val_loss_meter = averageMeter()
	time_meter = averageMeter()

	best_iou = -100.0
	i = start_iter
	flag = True

	while i <= cfg["training"]["train_iters"] and flag:
		for (images, labels) in trainloader:
			i += 1
			start_ts = time.time()
			scheduler.step()
			model.train()
			images = images.to(device)
			labels = labels.to(device)

			optimizer.zero_grad()
			outputs = model(images)

			loss = loss_fn(input=outputs, target=labels)

			loss.backward()
			optimizer.step()

			time_meter.update(time.time() - start_ts)

			if (i + 1) % cfg["training"]["print_interval"] == 0:
				fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
				print_str = fmt_str.format(
					i + 1,
					cfg["training"]["train_iters"],
					loss.item(),
					time_meter.avg / cfg["training"]["batch_size"],
				)

				print(print_str)
				logger.info(print_str)
				writer.add_scalar("loss/train_loss", loss.item(), i + 1)
				time_meter.reset()

			if (i + 1) % cfg["training"]["val_interval"] == 0 or (i + 1) == cfg["training"][
				"train_iters"
			]:
				model.eval()
				with torch.no_grad():
					for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
						images_val = images_val.to(device)
						labels_val = labels_val.to(device)

						outputs = model(images_val)
						val_loss = loss_fn(input=outputs, target=labels_val)

						pred = outputs.data.max(1)[1].cpu().numpy()
						gt = labels_val.data.cpu().numpy()

						running_metrics_val.update(gt, pred)
						val_loss_meter.update(val_loss.item())

				writer.add_scalar("loss/val_loss", val_loss_meter.avg, i + 1)
				logger.info("Iter %d Loss: %.4f" % (i + 1, val_loss_meter.avg))

				score, class_iou = running_metrics_val.get_scores()
				for k, v in score.items():
					print(k, v)
					logger.info("{}: {}".format(k, v))
					writer.add_scalar("val_metrics/{}".format(k), v, i + 1)

				for k, v in class_iou.items():
					logger.info("{}: {}".format(k, v))
					writer.add_scalar("val_metrics/cls_{}".format(k), v, i + 1)

				val_loss_meter.reset()
				running_metrics_val.reset()

				if score["Mean IoU : \t"] >= best_iou:
					best_iou = score["Mean IoU : \t"]
					state = {
						"epoch": i + 1,
						"model_state": model.state_dict(),
						"optimizer_state": optimizer.state_dict(),
						"scheduler_state": scheduler.state_dict(),
						"best_iou": best_iou,
					}
					save_path = os.path.join(
						writer.file_writer.get_logdir(),
						"{}_{}_best_model.pkl".format(cfg["model"]["arch"], cfg["data"]["dataset"]),
					)
					torch.save(state, save_path)

			if (i + 1) == cfg["training"]["train_iters"]:
				flag = False
				break


if __name__ == "__main__":
	# create an argument parser object
	parser = argparse.ArgumentParser(description="config")
	# add arguments to parser object
	parser.add_argument(
		"--config",
		nargs="?",
		type=str,
		default="configs/fcn8s_pascal.yml",
		help="Configuration file to use",
	)
	# parse arguments and store the parsing results into a variable.
	args = parser.parse_args()

	# read content from a yml file and return a dict
	with open(args.config) as fp:
		cfg = yaml.load(fp)

	# example logdir: 'runs/fcn8s_pascal/80739'
	# os.path.basename returns the final component of a path
	run_id = random.randint(1, 100000)
	logdir = os.path.join("runs", os.path.basename(args.config)[:-4], str(run_id))
	writer = SummaryWriter(log_dir=logdir)

	print("RUNDIR: {}".format(logdir))
	# copy config file (yaml file) into logdir
	shutil.copy(args.config, logdir)

	logger = get_logger(logdir)
	logger.info("Let the games begin")

	train(cfg, writer, logger)
