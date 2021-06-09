from torch.utils.data import DataLoader
from trainer import l2face_trainer
from data import ann_dataset


class TrainerV2():
    TRAIN_MODE = "train"
    EVAL_MODE = "test"
    MODES = [TRAIN_MODE, EVAL_MODE]
    def __init__(self, opt, logger):
        # Set up
        logger.info('==> Loading data ...')
        self.dataloaders = {
            self.TRAIN_MODE: DataLoader(ann_dataset.Dataset_(opt), batch_size=opt.batch_size, num_workers=opt.worker_number),
            self.EVAL_MODE: DataLoader(ann_dataset.Dataset_(opt), batch_size=1, num_workers=1)
        }
        logger.info('==> Building model ...')
        self.trainer = l2face_trainer.Trainer_(opt, logger)

        # Load options
        self.mode = opt.mode
        self.niter = opt.niter
        self.niter_decay = opt.niter_decay

    def train_epoch(self):
        self.trainer.run(
            dataloader=self.dataloaders[self.mode],
            mode=self.mode
        )

    def eval_epoch(self):
        self.trainer.run(
            dataloader=self.dataloaders[self.mode],
            mode=self.mode
        )

    def train(self):
        for epoch in range(1, 1 + self.niter + self.niter_decay):
            self.train_epoch()
        self.trainer.writer.close()
