from typing import Any, List

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy

from .modeling_t5 import *
from .configuration_t5 import T5Config
import torch.nn.functional as F
import transformers

import logging

torch.autograd.set_detect_anomaly(True)


class descriptive(nn.Module):
    def __init__(self, hidden_size, max_labels):
        super(descriptive, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, max_labels)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)
        return h


class AloeModule(LightningModule):
    def __init__(
        self,
        model_args: dict,
        lr: float = 0.000005,
        weight_decay: float = 0.0,
        num_warmup_steps: int = 1000,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.model = T5EncoderModel(T5Config()) # BertModel(model_args["huggingface"])
        self.model_dc = descriptive(model_args["huggingface"]["trasnformer"]["hidden"], model_args['max_labels'])
        
        # pre-trained BERT model
        self.t5 = transformers.T5EncoderModel.from_pretrained("t5-large")
        
        ## TODO: Play with finetuning bert
        for name, param in self.t5.named_parameters():
            param.requires_grad = False 

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, text_embd: torch.Tensor, visuals_emdb: torch.Tensor):
        # print("Text embedding shape : ", text_embd.shape)
        t5_out = self.model(inputs_embeds=text_embd, attention_mask=None, visual_embd=visuals_emdb)
        # print("t5_out shape : ", t5_out.last_hidden_state.shape)
        outputs = self.model_dc(t5_out.last_hidden_state[:, 0])
        # print("outputs shape : ", outputs.shape)
        return outputs

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def step(self, batch: Any):
        visuals, question, target = batch
        # print("Visuals ...\n")
        # print(type(visuals))
        # try:
        #     # print("Visuals shape : ", visuals.shape)
        # except:
        #     pass

        # print("Question ...\n")
        # print(type(question))
        # print(question['ids'])
        # print("Question ids : ", question['ids'].shape)

        # print("Target ...\n")
        # print(type(target))
        # try:
        #     print("Target shape : ", target.shape)
        # except:
        #     pass

        with torch.no_grad():
            text_embd = self.t5(question["ids"], question["mask"]).last_hidden_state
            # print("Text embedding shape : ", text_embd.shape)
            # print("Format of text embedding : [batch_size], [bert['max_len']], [hidden_size]")
        logits = self.forward(text_embd=text_embd, visuals_emdb=visuals)
        
        loss = self.criterion(logits, target)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, target

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_acc(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_acc(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_acc(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        logging.info(f"Total estimated stepping batches are: {self.trainer.estimated_stepping_batches}")
        opt = torch.optim.RAdam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        scheduler = transformers.get_linear_schedule_with_warmup(
            opt,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        # steps_per_epoch = (len(self.train_dataloader())//self.batch_size)//self.trainer.accumulate_grad_batches
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.hparams.lr, steps_per_epoch=self.trainer.estimated_stepping_batches, epochs=self.trainer.max_epochs)
        
        return [opt], [{"scheduler": scheduler, "interval": "step"}]
