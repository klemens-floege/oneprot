import os
from pytorch_lightning.callbacks import Callback
from peft import PeftModel

class PeftBestModelCheckpoint(Callback):
    def __init__(self, dirpath):
        super().__init__()
        self.dirpath = dirpath
        self.best_val_loss = float('inf')
        os.makedirs(self.dirpath, exist_ok=True)

    def on_validation_end(self, trainer, pl_module):
        current_val_loss = trainer.callback_metrics.get('val/loss')
        if current_val_loss is not None and current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self._save_checkpoint(pl_module)

    def _save_checkpoint(self, pl_module):
        # Save PEFT model
        pl_module.network["sequence"].transformer.save_pretrained(self.dirpath)
        print(f"Saved new best PEFT model checkpoint to {self.dirpath}")

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        # This method is called when loading a checkpoint
        # Implement logic here if needed to properly load your PEFT model
        pass