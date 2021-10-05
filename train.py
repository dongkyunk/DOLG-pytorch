from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import Trainer

from model.dolg import DolgNet
from config import Config


seed_everything(Config.seed)

model = DolgNet(
    input_dim=Config.input_dim,
    hidden_dim=Config.hidden_dim,
    output_dim=Config.output_dim,
    num_of_classes=Config.num_of_classes
)

trainer = Trainer(gpus=1, max_epochs=Config.epochs)

trainer.fit(model)
