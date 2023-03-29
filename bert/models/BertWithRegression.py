from collections import OrderedDict
import torch
import pytorch_lightning as pl
from loguru import logger
from torch import nn
from torch import optim
from torchmetrics.regression import MeanSquaredError
from transformers import T5EncoderModel


class BertWithRegression(pl.LightningModule):
    def __init__(
            self,
            vocab_size: int,
            nr_frozen_epochs: int,
            encoder_learning_rate: float,
            fine_tune_learning_rate: float,
            model_name: str = "Rostlab/prot_t5_xl_half_uniref50-enc"
    ):
        super().__init__()

        self.save_hyperparameters()

        # https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc/blob/main/config.json
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.encoder_learning_rate = encoder_learning_rate
        self.fine_tune_learning_rate = fine_tune_learning_rate
        self.nr_frozen_epochs = nr_frozen_epochs

        self.__build_model()
        self.__build_loss()

        if nr_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False

    def pool_strategy(self, features,
                      pool_cls=True, pool_max=True, pool_mean=True,
                      pool_mean_sqrt=True):
        token_embeddings = features['token_embeddings']
        cls_token = features['cls_token_embeddings']
        attention_mask = features['attention_mask']

        # Pooling strategy
        output_vectors = []
        if pool_cls:
            output_vectors.append(cls_token)
        if pool_max:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if pool_mean or pool_mean_sqrt:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if 'token_weights_sum' in features:
                sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if pool_mean:
                output_vectors.append(sum_embeddings / sum_mask)
            if pool_mean_sqrt:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

        output_vector = torch.cat(output_vectors, 1)
        return output_vector

    def __build_model(self):
        self.t5_model = T5EncoderModel.from_pretrained(self.model_name)
        self.encoder_features = 1024

        self.regression_head = nn.Sequential(
            nn.Linear(self.encoder_features * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def __build_loss(self):
        self.loss = MeanSquaredError()

    def unfreeze_encoder(self) -> None:
        """ un-freezes the encoder layer. """
        if self._frozen:
            logger.info(f"\n-- Encoder model fine-tuning")
            for param in self.t5_model.parameters():
                param.requires_grad = True
            self._frozen = False

    def freeze_encoder(self) -> None:
        """ freezes the encoder layer. """
        for param in self.t5_model.parameters():
            param.requires_grad = False
        self._frozen = True

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # 32,2477
        # input_ids = torch.tensor(input_ids)
        # input_ids = input_ids.squeeze(1)
        # 32,2477
        # attention_mask = torch.tensor(attention_mask)
        # attention_mask = attention_mask.squeeze(1)

        word_embeddings = self.t5_model(input_ids,
                                        attention_mask)[0]

        pooling = self.pool_strategy({"token_embeddings": word_embeddings,
                                      "cls_token_embeddings": word_embeddings[:, 0],
                                      "attention_mask": attention_mask,
                                      })
        # print(pooling.shape)
        out = self.regression_head(pooling)
        return out

    def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        inputs, targets = batch
        predict = self.forward(**inputs)
        loss = self.loss(predict.squeeze(-1), targets)
        self.log('train_loss', loss)
        tqdm_dict = {"train_loss": loss}
        output = OrderedDict(
            {"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
        return output

    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs):
        inputs, targets = batch
        model_out = self.forward(**inputs)

        loss_val = self.loss(model_out.squeeze(-1), targets)

        # output = OrderedDict({"val_loss": loss_val})
        self.log('val_loss', loss_val)

        # return output

    def on_validation_epoch_end(self):
        """ Pytorch lightning hook """
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
            self.unfreeze_encoder()

    def configure_optimizers(self):
        """ Sets different Learning rates for different parameter groups. """
        parameters = [
            {"params": self.regression_head.parameters()},
            {
                "params": self.t5_model.parameters(),
                "lr": self.encoder_learning_rate,
            },
        ]
        optimizer = optim.Adam(parameters, lr=self.fine_tune_learning_rate)
        return [optimizer], []


# if __name__ == '__main__':
#     model = BertWithRegression(1000, nr_frozen_epochs=5,
#                                encoder_learning_rate=5e-06,
#                                fine_tune_learning_rate=3e-05)
#     import torch
#     from torch.utils.data import Dataset, DataLoader
#     from transformers import BertTokenizer
#
#
#     class SampleDataset(Dataset):
#         def __init__(self, texts, labels, tokenizer, max_length):
#             self.texts = texts
#             self.labels = labels
#             self.tokenizer = tokenizer
#             self.max_length = max_length
#
#         def __len__(self):
#             return len(self.texts)
#
#         def __getitem__(self, idx):
#             text = self.texts[idx]
#             label = self.labels[idx]
#             encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length',
#                                       truncation=True)
#             return {'input_ids': encoding['input_ids'][0], 'attention_mask': encoding['attention_mask'][0]}, label
#
#
#     # Sample texts and labels
#     texts = [
#         "M K K Y T C T V C G Y I Y N P E D G D P D N G V N P G T D F K D I P D D W V C P L C G V G K D Q F E E V E E"]
#     labels = [1.0]
#
#     # Create the tokenizer
#     tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd")
#
#     # Create the dataset
#     dataset = SampleDataset(texts, labels, tokenizer, max_length=32)
#
#     # Create the data loader
#     data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
#
#     # Loop through the data loader
#     for batch in data_loader:
#         inputs, targets = batch
#         out = model(**inputs)
#         print(out.shape)
