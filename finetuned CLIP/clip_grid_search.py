import os
import cv2
import gc
import numpy as np
import pandas as pd
import itertools
from tqdm.autonotebook import tqdm
import albumentations as A
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import timm
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer

class config:
    image_path = "<path to images folder>"
    captions_path = "<path to folder where captions.csv is stored>"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_tokenizer = "distilbert-base-uncased"

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, tokenizer, transforms):
        """
        Initializes a CLIPDataset object.

        Args:
            image_filenames (list): List of image filenames.
            captions (list): List of captions corresponding to the images.
            tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizer):
                Tokenizer object used to tokenize the captions.
            transforms (torchvision.transforms.Compose): Image transformations to be applied.

        Note:
            - `image_filenames` and `captions` must have the same length.
              If there are multiple captions for each image, the `image_filenames` list
              must have repetitive file names.

        """

        self.image_filenames = image_filenames # one description per image
        self.captions = list(captions) # one description per image
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=200
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: A dictionary containing the encoded caption, image tensor,
                    and the original caption.

        """

        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        image = cv2.imread(f"{config.image_path}/{self.image_filenames[idx]}")
        if image is None:
            print("None image encountered")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]
        return item


    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.

        """
        return len(self.captions)


class ImageEncoder(nn.Module):
    """
    Encode images to a fixed-size vector.

    Args:
        model_name (str): Name of the image encoder model. Default is 'resnet50'.
        pretrained (bool): Whether to use pretrained weights for the model. Default is True.
        trainable (bool): Whether to make the model trainable. Default is True.

    Attributes:
        model (torchvision.models.ResNet): Image encoder model.
    
    """

    def __init__(
        self, model_name='resnet50', pretrained=True, trainable=True
    ):
        """
            Initializes an ImageEncoder object.

            Args:
                model_name (str): Name of the image encoder model. Default is 'resnet50'.
                pretrained (bool): Whether to use pretrained weights for the model. Default is True.
                trainable (bool): Whether to make the model trainable. Default is True.

        """
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        """
        Forward pass of the ImageEncoder.

        Args:
            x (torch.Tensor): Input images to be encoded.

        Returns:
            torch.Tensor: Encoded image representation.

        """
        return self.model(x)


class TextEncoder(nn.Module):
    """
    Encode text input to a fixed-size vector representation.

    Args:
        model_name (str): Name of the text encoder model. Default is "distilbert-base-uncased".
        pretrained (bool): Whether to use pretrained weights for the model. Default is True.
        trainable (bool): Whether to make the model trainable. Default is True.

    Attributes:
        model (transformers.DistilBertModel): Text encoder model.
        target_token_idx (int): Index of the target token used for sentence embedding.

    """

    def __init__(self, model_name="distilbert-base-uncased", pretrained=True, trainable=True):
        """
        Initializes a TextEncoder object.

        Args:
            model_name (str): Name of the text encoder model. Default is "distilbert-base-uncased".
            pretrained (bool): Whether to use pretrained weights for the model. Default is True.
            trainable (bool): Whether to make the model trainable. Default is True.

        """
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        # We are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the TextEncoder.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask indicating which tokens to attend to.

        Returns:
            torch.Tensor: Encoded text representation.

        """
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


class ProjectionHead(nn.Module):
    """
    Projection head for projecting the input embeddings to a lower-dimensional space.

    Args:
        embedding_dim (int): Dimensionality of the input embeddings.
        projection_dim (int): Dimensionality of the projected embeddings. Default is 256.
        dropout (float): Dropout probability. Default is 0.

    Attributes:
        projection (torch.nn.Linear): Linear projection layer.
        gelu (torch.nn.GELU): GELU activation function.
        fc (torch.nn.Linear): Fully connected layer.
        dropout (torch.nn.Dropout): Dropout layer.
        layer_norm (torch.nn.LayerNorm): Layer normalization layer.

    """

    def __init__(
        self,
        embedding_dim,
        projection_dim=256,
        dropout=0
    ):
        """
        Initializes a ProjectionHead object.

        Args:
            embedding_dim (int): Dimensionality of the input embeddings.
            projection_dim (int): Dimensionality of the projected embeddings. Default is 256.
            dropout (float): Dropout probability. Default is 0.

        """
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        """
        Forward pass of the ProjectionHead.

        Args:
            x (torch.Tensor): Input embeddings to be projected.

        Returns:
            torch.Tensor: Projected embeddings.

        """
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class CLIPModel(nn.Module):
    """
    Contrastive Language-Image Pretraining (CLIP) model.

    Args:
        temperature (float): Temperature parameter for the contrastive loss. Default is 1.
        image_embedding (int): Dimensionality of the image embeddings. Default is 2048.
        text_embedding (int): Dimensionality of the text embeddings. Default is 768.
        projection_dim (int): Dimensionality of the projected embeddings. Default is 256.

    Attributes:
        image_encoder (ImageEncoder): Image encoder module.
        text_encoder (TextEncoder): Text encoder module.
        image_projection (ProjectionHead): Projection head for image embeddings.
        text_projection (ProjectionHead): Projection head for text embeddings.
        temperature (float): Temperature parameter for the contrastive loss.

    """

    def __init__(
        self,
        temperature=1,
        image_embedding=2048,
        text_embedding=768,
        projection_dim = 256
    ):
        """
        Initializes a CLIPModel object.

        Args:
            temperature (float): Temperature parameter for the contrastive loss. Default is 1.
            image_embedding (int): Dimensionality of the image embeddings. Default is 2048.
            text_embedding (int): Dimensionality of the text embeddings. Default is 768.
            projection_dim (int): Dimensionality of the projected embeddings. Default is 256.

        """
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding, projection_dim=projection_dim)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding, projection_dim=projection_dim)
        self.temperature = temperature

    def forward(self, batch):
        """
        Forward pass of the CLIPModel.

        Args:
            batch (dict): Dictionary containing input data (image, input_ids, attention_mask).

        Returns:
            torch.Tensor: Mean contrastive loss for the batch.

        """
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        batch_size = len(image_embeddings) 
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    """
    Compute the cross-entropy loss between predicted logits and target probabilities.

    Args:
        preds (torch.Tensor): Predicted logits.
        targets (torch.Tensor): Target probabilities.
        reduction (str): Specifies the reduction to apply to the loss. 
                         Options: 'none' (no reduction), 'mean' (mean of the losses).
                         Default is 'none'.

    Returns:
        torch.Tensor: Cross-entropy loss.

    """
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


def make_train_valid_dfs():
    """
    Create train and validation dataframes from a captions CSV file.

    Returns:
        pandas.DataFrame: Train dataframe containing a subset of the captions.
        pandas.DataFrame: Validation dataframe containing a subset of the captions.

    """
    dataframe = pd.read_csv(f"{config.captions_path}/captions.csv")
    max_id = dataframe["id"].max() + 1
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    return train_dataframe, valid_dataframe


def build_loaders(dataframe, tokenizer, mode):
    """
    Build data loaders for training or validation using the given dataframe and tokenizer.

    Args:
        dataframe (pandas.DataFrame): Dataframe containing image and caption data.
        tokenizer: Tokenizer object for encoding the captions.
        mode (str): Mode of the data loader. Options: 'train' or 'valid'.

    Returns:
        torch.utils.data.DataLoader: Data loader for the specified mode.

    """
    transforms = A.Compose(
        [
            A.Resize(224, 224, always_apply=True),
            A.Normalize(max_pixel_value=255.0, always_apply=True),
        ]
    )
    dataset = CLIPDataset(
        dataframe["image"].values,
        dataframe["caption"].values,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=8,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

class AvgMeter:
    """
    Computes and tracks the average value of a metric.

    Args:
        name (str): Name of the metric. Default is "Metric".

    Attributes:
        name (str): Name of the metric.
        avg (float): Average value of the metric.
        sum (float): Sum of the metric values.
        count (int): Number of metric values.

    """
    def __init__(self, name="Metric"):
        """
        Initializes an AvgMeter object.

        Args:
            name (str): Name of the metric. Default is "Metric".

        """
        self.name = name
        self.reset()

    def reset(self):
        """
        Resets the metric values to zero.

        """
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        """
        Updates the metric values with the given value.

        Args:
            val (float): Value to update the metric with.
            count (int): Number of occurrences of the value. Default is 1.

        """
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        """
        Returns a string representation of the metric.

        Returns:
            str: String representation of the metric in the format "Name: Average".

        """
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    """
    Get the learning rate of the optimizer.

    Args:
        optimizer: Optimizer object.

    Returns:
        float: Learning rate.

    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]


dataframe, _ = make_train_valid_dfs()
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
transforms = A.Compose(
    [
        A.Resize(224, 224, always_apply=True),
        A.Normalize(max_pixel_value=255.0, always_apply=True),
    ]
)
dataset = CLIPDataset(
    dataframe["image"].values,
    dataframe["caption"].values,
    tokenizer=tokenizer,
    transforms=transforms,
  )


def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    """
    Train a model for one epoch.

    Args:
        model: Model to be trained.
        train_loader: Data loader for training data.
        optimizer: Optimizer for updating model parameters.
        lr_scheduler: Learning rate scheduler.
        step (str): Step mode for the learning rate scheduler. Options: 'batch' or 'epoch'.

    Returns:
        AvgMeter: Average loss meter for the epoch.

    """
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(config.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader):
    """
    Perform validation for one epoch.

    Args:
        model: Model to be validated.
        valid_loader: Data loader for validation data.

    Returns:
        AvgMeter: Average loss meter for the validation epoch.

    """
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(config.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


train_df, valid_df = make_train_valid_dfs()
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
train_loader = build_loaders(train_df, tokenizer, mode="train")
valid_loader = build_loaders(valid_df, tokenizer, mode="valid")



def parameters(lr):
    """
    Get the parameters and their associated learning rates for optimization.

    Args:
        lr (float): Learning rate for the projection parameters.

    Returns:
        list: List of dictionaries containing the parameters and learning rates.

    """
    return [
        {"params": model.image_encoder.parameters(), "lr": 1e-4},
        {"params": model.text_encoder.parameters(), "lr": 1e-5},
        {"params": itertools.chain(
            model.image_projection.parameters(), model.text_projection.parameters()
        ), "lr": lr, "weight_decay": 1e-3}
    ]



lr = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5]
projection_dim = [128, 512]

# Start grid search
for i in lr:
    for j in projection_dim:

        model = CLIPModel(projection_dim=j).to(config.device)
        params = parameters(i)
        optimizer = torch.optim.AdamW(params, weight_decay=0.)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=.9
        )
        step = "epoch"

        directory = "<path to where you want to store your grid search results>/gs/"+str(i)+"_"+str(j)+"/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        best_loss = float('inf')
        for epoch in range(100):
            print(f"Epoch: {epoch + 1}")
            model.train()
            train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
            model.eval()
            with torch.no_grad():
                valid_loss = valid_epoch(model, valid_loader)
            
            if valid_loss.avg < best_loss:
                best_loss = valid_loss.avg
                torch.save(model.state_dict(), directory+"TrainL_"+str(train_loss)+"_ValidL_"+ \
                            str(valid_loss)+"_Epoch_"+str(epoch)+".pt")
                print("Saved Best Model!")
            
            lr_scheduler.step(valid_loss.avg)


