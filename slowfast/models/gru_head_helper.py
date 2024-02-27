from typing import List, Optional
from loguru import logger

import torch
import torch.nn as nn


class GRUResNetBasicHead(nn.Module):
    """
    ResNe(X)t 2D head with a GRU before.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
        self,
        cfg,
        dim_in,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        act_func="softmax",
        gru_hidden_size=512,
        gru_num_layers=2,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetBasicHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p frequency temporal
                poolings, temporal pool kernel size, frequency pool kernel size in order.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            gru_hidden_size (int): the hidden size of the GRU. It has to match the size of the noun embeddings, which are CLIP embeddings.
            gru_num_layers (int): the number of layers of the GRU
        """
        super(GRUResNetBasicHead, self).__init__()
        assert len({len(pool_size), len(dim_in)}) == 1, "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)

        self.cfg = cfg

        assert (
            len(num_classes) == 2
        ), f"num_classes must be a list of length 2 but was {len(num_classes)}: {num_classes}"
        self.num_classes = num_classes

        # GRU specific parameters
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers
        self.dim_in = dim_in

        F = sum(self.dim_in)
        V, N = self.num_classes

        for pathway in range(self.num_pathways):
            avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        # GRU Module
        self.gru = nn.GRU(
            input_size=F,  # Assuming the input size is the sum of the dimensions of the pathways
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,  # Assuming that the first dimension of the input is the batch
            bidirectional=True,  # To prevent labelling of empty frames
        )
        D = 2 if self.gru.bidirectional else 1

        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.

        self.projection_verb = nn.Linear(F, V, bias=True)
        self.projection_noun = nn.Linear(F, N, bias=True)
        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=-1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()

        else:
            raise NotImplementedError("{} is not supported as an activation function.".format(act_func))

    def forward(
        self,
        inputs: torch.Tensor,
        lengths: List[int],
        initial_batch_shape: torch.Size,
        noun_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward function for the GRU ResNet head. It first passes the spectrograms embeddings through the GRU to output
        a temporal sequence. The GRU is initialized with the nouns CLIP embeddings to focus attention to specific objects.
        Then the temporal sequence is passed through 3 fully connected layers, for the verb, noun and state vector respectively.

        "Projecting" here means that we are reducing the dimensionality of the input tensor to the number of classes.
        """
        logger.debug(f"Input: {inputs.shape if isinstance(inputs, torch.Tensor) else [i.shape for i in inputs]}")

        assert len(inputs) == self.num_pathways, "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))

        logger.debug(f"Cat: {pool_out.shape if isinstance(pool_out, torch.Tensor) else [i.shape for i in pool_out]}")
        x = torch.cat(pool_out, 1)

        # (B*L, C, T, H, W) -> (B*L, T, H, W, C).
        logger.debug(f"Permute: {x.shape if isinstance(x, torch.Tensor) else [i.shape for i in x]}")
        x = x.permute((0, 2, 3, 4, 1))

        logger.debug(f"After permute: {x.shape if isinstance(x, torch.Tensor) else [i.shape for i in x]}")

        """
        Starting from here, x has shape (B * N_s, 1, 1, n_features_asf)
        """

        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        logger.debug(f"Before GRU: {x.shape if isinstance(x, torch.Tensor) else [i.shape for i in x]}")
        x = self._gru(
            x=x,
            noun_embeddings=noun_embeddings,
            initial_batch_shape=initial_batch_shape,
            lengths=lengths,
        )
        logger.debug(f"After GRU: {x.shape if isinstance(x, torch.Tensor) else [i.shape for i in x]}")

        # Recover B, N
        B, N = initial_batch_shape
        N_v, N_n = self.num_classes

        x_v = self.projection_verb(x)  # (B*N, 1, 1, F) -> (B*N, 1, 1, N_v)
        x_v = self.fc_inference(x_v, self.act)
        x_v = x_v.view(B, N, N_v)  # (B*N, 1, 1, N_v) -> (B, N, N_v)

        x_n = self.projection_noun(x)  # (B*N, 1, 1, F) -> (B*N, 1, 1, N_n)
        x_n = self.fc_inference(x_n, self.act)
        x_n = x_n.view(B, N, N_n)  # (B*N, 1, 1, N_n) -> (B, N, N_n)

        x_n_mean = torch.zeros(B, N_n).to(x_n.device)
        x_v_mean = torch.zeros(B, N_v).to(x_v.device)
        for i, length in enumerate(lengths):
            # First index is slected, hence dim=0 for 1st dimension
            x_n_mean[i] = x_n[i, :length, :].mean(dim=0)
            x_v_mean[i] = x_v[i, :length, :].mean(dim=0)

        x_n = x_n_mean
        x_v = x_v_mean

        assert x_n.shape == (B, N_n), f"x_n.shape must be {(B, N_n)} but was {x_n.shape}"
        assert x_v.shape == (B, N_v), f"x_v.shape must be {(B, N_v)} but was {x_v.shape}"

        logger.debug(f"Output verb: {x_v.shape if isinstance(x_v, torch.Tensor) else [i.shape for i in x_v]}")

        logger.debug(f"Output noun: {x_n.shape if isinstance(x_n, torch.Tensor) else [i.shape for i in x_n]}")

        return (x_v, x_n)

    def fc_inference(self, x: torch.Tensor, act: nn.Module) -> torch.Tensor:
        """
        Perform fully convolutional inference.

        Args:
            x (tensor): input tensor.
            act (nn.Module): activation function.

        Returns:
            tensor: output tensor.
        """
        if not self.training:
            x = act(x)
            x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)

        return x

    def _gru(
        self,
        x: torch.Tensor,
        noun_embeddings: torch.Tensor,
        initial_batch_shape: torch.Size,
        lengths: List[int],
    ) -> torch.Tensor:
        """
        From Table 1 of the paper (https://arxiv.org/pdf/2103.03516.pdf), n_features_asf = 2304
        (2048 (Slow) + 256 (Fast)) just before pooling, concatenation and FC layers for classification.

        - The GRU expects a tensor of the shape $(B, N, n_features_asf)$, but in the
        forward function of the model, we reshape the input tensor of shape (B, N, C=1, T, F)
        to a tensor of shape (B*N, C=1, T, F). We then need to reshape it back to (B, N, n_features_asf)
        before passing it to the GRU.
        - n_features_asf is the number of features at the output of the ResNet module, which is 2304.

        The input vector of the GRUResNetBasicHead is of shape (B*N, 1, 1, n_features_asf). We perform
        the following operations:

        1. Squeeze it: (B*N, 1, 1, n_features_asf) -> (B*N, n_features_asf)
        2. View it: (B*N, n_features_asf) -> (B, N, n_features_asf)
        3. Pass it through the GRU. Output of the GRU is (B, N, D * gru_hidden_size) = (B, N, 2 * 512)
        4. Reshape it back: (B, N, 1024) -> (B*N, 1024)
        5. Unsqueeze it to add the channel dimension: (B*N, 1024) -> (B*N, 1, 1, 1024)
        6. Project it back to CLIP embedding space: (B*N, 1, 1, 1024) -> (B*N, 1, 1, 512)
        7. Project it back to the number of features of the input:
           (B*N, 1, 1, 512) -> (B*N, 1, 1, n_features_asf)
        """
        # Reshape noun_embeddings to be (2 * num_gpu_layers, batch_size, embedding_size)
        B, L = initial_batch_shape
        F = x.shape[-1]
        D = 2 if self.gru.bidirectional else 1

        logger.warning(f"{B=}, {L=}, {F=}, {D=}")

        # (B*N, 1, 2, 2, n_features_sf) -> (B*N, 2, 2, n_features_sf)
        logger.debug(f"Squeeze: {x.shape if isinstance(x, torch.Tensor) else [i.shape for i in x]}")
        x = x.squeeze(1)

        # (B*N, 2, 2, n_features_sf) -> (B*N, 2*2*n_features_sf)
        logger.debug(f"View 1: {x.shape if isinstance(x, torch.Tensor) else [i.shape for i in x]}")
        x = x.reshape(B * L, F)

        # (B*N, 2*2n_features_sf) -> (B, N, 2*2*n_features_sf)
        # (B*N, n_features_sf) -> (B, N, n_features_sf)
        logger.debug(f"View 2: {x.shape if isinstance(x, torch.Tensor) else [i.shape for i in x]}")
        x = x.view(B, L, F)

        logger.debug(f"Before pack padded sequence: {x.shape if isinstance(x, torch.Tensor) else [i.shape for i in x]}")

        # Pass the transformed batch through the GRU
        # (B, L, 4*F) = (B, L, 2 * gru_hidden_size)
        x = torch.nn.utils.rnn.pack_padded_sequence(
            input=x,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        x, _ = self.gru(x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        logger.debug(f"After GRU: {x.shape if isinstance(x, torch.Tensor) else [i.shape for i in x]}")
        # (B, N, 1024) -> (B*N, 1024)
        x = x.reshape(B * L, D * self.gru.hidden_size)
        logger.debug(f"Reshape: {x.shape if isinstance(x, torch.Tensor) else [i.shape for i in x]}")
        x = x.unsqueeze(1)
        logger.debug(f"After unsqueeze: {x.shape if isinstance(x, torch.Tensor) else [i.shape for i in x]}")

        x = x.unsqueeze(1).unsqueeze(1)

        return x
