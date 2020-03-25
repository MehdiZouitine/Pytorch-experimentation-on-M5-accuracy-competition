from torch.utils.tensorboard import SummaryWriter
from typing import NoReturn


class TensorboardLogger:
    """Short summary.

    Parameters
    ----------
    kwargs_writer : Dict[str, str]
        Description of parameter `kwargs_writer`.

    """

    def __init__(self, kwargs_writer: Dict[str, str]) -> NoReturn:
        """Short summary.

        Parameters
        ----------
        kwargs_writer : Dict[str, str]
            Description of parameter `kwargs_writer`.

        Returns
        -------
        NoReturn
            Description of returned object.

        """

        self.writer = SummaryWriter(**kwargs_writer)

    def add(
        self,
        fit_loss: float = None,
        val_loss: float = None,
        model_for_gradient: one_layer_LSTM_model.LstmModel = None,
        norm_weight: float = None,
        step: int = None,
    ) -> NoReturn:
        """Short summary.

        Parameters
        ----------
        fit_loss : float
            Description of parameter `fit_loss`.
        val_loss : float
            Description of parameter `val_loss`.
        model_for_gradient : one_layer_LSTM_model.LstmModel
            Description of parameter `model_for_gradient`.
        norm_weight : float
            Description of parameter `norm_weight`.
        step : int
            Description of parameter `step`.

        Returns
        -------
        NoReturn
            Description of returned object.

        """

        if fit_loss is not None:
            self.writer.add_scalar("fiting loss", fit_loss, global_step=step)

        if model_for_gradient is not None:
            for tag, param in model_for_gradient.named_parameters():
                if param.data.grad is not None:
                    self.writer.add_histogram(tag, param.data.grad.cpu().numpy(), epoch)
        if val_loss is not None:
            self.writer.add_scalar("val loss", val_loss, global_step=step)
