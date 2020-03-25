from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:
    def __init__(self, kwargs_writer: dict):

        self.writer = SummaryWriter(**kwargs_writer)

    def add(
        self,
        fit_loss=None,
        val_loss=None,
        model_for_gradient=None,
        norm_weight=None,
        step=None,
    ):

        if fit_loss is not None:
            self.writer.add_scalar("fiting loss", fit_loss, global_step=step)

        if model_for_gradient is not None:
            for tag, param in model_for_gradient.named_parameters():
                if param.data.grad is not None:
                    self.writer.add_histogram(tag, param.data.grad.cpu().numpy(), epoch)
        if val_loss is not None:
            self.writer.add_scalar("val loss", val_loss, global_step=step)
