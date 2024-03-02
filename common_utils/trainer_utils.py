from transformers.integrations import WandbCallback
from transformers.integrations.integration_utils import rewrite_logs
from transformers.trainer_callback import ProgressCallback, PrinterCallback, CallbackHandler

def update_logs_with_losses(logs, state):
    losses = getattr(state, 'losses_for_record', None)
    if losses is not None:
        logs.update(losses)
    else:
        print(f"You used a `WithLosses` Callback but no sub losses are reported!")
    return logs


class WandbCallbackWithLosses(WandbCallback):
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            logs = update_logs_with_losses(logs, state)
            logs = rewrite_logs(logs)
            self._wandb.log({**logs, "train/global_step": state.global_step})

class ProgressCallbackWithLosses(ProgressCallback):
    # hack some necessary callbacks is enough
    def on_train_begin(self, args, state, control, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        if not hasattr(state, "losses_for_record"):
            state.losses_for_record = dict()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and self.training_bar is not None:
            logs = update_logs_with_losses(logs, state)
            _ = logs.pop("total_flos", None)
            self.training_bar.write(str(logs))

class PrinterCallbackWithLosses(PrinterCallback):
    # hack some necessary callbacks is enough
    def on_train_begin(self, args, state, control, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        if not hasattr(state, "losses_for_record"):
            state.losses_for_record = dict()

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = update_logs_with_losses(logs, state)
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            print(logs)

REPLACE_MAPPING = {
    WandbCallback: WandbCallbackWithLosses,
    ProgressCallback: ProgressCallbackWithLosses,
    PrinterCallback: PrinterCallbackWithLosses,
}

def hack_callbacks_and_replace(callback_handler: CallbackHandler):
    for i, callback in enumerate(callback_handler.callbacks):
        for ins in REPLACE_MAPPING.keys():
            if isinstance(callback, ins):
                callback_handler.callbacks[i] = REPLACE_MAPPING[ins]()
                break
    return callback_handler