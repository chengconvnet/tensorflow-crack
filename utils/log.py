from utils.path_parse import mkdir
from utils.write_plot import create_history, plot_history


def log_model(model, model_name, train_gen, val_gen,
              epochs, batch_size,
              save_dir, model_root):
    save_dir = f'{save_dir}/{model_name}'
    mkdir(save_dir)
    checkpoint_filepath = f'{save_dir}/' + \
        'weights.{epoch:03d}-{val_loss:.7f}.h5'
    history = create_history(
        model, epochs, batch_size,
        train_gen, val_gen,
        checkpoint_filepath)
    plot_history(history, save_dir, model_name)
    model.save(f'{model_root}/{model_name}.h5')
    return history
