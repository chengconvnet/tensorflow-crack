from matplotlib import pyplot as plt
import tensorflow as tf

def create_history(model, epochs, batch_size, train_generator, validation_generator, checkpoint_filepath='../checkpoint'):
    # checkpoint_filepath = '../checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='auto', save_freq='epoch',
        save_best_only=True)

    history = model.fit(x=train_generator,
                        batch_size=train_generator.n // batch_size,
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=validation_generator.n // batch_size,
                        callbacks=[model_checkpoint_callback])
    return history


def write_csv_result(save_dir, model_name, accs, val_accs, losses, val_losses):
    with open(f"{save_dir}/{model_name}_train_acc.csv", "w") as f:
        [f.write(str(acc)+'\n') for acc in accs]
    with open(f"{save_dir}/{model_name}_train_loss.csv", "w") as f:
        [f.write(str(loss)+'\n') for loss in losses]
    with open(f"{save_dir}/{model_name}_val_acc.csv", "w") as f:
        [f.write(str(acc)+'\n') for acc in val_accs]
    with open(f"{save_dir}/{model_name}_val_loss.csv", "w") as f:
        [f.write(str(loss)+'\n') for loss in val_losses]


def plot_history(history, save_dir, model_name):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    write_csv_result(save_dir, model_name, acc, val_acc, loss, val_loss)
    plt.plot(acc, 'b', label='Training acc')
    plt.plot(val_acc, 'r--', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(
        f'{save_dir}/{model_name}_training_validation_accuracy.png')
    plt.figure()

    plt.plot(loss, 'b', label='Training loss')
    plt.plot(val_loss, 'r--', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(
        f'{save_dir}/{model_name}_training_validation_loss.png')
    plt.show()
