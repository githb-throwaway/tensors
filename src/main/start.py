import tensorflow
import numpy
import pandas
import requests
import yaml
import logging
import json
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.figsize'] = (20, 6)
mpl.rcParams['axes.grid'] = False

tensorflow.random.set_seed(7)


def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])

    return numpy.array(data), numpy.array(labels)


def create_time_steps(length):
    return list(range(-length, 0))


def multi_step_plot(history, true_future, prediction, step_size):
    plt.figure(figsize=(20, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, numpy.array(history[:, 0]), label='History')
    plt.plot(numpy.arange(num_out) / step_size, numpy.array(true_future), 'bo',
             label='True Future')
    if prediction.any():
        plt.plot(numpy.arange(num_out) / step_size, numpy.array(prediction), 'ro',
                 label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()


def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()


def main():
    # Derive run configuration parameters
    run_args = {sys.argv[i].replace("--", ""): sys.argv[i + 1] for i in range(1, len(sys.argv), 2)}

    if run_args["env"] == "local":
        symbol = "AAPL"
        window = 100
    else:
        # Derive stock to analyze
        symbol = input("Stock Symbol: ")

        # Derive estimation window
        window = int(input("How many days forward to estimate: "))

    # Extract properties
    logging.info("Reading secure properties...")
    properties = open("resources/properties.yml", "r")
    properties_yml = yaml.safe_load(properties)
    properties.close()

    alphavantage_key = properties_yml.get("api")["alphavantage"]

    training_coef = float(properties_yml.get("training_coef"))
    step_size = int(properties_yml.get("step_size"))
    batch_size = int(properties_yml.get("batch_size"))
    buffer_size = int(properties_yml.get("buffer_size"))
    epochs = int(properties_yml.get("epochs"))

    # Make request to AlphaVantage
    if run_args["env"] == "local":
        json_file = open("resources/aapl.json", "r")
        time_series_daily = json.load(json_file)
        json_file.close()
    else:
        logging.info("Making request to AlphaVantage...")
        response = requests.get(
            f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&outputsize=full&symbol={symbol}&apikey={alphavantage_key}&datatype=json")
        time_series_daily = response.json()

        with open("resources/aapl.json", "w") as json_file:
            json.dump(time_series_daily, json_file)

    if not time_series_daily["Meta Data"]:
        logging.error(f"Symbol {symbol} does not exist.")
        raise ValueError()

    logging.info(f"Received {time_series_daily['Meta Data']['1. Information']}")

    # Amalgamate data into two dimensional dataframe
    series_data = time_series_daily["Time Series (Daily)"].items()

    pre_pandas = [[day[0], *[float(x) for x in day[1].values()]] for day in series_data]

    feature_columns = ["_".join(x.split()[1:]) for x in list(series_data)[0][1].keys()]

    df = pandas.DataFrame(data=pre_pandas, columns=["date", *feature_columns]).sort_values(by=["date"])

    features = df[["adjusted_close"]]
    features.index = df["date"]

    features.plot(subplots=True)

    dataset = features.values
    training_term_index = int(len(features) * training_coef)
    data_mean = dataset[:training_term_index].mean(axis=0)
    data_std = dataset[:training_term_index].std(axis=0)

    dataset = (dataset - data_mean) / data_std

    window_steps_modified = int(window * step_size)

    history_size = int(len(features) - 1.1*training_term_index - 1)

    x_train_multi, y_train_multi = multivariate_data(dataset=dataset,
                                                     target=dataset[:, 0],
                                                     start_index=0,
                                                     end_index=training_term_index,
                                                     history_size=history_size,
                                                     target_size=window_steps_modified,
                                                     step=step_size)

    x_validate_multi, y_validate_multi = multivariate_data(dataset=dataset,
                                                           target=dataset[:, 0],
                                                           start_index=training_term_index,
                                                           end_index=None,
                                                           history_size=history_size,
                                                           target_size=window_steps_modified,
                                                           step=step_size)

    train_data_multi = tensorflow.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
    train_data_multi = train_data_multi.cache().shuffle(buffer_size).batch(batch_size).repeat()

    val_data_multi = tensorflow.data.Dataset.from_tensor_slices((x_validate_multi, y_validate_multi))
    val_data_multi = val_data_multi.batch(batch_size).repeat()

    for x, y in train_data_multi.take(1):
        multi_step_plot(x[0], y[0], numpy.array([0]), step_size)

    multi_step_model = tensorflow.keras.models.Sequential()
    multi_step_model.add(tensorflow.keras.layers.LSTM(32, return_sequences=True, input_shape=x_train_multi.shape[-2:]))
    multi_step_model.add(tensorflow.keras.layers.LSTM(16, activation='relu'))
    multi_step_model.add(tensorflow.keras.layers.Dense(72))

    multi_step_model.compile(optimizer=tensorflow.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

    multi_step_history = multi_step_model.fit(train_data_multi, epochs=epochs, steps_per_epoch=200, validation_data=val_data_multi, validation_steps=50)

    multi_step_model.save(f"{symbol}_model.h5")

    plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')

    for x, y in val_data_multi.take(3):
        multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0], step_size)


main()
