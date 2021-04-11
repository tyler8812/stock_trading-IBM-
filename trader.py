

if __name__ == '__main__':
    # You should not modify this part.
    import argparse

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-p", "--predict", action="store_true", help="predict electricity")  
    group.add_argument("-t", "--training", action="store_true", help="training model")
    parser.add_argument('--model_input',
                       help='input model file name')
    parser.add_argument('--train_input',
                       default='training_data.csv',
                       help='input training data file name')
    parser.add_argument('--test_input',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        help='output file name')
    args = parser.parse_args()
    
    import math
    import os
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, Dropout
    import matplotlib.pyplot as plt
    import matplotlib
    import datetime
    from keras.models import load_model

    if args.training:
        print("training model")
        n = 15
        epoch = 3
        df = pd.read_csv(args.train_input, header=None, names=["open", "high", "low", "close"])
        dt = pd.read_csv(args.test_input, header=None, names=["open", "high", "low", "close"])
        # create new dataframe with only close
        both = pd.concat([df, dt], ignore_index=True)
        data = df.filter(['close'])
        # convert to numpy
        dataset = data.values
        # get number of rows to train
        training_data_len = math.ceil(len(dataset) * 0.8)
        # scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(both)
        # create training dataset
        # create the scaled training dataset
        train_data = scaled_data[0:training_data_len, :]
        # split the data into x_train and y_train dataset
        x_train = []
        y_train = []
        for i in range(n, len(train_data)):
            x_train.append(train_data[i-n:i, 0])
            y_train.append(train_data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

        # build LSTM model
        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape= (x_train.shape[1],1)))
        model.add(Dropout(0.5))
        model.add(LSTM(100, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(100, return_sequences=False))
        model.add(Dropout(0.5))
        model.add(Dense(50))
        model.add(Dense(25))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size=16, epochs=epoch)
        if os.path.isfile(args.output): 
            os.remove(args.output)
        model.save(args.output)

    elif args.predict:
        print("predicting and get output.csv")
        n = 15
        model = load_model(args.model_input)
        df = pd.read_csv(args.train_input, header=None, names=["open", "high", "low", "close"])
        dt = pd.read_csv(args.test_input, header=None, names=["open", "high", "low", "close"])
        # create new dataframe with only close
        both = pd.concat([df, dt], ignore_index=True)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(both)

        # create new dataframe with only close
        data_train = df.filter(['close'])
        data_test = dt.filter(['close'])
        scaled_both = scaler.fit_transform(both.filter(['close']).values)
        previous_both = both.filter(['close']).values
        predict_list = []
        status = 0
        action_list = []
        for i in range(data_test.shape[0]-1):
            temp1 = pd.DataFrame()
            temp2 = pd.DataFrame()
            
            temp3 = scaled_both[-20+i-n:-20+i]
            previous = previous_both[-20+i-1]
            temp3 = np.reshape(temp3, (1, temp3.shape[0], 1))

            predictions = model.predict(temp3)
            predictions = scaler.inverse_transform(predictions)

            add = np.reshape(predictions, (1))
            
            if status == 1:
                if previous < add:
                    action_list.append(0)
                else:
                    status = 0
                    action_list.append(-1)
            elif status == 0:
                if previous < add:
                    status = 1
                    action_list.append(1)
                else:
                    status = -1
                    action_list.append(-1)
            elif status == -1:
                if previous < add:
                    status = 0
                    action_list.append(1)
                else:
                    action_list.append(0)

            predict_list.append(add)

        action_df = pd.DataFrame(action_list)
        action_df.to_csv(args.output, index=False, header=False)









