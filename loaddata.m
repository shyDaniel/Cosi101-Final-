%% Load all datasets
weather_train = readtable('weather_train1.csv');
building = readtable('building_metadata.csv');
train = readtable('train.csv');
weather_test = readtable('weather_test1.csv');
test = readtable('test.csv');

%% Merge the datasets to get the train and test datasets
train = join(join(train, building), weather_train);
test = join(join(test, building), weather_test);

%% Convert timestamps to integers
train.timestamp = datenum(train.timestamp);
test.timestamp = datenum(test.timestamp);

%% Output the train and test datasets
writetable(train, 'train1.csv', 'Delimiter', ',', 'QuoteStrings', true);
writetable(test, 'test1.csv', 'Delimiter', ',', 'QuoteStrings', true);
