import pandas as pd


'''upload the data'''
# save filepath to variable for easier access
melbourne_file_path = r'C:\Users\ron.l\Desktop\ML\Kaggel\house price data/melb_data.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path)
# print a summary of the data in Melbourne data
print(melbourne_data.describe())

'''clean'''
melbourne_data = melbourne_data.dropna(axis=0)

'''Selecting The Prediction Target'''
y = melbourne_data.Price

'''Choosing Features'''
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
print(X.describe())

'''Building A Model'''
from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))

# Fit model
melbourne_model.fit(X, y)

print("DAMN!")



# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
