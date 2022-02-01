# ML
###Machine Learning Algorithms Implementation Project.
This project present a few ML algorithms I implemented.
The motivation to this project is self learning and dipper understanding of the algorithms,
as part of professional development. Each algorithm have a unique text file that explains 

## Download the Data from GoogleDrive
Download the 'data' folder using the following link: [My_GoogleDrive](https://drive.google.com/drive/folders/1pohCzjaTY1ZTzqvKXm8KPB2pC6BEuovB).
Put the data in a folder on your computer.

## Clone the Project
Clone the project
```bash
https://github.com/ronleyzer/ML.git
```

## Installation
I used Python 3.9.1 to create this project.
Use the terminal or command line to install the requirements.txt file.

```bash
pip install -r requirements.txt
```

## Run the code
Assuming that (for example):
1. You are running anomaly_detection.py 
2. The ML folder is in C:\Users\ronro\PycharmProjects
3. The data folder is in C:\Users\ronro\Desktop

First option   - Run the code using the command line- 
Run the following command:
```python
python C:\Users\ronro\PycharmProjects\ML\algorithms\anomaly_and_outliers\anomaly_detection.py --path_in "C:\Users\ronro\Desktop\data"
```

Second option - Run the code using the configuration:
```python
# Go to "Edit Configurations" in PyCharm
# Add to the parameters the location of the data folder you just downloaded
--path_in "C:\Users\ronro\Desktop\data"
# Then press Run
```

###Path to Optional Codes in this Project
1. \ML\algorithms\anomaly_and_outliers\anomaly_detection.py
2. \ML\algorithms\anomaly_and_outliers\anomaly_detection_isolation_forest.py
3. \ML\algorithms\clustering\dbscan\dbscan.py
4. \ML\algorithms\clustering\gmm\gmm_gaussian_mixture_model.py
5. \ML\algorithms\time_series\arima\arima.py
6. \ML\algorithms\time_series\holt_winters\holt_winters.py