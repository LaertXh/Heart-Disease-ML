from datetime import datetime
print(f'Run time: {datetime.now().strftime("%D %T")}')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer            
from sklearn.preprocessing import LabelEncoder       
from sklearn.preprocessing import StandardScaler     
from sklearn.preprocessing import OrdinalEncoder
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os
