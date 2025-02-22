from imports import *

# Read data from file (Framingham_heart_disease_4000.csv) into dataframe
path = os.path.abspath(os.path.join(__file__, "../../../data"))
file_path = os.path.join(path, "Framingham_heart_disease_4000.csv")

df = pd.read_csv(file_path, sep=',')

pd.set_option('display.max_columns', None)

# replace missing values 
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
df['cigsPerDay'] = imp_mean.fit_transform(df[['cigsPerDay']])
df['totChol'] = imp_mean.fit_transform(df[['totChol']])
df['BMI'] = imp_mean.fit_transform(df[['BMI']])
df['glucose'] = imp_mean.fit_transform(df[['glucose']])

imp_most_freq  = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df['attended_college'] = imp_most_freq.fit_transform(df[['attended_college']]).ravel()
df['BPMeds'] = imp_most_freq.fit_transform(df[['BPMeds']]).ravel()

# Generate encodings for 'attended_college', 'gender'
# I do not believe these columns have anything related to the dependant variable 
df = pd.get_dummies(df, columns=['attended_college'], dtype=int)
df = pd.get_dummies(df, columns=['gender'], dtype=int)

#ordinal encode continued, reminder: I'll ordinal encode where 0 is good and 1 is bad 
oe = OrdinalEncoder(categories=[['no', 'yes']])
df['BPMeds'] = oe.fit_transform(df[['BPMeds']])

oe = OrdinalEncoder(categories=[['No', 'Yes']])
df['currentSmoker'] = oe.fit_transform(df[['currentSmoker']])

oe = OrdinalEncoder(categories=[['NO', 'YES']])
df['prevalentStroke'] = oe.fit_transform(df[['prevalentStroke']])

oe = OrdinalEncoder(categories=[['no', 'yes']])
df['prevalentHyp'] = oe.fit_transform(df[['prevalentHyp']])

oe = OrdinalEncoder(categories=[['N', 'Y']])
df['diabetes'] = oe.fit_transform(df[['diabetes']])

# Encode 'TenYearCHD' column, this is dependant var 
le = LabelEncoder()
df['TenYearCHD'] = le.fit_transform(df['TenYearCHD'])
print (le.classes_, le.transform(list(le.classes_)))

X = df.drop("TenYearCHD", axis=1)
y = df["TenYearCHD"]

# I want to separate binary data from continuous so that PCA is not impacted. Also scaling binary data doesn't make much sense 
binary_cols = ["currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes", "attended_college_n","attended_college_y", "gender_Female", "gender_Male" ]
binary_data = X[binary_cols]  # Extract binary variables
X = X.drop(columns=binary_cols)  # Extract all remaining continuous variables 

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

pca = decomposition.PCA(n_components=3, random_state=42)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)

X_pca = np.hstack([X_pca, binary_data.values])

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, stratify=y,
                                                    test_size=0.25, random_state=42)
