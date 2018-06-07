from sklearn import preprocessing
le = preprocessing.LabelEncoder()
city = le.fit_transform(["paris", "paris", "tokyo", "amsterdam"])

print(city)


city1 = le.fit(["paris", "paris", "tokyo", "amsterdam"])
print(city1)