import pickle

# Example: Serialize a Python object and save it to a file
data = [["a","b"],["c","d"]]

with open('data.pkl', 'wb') as file:
    pickle.dump(data, file)

# Now, let's deserialize the object using pickle.load
with open('data.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

print(loaded_data)
