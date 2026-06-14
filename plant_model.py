import pickle

with open('model.pkl', 'rb') as file:
    checkpoint = pickle.load(file)

model = checkpoint['model']
label_encoders = checkpoint['label_encoders']

plant_names = [
    'Calathea','Snake Plant','Jade Plant','Spider Plant',
    'Rubber Plant','ZZ Plant','Aloe Vera','Boston Fern',
    'Areca Palm','Peace Lily'
]

def predict_plant(user_input):

    input_data = [
        label_encoders['Light Level'].transform([user_input["lightLevel"]])[0],
        float(user_input["humidity"]),
        float(user_input["temperature"]),
        label_encoders['Space'].transform([user_input["space"]])[0],
        float(user_input["wateringFrequency"]),
        label_encoders['Care Level'].transform([user_input["careLevel"]])[0],
        label_encoders['Pet Safe'].transform([user_input["petSafe"]])[0],
        label_encoders['Soil Type'].transform([user_input["soilType"]])[0],
        label_encoders['Fertilizer Need'].transform([user_input["fertilizerNeed"]])[0]
    ]

    prediction = model.predict([input_data])
    return plant_names[int(prediction[0])]