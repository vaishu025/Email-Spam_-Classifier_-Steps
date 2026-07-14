import joblib

# Load the trained model
model = joblib.load("spam_model.pkl")

print("===== Email Spam Classifier =====")

email = input("Enter your email message: ")

prediction = model.predict([email])

if prediction[0] == 1:
    print("\nResult: SPAM EMAIL")
else:
    print("\nResult: HAM (NOT SPAM)")