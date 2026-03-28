import google.generativeai as genai

genai.configure(api_key="enter_api")

model = genai.GenerativeModel("models/gemini-2.5-flash")

response = model.generate_content("Say hello")

print(response.text)