# ML-prediction-SberAutopodpiska
A repository of my project for the 'Introduction to Data Science' course from Skillbox.

- Data (folder)
  - contains all moduls, additional files and piplines
- Dockerfile
- main.py
- measure_response.py
- requirements.txt

---

**About Service:**
"Sber SberAutopodpiska is a long-term car rental service for individuals.

The client pays a fixed monthly fee and receives a car for a period of six months to three years.
Insurance, maintenance and some more services are also included in the payment.

---

The objective here was to train ML-model that will predict whether a user will perform one of the target actions with **ROC-AUC > 0.65** and wrap it into application like FastAPI or Flask.
Among the target actions are such as: applying for a car, starting a dialog, subscribing, requesting a callback & etc.

This was the machine learning task with a teacher – binary classification

Initially there were 2 datasets

1. with 1.8 mln records for raw data
2. with 15.7 mln records for target actions data
   
After processing and encoding I got a dataset with 1.7 mln records 
In the 'dataset' folder I have provided a sample from processed dataset with 1000 rows.
