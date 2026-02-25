# Beginner's Guide: Deep Learning for Credit Default Prediction

Welcome to the non-technical breakdown of our Credit Default Prediction project! 

In this project, we built a **Virtual Brain** (an Artificial Neural Network) that learns how to predict credit risk. Just like a human learns from reading books, our virtual brain learned by reading 41,068 historical bank records. 

Here is a step-by-step explanation of everything we did and how the code works.

---

## Part 1: `data_prep.py` (Teaching the Brain How to Read)

Before our Virtual Brain can learn, we have to translate the raw bank data into a format it understands. Neural Networks *only* understand numbers (math). They do not understand words like "Graduate" or "Married", and they get very confused by missing information. The `data_prep.py` file is our translator.

### 1. Loading the Data
We start by loading the two distinct Excel files (Internal Bank data and External CIBIL data) and merging them into one giant spreadsheet based on the `PROSPECTID` (which acts like a unique Customer ID).

### 2. Handling Missing Values
In this specific dataset, if the bank didn't know a piece of information (for example, someone's income), they wrote `-99999`. 
* We told our Python script to find every `-99999` and replace it with a "blank" space (`NaN`).
* We then filled those blanks with the *average* (median) of everyone else's data, so the brain isn't left guessing or learning from bad inputs.

### 3. Encoding Words into Numbers
Our target was the `Approved_Flag`, which had 4 categories indicating risk: `P1`, `P2`, `P3`, and `P4`. 
* We translated these categories into numbers: `0, 1, 2, 3`. 
* We also took other words like "Home Owner" and "Renter" and turned them into binary columns (1 if true, 0 if false).

### 4. Train-Test Split
We chopped our 51,000 customers into two sets: 
* **Train Set (80%):** This is the "Textbook" we use to teach the brain.
* **Test Set (20%):** This is the "Final Exam" we keep firmly hidden from the brain until the very end, to prove it actually learned the concepts and didn't just memorize the textbook.

### 5. Scaling Data
Imagine comparing an Income of `₹1,500,000` to a credit score of `750`. The brain might mistakenly think the Income is millions of times more important simply because the number is vastly bigger. Scaling squashes all numbers into a tiny, uniform range (usually between -3 and +3) so the brain treats all features fairly.

---

## Part 2: `ann_model.py` (Building and Training the Brain)

This is where the Deep Learning actually happens using **PyTorch** (a popular AI library built by Meta/Facebook).

### 1. The Architecture (Building the Layers)
This is the structure of our Artificial Neural Network (ANN):
* **Input Layer (93 Neurons):** It takes our 93 customer data points (Income, age, late payments, etc.) and passes them into the network.
* **Hidden Layers (128 -> 64 -> 32 Neurons):** These are the deep layers where the brain looks for patterns.
* **Activation Function (`ReLU`):** This enables the brain to understand *non-linear* logic. For example: "If you are young AND have high income, BUT missed a payment recently = high risk."
* **Dropout (`Dropout(0.3)`):** Think of this like giving the brain a blindfold for 30% of its neurons. It forces the remaining 70% of neurons to work harder to find overall patterns, preventing the brain from merely memorizing specific customers.
* **Output Layer (4 Neurons):** It takes all the learned patterns and squashes them into 4 final neurons representing the predictions for P1, P2, P3, and P4.

### 2. The Loss Function (How the Brain Learns)
When the brain makes a guess, the `Loss Function` calculates how wrong it was. 
* If the brain sees an actual High-Risk customer but guesses they are Low-Risk, the Loss Function yells at the brain *very* loudly. We specifically added **class weights** to punish these dangerous "False Negative" mistakes severely. 
* Expanding on this, the `Optimizer` takes that yelling, goes back into the network, and slightly tweaks the mathematical weights inside the neurons so the brain guesses better next time.

### 3. The Training Loop (Epochs)
An **Epoch** is one complete read-through of the "Textbook" (the training records). 
We loop through the data 20 times (`Num_Epochs = 20`). Each time, the brain guesses, the Loss Function yells, the Optimizer tweaks the math, and the brain gets a little smarter. By Epoch 20, the brain is highly trained.

### 4. The Final Exam (Evaluation)
Finally, we lock the brain so it cannot learn anymore. We then give it the customers from the "Test Set" that it has never seen before. We compare its guesses to reality, which is how we calculated our final **92% Accuracy**.

---

## Summary
At a high level: We took messy bank records, translated them into pure math (Scaling & Encoding), built a 3-layer virtual brain (PyTorch ANN), forced it to read the records 20 times while heavily penalizing it for bad guesses, and then proved it was smart by testing it on unseen data.
