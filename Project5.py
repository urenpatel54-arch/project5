import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

plt.style.use("default")

# dataset

np.random.seed(10)

age = np.random.randint(18,65,300)
bmi = np.random.uniform(14,44,300)
income = np.random.randint(20000,120000,300)

df = pd.DataFrame({
    "age":age,
    "bmi":bmi,
    "income":income
})



# Q1 Hypothesis Testing

sample_mean = np.mean(df["bmi"])

if sample_mean > 25:
    print("Q1 Result: Reject H0")

else:
    print("Q1 Result: Accept H0")



# Q2 Mean and Confidence Interval

mean_age = np.mean(df["age"])

std_age = np.std(df["age"])

n = len(df["age"])

z = 1.96

margin = z * (std_age / np.sqrt(n))

lower = mean_age - margin
upper = mean_age + margin

print("Q2 Mean Age:", round(mean_age,2))
print("Q2 Confidence Interval:", round(lower,2), "to", round(upper,2))



# Q3 Critical Value

critical = norm.ppf(0.975)

print("Q3 Critical Value:", round(critical,2))



# Q4 One Sample Test

t,p = stats.ttest_1samp(df["bmi"],25)

if p < 0.05:
    print("Q4 Result: Reject H0")

else:
    print("Q4 Result: Accept H0")



# Q5 Correlation Test

corr,pvalue = stats.pearsonr(df["age"],df["bmi"])

if pvalue < 0.05:
    print("Q5 Result: Significant relation")

else:
    print("Q5 Result: No significant relation")



# Q6 Two Sample Test

group1 = df["bmi"][:150]
group2 = df["bmi"][150:]

t2,p2 = stats.ttest_ind(group1,group2)

if p2 < 0.05:
    print("Q6 Result: Significant difference")

else:
    print("Q6 Result: No significant difference")



# Q7 Correlation and Covariance

correlation = np.corrcoef(df["age"],df["bmi"])[0][1]

covariance = np.cov(df["age"],df["bmi"])[0][1]

print("Q7 Correlation:", round(correlation,4))
print("Q7 Covariance:", round(covariance,4))



# Graph 1

plt.figure(figsize=(8,6))

plt.scatter(df["age"],df["bmi"])

plt.title("Age vs BMI")
plt.xlabel("age")
plt.ylabel("bmi")

plt.show()



# Graph 2

plt.figure(figsize=(8,6))

plt.hist(df["age"],bins=10,edgecolor="black")

plt.title("Age Distribution")
plt.xlabel("age")
plt.ylabel("count")

plt.show()

print("Done")