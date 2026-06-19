import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

data = pd.read_csv("Datasets\\normal_outliers.csv")

x = data.drop("anamoly_score", axis=1)
y = data["anamoly_score"]

x = scaler.fit_transform(x)

# Checking if data is normally distributed bcz elliptical envelope works good with normally distributed data.
# plt.scatter(x[:,0], x[:,1], c = y, cmap="viridis")
# plt.show()

trainx , testx, trainy ,testy = train_test_split(x,y, test_size=0.3, random_state=33)

envelope = EllipticEnvelope(contamination = 0.2)
envelope.fit(trainx)  # Unsupervised trainy will not be added
predy = envelope.predict(testx)

# Correct mapping: -1 → 1 (fraud), 1 → 0 (normal)

predy = np.where(predy == -1, 1, 0)

print("The Accuracy score after using Isolation forest for anamoly detection is:",accuracy_score(testy,predy))

plt.scatter(trainx[:,0], trainx[:,1], c = trainy, cmap="viridis")

xx, yy = np.meshgrid(
    np.linspace(trainx[:,0].min(), trainx[:,0].max(), 100),
    np.linspace(trainx[:,1].min(), trainx[:,1].max(), 100)
)

# Compute decision boundary
z = envelope.decision_function(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)

# Draw ellipse boundary (0 = separation line)
plt.contour(xx, yy, z, levels=[0], linewidths=2, colors="red")

plt.show()