from isrt import Model
import numpy as np

Y = np.random.choice(2, 50)
X = []

for y in Y:
    X.append(np.random.random((np.random.randint(10, 100), 10)))
    X[-1][:, 0] = y

m = Model(bagging=.5)

m.fit(X, Y)

print(Y)
print((np.array(m.predict(X)) > 0.5).astype(int))

