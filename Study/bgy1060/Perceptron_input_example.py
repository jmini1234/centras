import warnings
warnings.simplefilter("ignore")
from sklearn.neural_network import MLPClassifier #sklearn 의 neural_network사용
                                                 #MLP=Multi Layer Perceptron

X = [[0., 0.], [0., 1.], [1., 0.],  [1., 1.]] #입력(List로 선언)
y = [[0,0],[1,0], [1,0],[0,1]] #출력(List로 선언)
clf = MLPClassifier(solver='lbfgs', alpha=0.01#Learning rate
                    , hidden_layer_sizes=(6,4) #가운데 hidden layer가 6개 4개짜리
                    , random_state=100, activation='logistic', max_iter=200)

clf.fit(X, y) #w를 조정해서 x가 들어오면 y값이 변경되게 해봐

print("------------------------------------------------------------")
print("Training loss = " + str(clf.loss_))
print()
print("Coefficients :")
print(clf.coefs_)
print()
print("Intercepts :")
print(clf.intercepts_)
print()
print("Predict for [[0., 0.], [0., 1.], [1., 0.],  [1., 1.]]")
print("Predicted value = "+ str(clf.predict([[0., 0.], [0., 1.], [1., 0.],  [1., 1.]])))
print("------------------------------------------------------------")