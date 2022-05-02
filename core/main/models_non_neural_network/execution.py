from core.main.models_non_neural_network import models_sm as sm
from core.main.models_non_neural_network import models_smj as smj
from core.main.models_non_neural_network import models_s as s
from core.main.models_non_neural_network import models_sj as sj

print("\n-------------- Scoring (S) --------------")
s.decision_tree()
print("\n")
s.logistic_regression()
print("\n")
s.support_vector_machine()
print("\n")
s.kNN(3)
print("\n")
s.kNN(7)


print("\n-------------- Scoring (SJ) --------------")
sj.decision_tree()
print("\n")
sj.logistic_regression()
print("\n")
sj.support_vector_machine()
print("\n")
sj.kNN(3)
print("\n")
sj.kNN(7)


print("\n-------------- Scoring (SM) --------------")
sm.decision_tree()
print("\n")
sm.logistic_regression()
print("\n")
sm.support_vector_machine()
print("\n")
sm.kNN(3)
print("\n")
sm.kNN(7)


print("\n-------------- Scoring (SMJ) --------------")
smj.decision_tree()
print("\n")
smj.logistic_regression()
print("\n")
smj.support_vector_machine()
print("\n")
smj.kNN(3)
print("\n")
smj.kNN(7)

