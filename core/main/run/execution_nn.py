from core.main.models_neural_network import model_sm as sm
from core.main.models_neural_network import model_smj as smj
from core.main.models_neural_network import model_s as s
from core.main.models_neural_network import model_s as sj

print("\n-------------- Scoring (S) --------------")
s.lstm(in_dim=10240)

print("\n-------------- Scoring (SJ) --------------")
sj.lstm(in_dim=10240)

print("\n-------------- Scoring (SM) --------------")
sm.lstm(in_dim=20480)

print("\n-------------- Scoring (SMJ) --------------")
smj.lstm(in_dim=21500)
