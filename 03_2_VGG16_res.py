import sys
sys.path.insert(0, 'C:\ProgramData\Anaconda3\Lib\site-packages')

import matplotlib.pyplot as plt

epochs = range(1, 24)

print(epochs)

loss = [2.9124,2.3630,2.1558,2.0097,1.9654,1.8762,1.8477,1.7910 ,1.7153,1.7303,
1.7032,1.6914,1.6198,1.6157,1.6101,1.6050,1.5647,1.5807,1.5562,1.5193,
1.5385,1.4881,1.4910]
val_loss = [2.7105,2.4856,2.3438,2.3040,2.2191,2.1938,2.0967,2.1372,2.1931,2.1585,
2.1340,2.0549,2.0517,2.0547,2.1130,2.0781,1.9862,2.0528,2.0740,2.1029,
2.0661,2.1116,2.1070]
acc = [0.1742,0.2812,0.3630,0.3828,0.4012,0.4149,0.4345,0.4584,0.4646,0.4658,
0.4591,0.4683,0.4935,0.4991,0.4929,0.4738,0.4976,0.4968,0.4874,0.5108,
0.5102,0.5169,0.5305]
val_acc = [0.2148,0.2656,0.3438,0.3125,0.3242,0.3555,0.3633,0.3867,0.3750,0.3672,
0.3516,0.3828,0.4141,0.3906,0.3633,0.4102,0.3867,0.4180,0.3984,0.3945,
0.3984,0.3828,0.3594]


plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()