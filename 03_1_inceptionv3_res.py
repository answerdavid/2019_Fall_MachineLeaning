import sys
sys.path.insert(0, 'C:\ProgramData\Anaconda3\Lib\site-packages')

import matplotlib.pyplot as plt

epochs = range(1, 31)

print(epochs)

loss = [6.0747,3.7929,2.4533,1.9716,1.6841,1.6044,1.5878,1.4508,1.3871,1.4446,
1.4361,1.3176,1.2161,1.2780,1.2526,1.2863,1.3247,1.3472,1.2343,1.1870,
1.1133,1.1334,1.1576,1.1114,1.2294,1.1412,1.1865,1.1536,1.2154,1.1770]
val_loss = [8.7587,5.2891,3.8696,4.0272,3.9819,3.6830,3.9985,3.4073,3.8348,4.0175,
3.8317,4.3869,4.1662,3.9874,4.4619,3.8406,4.6093,4.5120,4.9721,4.2945,
4.0850,4.4067,4.7151,4.5442,4.0538,4.8974,4.9872,4.6926,4.9227,4.8963]
acc = [0.0886,0.2374,0.3483,0.4155,0.4892,0.5200,0.5231,0.5446,0.5829,0.5477,
0.5586,0.5655,0.6129,0.5931,0.5877,0.6004,0.5971,0.5735,0.6179,0.6250,
0.6444,0.6418,0.6363,0.6258,0.6234,0.6308,0.6311,0.6382,0.6314,0.6322]
val_acc = [0.1797,0.1836,0.2500,0.2656,0.2695,0.3281,0.3438,0.3125,0.3164,0.2773,
0.3438,0.2812,0.3281,0.3398,0.3438,0.3242,0.3320,0.3047,0.3008,0.3242,
0.3594,0.3008,0.2969,0.3242,0.3750,0.3281,0.3203,0.3242,0.3047,0.3125]


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