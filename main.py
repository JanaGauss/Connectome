# This is a sample Python script.
import matplotlib.pyplot as plt

datapath = "C:/Users/katha/OneDrive/Desktop/Statistik WiSe 2021/04) Innovationslabor Big Data Science/SBC_01/"
# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.



import mat73
import matplotlib.pyplot as plt
import seaborn as sns

global_connectivity = []
for i in range(1,813):

    if i < 10:
        filenumber = f"00{i}"
    elif i < 100:
        filenumber = f"0{i}"

    else:
        filenumber = i

    datafile_path = datapath + f"resultsROI_Subject{filenumber}_Condition001.mat"

    data_dict = mat73.loadmat(datafile_path)



    data_dict.keys()

    connectivity_matrix = data_dict["Z"]
    # .append zum erstellen von listen
    global_connectivity.append(connectivity_matrix)



    fig1 = plt.figure(figsize = (6.5,4.5))
    sns.heatmap(connectivity_matrix)

    # plt.show()
    # Achtung: Zukünftig nur plt.savefig"test.svg" (selektieren), da das folgende alles abspeichert

    # plt.savefig(f"plots/heatmap{i}.svg")
    print("end")

for i in range(len(global_connectivity)):

    print(global_connectivity[i][1,0])
    # 1.4306549437823974
    # 0.7001152237820871 für erste und zweite iteration