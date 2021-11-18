import mat73
import matplotlib.pyplot as plt
import seaborn as sns


data_path = "C:/Users/katha/Downloads/SBC_01/"

global_connectivity = []
for i in range(1, 813):

    if i < 10:
        filenumber = f"00{i}"
    elif i < 100:
        filenumber = f"0{i}"

    else:
        filenumber = i

    datafile_path = data_path + f"resultsROI_Subject{filenumber}_Condition001.mat"

    data_dict = mat73.loadmat(datafile_path)

    # data_dict.keys()

    connectivity_matrix = data_dict["Z"]
    # .append zum erstellen von listen
    global_connectivity.append(connectivity_matrix)



    fig1 = plt.figure(figsize = (6.5, 4.5))  #TODO: figuresize
    sns.heatmap(connectivity_matrix)

    # plt.show()
    # Achtung: Zukünftig nur plt.savefig"test.svg" (selektieren), da das folgende alles abspeichert

    # plt.savefig(f"plots/heatmap{i}.svg")
    print("end")

for i in range(len(global_connectivity)):

    print(global_connectivity[i][1,0])
    # 1.4306549437823974
    # 0.7001152237820871 für erste und zweite iteration