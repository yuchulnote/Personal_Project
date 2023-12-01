import matplotlib as plt

def graph(data_list, title, color, save_path):
    batch_num_list = [i for i in range(0, len(data_list))]
    
    plt.figure(figsize=(20, 10))
    plt.rc('font', size=25)
    plt.plot(batch_num_list, data_list, color=color, marker='o', linestyle='solid')
    plt.title(title)
    plt.xlabel('Epoch')
    
    title = plt.ylabel(title)

    plt.savefig(save_path, dpi=600)
    # plt.show()
    plt.close()