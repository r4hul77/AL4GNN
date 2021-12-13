import matplotlib.pyplot as plt

def plot_results(df, method, dataset, save_path=None):
    fig1 = plt.figure()
    plt.plot(df['train_size'], df['train_acc'], label='training_accuracy')
    plt.plot(df['train_size'], df['test_acc'], label='test_accuracy')
    title =  method + "_" + dataset
    plt.xlabel("Training Dataset size [N]")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Vs Training Samples " + " " + title)
    fig1.legend()
    plt.xlabel("Training dataset size(N)")
    fig1.legend()
    if save_path:
        fig1.savefig(save_path + "_" + title + "_"  + "Accuracy.JPEG")
    fig2 = plt.figure()
    plt.plot(df['train_size'], df['train_f1'], label='training_f1_score')
    plt.plot(df['train_size'], df['test_f1'], label='test_f1_score')
    plt.xlabel("Training Dataset size [N]")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Training Samples" + " " + title)
    fig2.legend()
    if save_path:
        fig1.savefig(save_path + "_" + title + "_" + "F1_Score.JPEG")

    plt.show()