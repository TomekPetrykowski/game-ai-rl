import matplotlib.pyplot as plt


def loading(curr, all=100, chars=50):
    done = int((curr * 50) / all)
    undone = chars - done
    print(f"\r{curr}/{all}\t{'#' * done}{'-' * undone}", end=" ")


def plot(scores, mean_scores):
    plt.figure(figsize=(10, 5))
    plt.title("Training")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.grid(True)
    plt.show()
