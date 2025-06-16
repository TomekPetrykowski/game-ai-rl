def loading(curr, all=100, chars=50):
    done = int((curr * 50) / all)
    undone = chars - done
    print(f"\r{curr}/{all}\t{'#' * done}{'-' * undone}", end=" ")
