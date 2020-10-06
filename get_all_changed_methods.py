from get_java_methods import ChangedMethodsFinder


def read_commit_names(path='commit_hashes.txt'):
    f = open(path, 'r')
    commits = f.readlines()
    f.close()
    commits_count = len(commits)
    path = "C:\\Users\\lissrbay\\Desktop\\bugml\\intellij-community"
    for i in range(commits_count-1):
        commit_a = commits[i].rstrip()
        commit_b = commits[i + 1].rstrip()
        cmf = ChangedMethodsFinder()
        print("Changed methods between {} and {}".format(commit_a, commit_b))
        print(cmf.find_changed_methods(path, [commit_a, commit_b]))


read_commit_names()
