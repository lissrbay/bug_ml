import argparse
import subprocess


def save_fix_commits(repo_path: str, data_dir: str):
    """
    Collects all fixed commits from repository and saves to data directory.
    """
    print(f"Collecting all commits")
    args = ["sh", "collect_fix_commits.sh", repo_path, data_dir]
    subprocess.Popen(args).communicate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_path", type=str)
    parser.add_argument("--data_dir", type=str)
    args = parser.parse_args()

    save_fix_commits(args.repo_path, args.data_dir)
