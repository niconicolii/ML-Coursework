from sklearn.datasets import load_iris

def test_code():
    fp = open("clean_fake.txt", "r")
    read_fp = fp.read().split('\n')
    print(read_fp)
    fp.close()
    return None


if __name__ == "__main__":
    test_code()