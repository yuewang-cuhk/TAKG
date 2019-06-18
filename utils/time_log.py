import time

def time_since(start_time):
    return time.time()-start_time


def convert_time2str(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%dh-%02dm" % (h, m)


if __name__ == '__main__':
    start_time = time.time()
    time.sleep(5)
    print(time_since(start_time))
