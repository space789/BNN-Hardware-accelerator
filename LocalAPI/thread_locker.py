import random
import threading
import time
import sys

# sys.path.append('/home/xilinx/jupyter_notebooks/[TOP] Flexible University Project/')
lock = threading.Lock()


# import main_code

def thread_entry(data):
    lock.acquire()
    ret = main_function(data)
    lock.release()

    return ret


def main_function(data):
    """
    >> 當有外部請求的時候，會呼叫這個程式。欲與後端整合請修改這邊。 <<
    ---
    後端與訓練程式的連結
    :return: 用JSON格式表示的Python Dictionary
    """

    print('Get Data>', data)

    # Simulate a long execution time function
    for _ in range(0, 30):
        time.sleep(0.1)
        print('Processing..', _)

    # JSON-like Dictionary Object
    return {'result': random.randint(0, 9)}
