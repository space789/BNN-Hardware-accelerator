from thread_locker import *
from fastapi import FastAPI

app = FastAPI()


@app.post("/")
async def root(data: dict):
    print(data['data'])
    ret = {
        'result': '1',
        'probability_arr': [3.4994741e-06, 9.9958432e-01, 2.4019920e-08, 1.7870320e-08,
                            1.0729757e-05, 1.0367481e-06, 7.5637257e-07, 3.9836887e-04,
                            5.4714109e-09, 1.2623170e-06],
        'total_time': 0.08027005195617676,
        'data_prep_time': 0.045114755630493164,
        'conv_time': 0.021091222763061523,
        'pool_time': 0.01205897331237793,
        'fc_time': 0.0007233619689941406
    }
    return ret
