from multiprocessing import cpu_count

# Socket Path
bind = 'unix:~/FastAPI/gunicorn.sock'

# Worker Options
workers = cpu_count() + 1
worker_class = 'uvicorn.workers.UvicornWorker'

# Logging Options
loglevel = 'debug'
accesslog = '~/FastAPI/access_log'
errorlog = '~/FastAPI/error_log'
