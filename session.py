from requests.adapters import HTTPAdapter, Retry
from requests import Session

# 推理最大并发数量
MAX_PARALLEL_CALL = 200
# 推理最大超时时间
GLOBAL_TIMEOUT = 50

# 重试策略
retry_strategy = Retry(
    total=5,  # 最大重试次数（包括首次请求）
    backoff_factor=1,  # 重试之间的等待时间因子
    status_forcelist=[404, 429, 500, 502, 503, 504],  # 需要重试的状态码列表
    allowed_methods=["POST"]  # 只对POST请求进行重试
)

adapter = HTTPAdapter(max_retries=retry_strategy, pool_maxsize=MAX_PARALLEL_CALL)
# 创建会话并添加重试逻辑
session = Session()
session.mount("https://", adapter)
session.mount("http://", adapter)