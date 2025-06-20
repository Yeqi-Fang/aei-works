import concurrent.futures
import time
import math

# 例子1: 计算平方数
def calculate_square(n):
    """计算平方数的函数"""
    time.sleep(1)  # 模拟耗时操作
    return n * n

def example1_basic():
    """基本的多进程计算例子"""
    print("=== 例子1: 基本多进程计算 ===")
    numbers = [1, 2, 3, 4, 5]
    
    # 使用ProcessPoolExecutor创建进程池
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        # 提交任务，返回future对象
        futures = [executor.submit(calculate_square, num) for num in numbers]
        
        # 获取结果
        results = []
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"计算完成，结果: {result}")
    
    print(f"所有结果: {sorted(results)}")

# 例子2: 使用map方法
def calculate_factorial(n):
    """计算阶乘"""
    return math.factorial(n)

def example2_map():
    """使用map方法的例子"""
    print("\n=== 例子2: 使用map方法 ===")
    numbers = [5, 6, 7, 8, 9]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        # 使用map方法，更简洁
        results = list(executor.map(calculate_factorial, numbers))
    
    for num, result in zip(numbers, results):
        print(f"{num}! = {result}")

# 例子3: 处理文件或数据
def process_data_chunk(data_chunk):
    """处理数据块"""
    # 模拟数据处理：计算平均值
    avg = sum(data_chunk) / len(data_chunk)
    return {
        'chunk_size': len(data_chunk),
        'average': avg,
        'sum': sum(data_chunk)
    }

def example3_data_processing():
    """数据处理例子"""
    print("\n=== 例子3: 数据处理 ===")
    # 模拟大量数据
    big_data = list(range(1, 101))  # 1到100的数字
    
    # 将数据分成4块
    chunk_size = 25
    data_chunks = [big_data[i:i+chunk_size] for i in range(0, len(big_data), chunk_size)]
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # 提交所有任务
        future_to_chunk = {
            executor.submit(process_data_chunk, chunk): i 
            for i, chunk in enumerate(data_chunks)
        }
        
        # 获取结果
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_index = future_to_chunk[future]
            try:
                result = future.result()
                print(f"数据块 {chunk_index}: {result}")
            except Exception as exc:
                print(f"数据块 {chunk_index} 产生异常: {exc}")

# 例子4: 带超时处理
def slow_task(n):
    """模拟慢任务"""
    time.sleep(n)
    return f"任务{n}完成"

def example4_timeout():
    """带超时处理的例子"""
    print("\n=== 例子4: 超时处理 ===")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        # 提交任务
        futures = [executor.submit(slow_task, i) for i in [1, 2, 3]]
        
        # 设置2秒超时
        for future in concurrent.futures.as_completed(futures, timeout=2):
            try:
                result = future.result()
                print(f"结果: {result}")
            except concurrent.futures.TimeoutError:
                print("任务超时!")
            except Exception as exc:
                print(f"任务异常: {exc}")

if __name__ == "__main__":
    # 运行所有例子
    example1_basic()
    example2_map()
    example3_data_processing()
    
    print("\n注意: 例子4会有超时，可能不会完成所有任务")
    # example4_timeout()  # 取消注释来运行超时例子