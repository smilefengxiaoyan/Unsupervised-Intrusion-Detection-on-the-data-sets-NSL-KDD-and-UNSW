# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):

    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
import time

# 定义一个需要计算的复杂数学函数
def complex_math_calculation():
    result = 0
    for i in range(1000000):
        result += i * i
    return result

# 测试计算机算力的函数
def test_computer_speed():
    start_time = time.time()  # 记录开始时间
    result = complex_math_calculation()  # 执行复杂数学计算
    end_time = time.time()  # 记录结束时间
    calculation_time = end_time - start_time  # 计算运算时间
    print("计算完成！结果为：", result)
    print("计算耗时：", calculation_time, "秒")

# 执行测试
if __name__ == "__main__":
    kdd_features = set(nsl_kdd_train.columns)
    # 获取UNSW-NB15数据集的特征列表
    unsw_features = set(UNSW.columns)


# Press the green button in the gutter to run the script.




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
