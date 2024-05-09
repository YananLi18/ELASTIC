def convert_multiple_rows_to_latex():
    import sys
    rows = []  # 用于存储所有输入行
    column_extremes = []  # 存储每列的最小或最大值
    extreme_type = []  # 记录是寻找最小值还是最大值

    print("Enter multiple rows of data (Model Name followed by numbers), separated by spaces.")
    print("Input '!!' to finish.")

    while True:
        # 从命令行读取输入
        input_data = input()
        
        # 检查是否为结束标志
        if input_data == "!!":
            break
        else:
            # 将输入的每行数据转换为数组
            row_data = input_data.split()
            # 初始化或更新每列的极值
            for i, value in enumerate(row_data[1:], start=1):  # 从第一列数字开始
                try:
                    numeric_value = float(value)
                    # 根据列索引确定是寻找最大值还是最小值
                    if (i-1) % 3 == 2:  # 每第3列，寻找最大值
                        extreme_condition = max
                        extreme_initial = float('-inf')
                    else:  # 其余列，寻找最小值
                        extreme_condition = min
                        extreme_initial = float('inf')

                    if len(column_extremes) < i:
                        column_extremes.append(numeric_value)
                        extreme_type.append(extreme_condition)
                    else:
                        column_extremes[i-1] = extreme_condition(column_extremes[i-1], numeric_value)
                except ValueError:
                    print(f"Warning: Non-numeric value '{value}' encountered and ignored for extreme value comparison.", file=sys.stderr)

            # 存储原始行数据
            rows.append(row_data)

    # 输出所有转换后的LaTeX表格行
    print("\nLaTeX Table Rows:")
    for row in rows:
        formatted_row = []
        for i, item in enumerate(row):
            if i == 0:
                formatted_row.append(item)  # Model名称直接添加
            else:
                try:
                    # 检查并格式化极值
                    if float(item) == column_extremes[i-1]:
                        formatted_row.append(f"\\textbf{{{item}}}")
                    else:
                        formatted_row.append(item)
                except ValueError:
                    formatted_row.append(item)  # 非数字直接添加
        print(" & ".join(formatted_row) + " \\\ \hline")

# 调用函数
convert_multiple_rows_to_latex()
