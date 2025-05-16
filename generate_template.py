import pandas as pd

# 创建示例数据
df = pd.DataFrame({
    'x': [0, 1, 2, 3],
    'y': [1, 2, 0, 4]
})

# 保存为 Excel 文件
df.to_excel("interpolation_template.xlsx", index=False)
print("模板文件已生成：interpolation_template.xlsx")
