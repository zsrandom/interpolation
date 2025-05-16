import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STSong']  # 指定默认字体


class InterpolationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("插值计算工具")

        # 初始化数据存储
        self.x_data = []
        self.y_data = []
        self.interp_x = None
        self.interp_y = None

        # 创建GUI组件
        self.create_widgets()

    def create_widgets(self):
        # 数据输入区域
        input_frame = ttk.LabelFrame(self.root, text="数据输入")
        input_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

        ttk.Button(input_frame, text="手动输入数据", command=self.manual_input).grid(row=0, column=0, padx=5)
        ttk.Button(input_frame, text="从Excel导入", command=self.excel_input).grid(row=0, column=1, padx=5)

        # 插值方法选择
        method_frame = ttk.LabelFrame(self.root, text="插值方法")
        method_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        self.method_var = tk.StringVar(value="lagrange")
        ttk.Radiobutton(method_frame, text="拉格朗日插值", variable=self.method_var,
                        value="lagrange").grid(row=0, column=0)
        ttk.Radiobutton(method_frame, text="牛顿插值", variable=self.method_var,
                        value="newton").grid(row=0, column=1)

        # 插值点输入
        interp_frame = ttk.LabelFrame(self.root, text="插值计算")
        interp_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        ttk.Label(interp_frame, text="插值点x:").grid(row=0, column=0)
        self.x_entry = ttk.Entry(interp_frame)
        self.x_entry.grid(row=0, column=1, padx=5)
        ttk.Button(interp_frame, text="计算", command=self.calculate).grid(row=0, column=2)

        # 图表控制
        control_frame = ttk.LabelFrame(self.root, text="图表控制")
        control_frame.grid(row=3, column=0, padx=10, pady=5, sticky="ew")

        ttk.Label(control_frame, text="X范围:").grid(row=0, column=0)
        self.xmin_entry = ttk.Entry(control_frame, width=6)
        self.xmin_entry.grid(row=0, column=1)
        ttk.Label(control_frame, text="-").grid(row=0, column=2)
        self.xmax_entry = ttk.Entry(control_frame, width=6)
        self.xmax_entry.grid(row=0, column=3)

        ttk.Button(control_frame, text="更新图表", command=self.plot).grid(row=0, column=4, padx=5)
        ttk.Button(control_frame, text="保存图片", command=self.save_plot).grid(row=0, column=5)

        # 图表区域
        self.figure = plt.Figure(figsize=(8, 4))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().grid(row=4, column=0, padx=10, pady=10)

    def manual_input(self):
        win = tk.Toplevel()
        win.title("手动输入数据")

        tk.Label(win, text="x值（空格分隔）:").grid(row=0, column=0)
        x_entry = tk.Entry(win, width=40)
        x_entry.grid(row=0, column=1)

        tk.Label(win, text="y值（空格分隔）:").grid(row=1, column=0)
        y_entry = tk.Entry(win, width=40)
        y_entry.grid(row=1, column=1)

        def save_data():
            try:
                self.x_data = list(map(float, x_entry.get().split()))
                self.y_data = list(map(float, y_entry.get().split()))
                if len(self.x_data) != len(self.y_data):
                    raise ValueError
                win.destroy()
            except:
                messagebox.showerror("错误", "数据格式不正确")

        ttk.Button(win, text="确定", command=save_data).grid(row=2, columnspan=2)

    # 新增的excel_input方法
    def excel_input(self):
        filepath = filedialog.askopenfilename(filetypes=[("Excel文件", "*.xlsx")])
        if filepath:
            try:
                df = pd.read_excel(filepath)
                self.x_data = df.iloc[:, 0].tolist()
                self.y_data = df.iloc[:, 1].tolist()
                messagebox.showinfo("成功", f"成功导入{len(self.x_data)}个数据点")
            except Exception as e:
                messagebox.showerror("错误", f"读取文件失败: {str(e)}")

    def lagrange_interp(self, x):
        n = len(self.x_data)
        result = 0.0
        for i in range(n):
            term = self.y_data[i]
            for j in range(n):
                if i != j:
                    term *= (x - self.x_data[j]) / (self.x_data[i] - self.x_data[j])
            result += term
        return result

    def newton_interp(self, x):
        n = len(self.x_data)
        # 计算差商表
        diff_quot = [self.y_data.copy()]
        for i in range(1, n):
            diff_quot.append([])
            for j in range(n - i):
                num = diff_quot[i - 1][j + 1] - diff_quot[i - 1][j]
                den = self.x_data[j + i] - self.x_data[j]
                diff_quot[i].append(num / den)

        # 计算插值结果
        result = diff_quot[0][0]
        product = 1.0
        for i in range(1, n):
            product *= (x - self.x_data[i - 1])
            result += diff_quot[i][0] * product
        return result

    def calculate(self):
        if not self.x_data:
            messagebox.showerror("错误", "请先输入数据")
            return

        try:
            x = float(self.x_entry.get())
            if self.method_var.get() == "lagrange":
                y = self.lagrange_interp(x)
            else:
                y = self.newton_interp(x)
            self.interp_x = x
            self.interp_y = y
            messagebox.showinfo("结果", f"插值结果：f({x}) = {y:.4f}")
            self.plot()
        except ValueError:
            messagebox.showerror("错误", "无效的插值点")
        except Exception as e:
            messagebox.showerror("错误", f"计算时发生错误: {str(e)}")

    def plot(self):
        self.ax.clear()

        if not self.x_data:
            return

        # 获取绘图参数
        try:
            xmin = float(self.xmin_entry.get()) if self.xmin_entry.get() else min(self.x_data) - 1
            xmax = float(self.xmax_entry.get()) if self.xmax_entry.get() else max(self.x_data) + 1
            res = 100  # 固定分辨率
        except:
            messagebox.showerror("错误", "无效的图表参数")
            return

        # 生成插值曲线
        x_vals = np.linspace(xmin, xmax, res)
        try:
            if self.method_var.get() == "lagrange":
                y_vals = [self.lagrange_interp(x) for x in x_vals]
            else:
                y_vals = [self.newton_interp(x) for x in x_vals]
        except Exception as e:
            messagebox.showerror("绘图错误", f"生成曲线失败: {str(e)}")
            return

        # 绘制图形
        self.ax.plot(x_vals, y_vals, label='插值曲线')
        self.ax.scatter(self.x_data, self.y_data, c='red', label='原始数据点')
        if self.interp_x:
            self.ax.scatter([self.interp_x], [self.interp_y], c='green',
                            marker='x', s=100, label='插值点')
        self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw()

    def save_plot(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".png")
        if filepath:
            try:
                self.figure.savefig(filepath)
                messagebox.showinfo("成功", f"图片已保存至：{filepath}")
            except Exception as e:
                messagebox.showerror("保存失败", f"无法保存图片: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = InterpolationApp(root)
    root.mainloop()
    