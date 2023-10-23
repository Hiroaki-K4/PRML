import numpy as np
from scipy.stats import norm # 1次元ガウス分布
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def main():
    # 平均を指定
    mu = 1.0

    # 標準偏差を指定
    sigma = 2.5

    # データ数を指定
    N = 1000

    # ガウス分布に従う乱数を生成
    x_n = np.random.normal(loc=mu, scale=sigma, size=N)
    print(x_n[:5])
    
    # 作図用のxの点を作成
    x_vals = np.linspace(mu - sigma*4.0, mu + sigma*4.0, num=250)

    # ガウス分布を計算
    density = norm.pdf(x=x_vals, loc=mu, scale=sigma)

    # サンプルのヒストグラムを作成
    plt.figure(figsize=(12, 9)) # 図の設定
    plt.hist(x=x_n, bins=50, range=(x_vals.min(), x_vals.max()), color='#00A968') # ヒストグラム
    plt.xlabel('x') # x軸ラベル
    plt.ylabel('frequency') # y軸ラベル
    plt.suptitle('Gaussian Distribution', fontsize=20) # 全体のタイトル
    plt.title('$\mu=' + str(mu) + ', \sigma=' + str(sigma) + ', N=' + str(N) + '$', loc='left') # タイトル
    plt.grid() # グリッド線
    plt.show() # 描画


if __name__ == "__main__":
    main()
