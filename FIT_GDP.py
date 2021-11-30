import pandas as pd
import openpyxl as xl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from openpyxl.styles import Font
from mpl_toolkits.mplot3d import Axes3D  # 3Dplot
import matplotlib.pyplot as plt
import seaborn as sns

read_filename = 'GDP.xlsx'


df = pd.read_excel(read_filename, index_col='年度')
date = df.index
df['年度'] = df.index
df_2011 = df[(df.index < 2012)]
df_now = df[(df.index >= 2012)]
x1 = df_2011[['Yr', '年度']]
y1 = df_2011[['C']]


def predict_model(x, y):
    model_lr = LinearRegression()
    model_lr.fit(x, y)
    β = model_lr.coef_  # 係数
    α = model_lr.intercept_  # 切片
    r = model_lr.score(x, y)
    return α, β, r


C0 = predict_model(x1, y1)[0][0]
C1 = predict_model(x1, y1)[1][0][0]
C2 = predict_model(x1, y1)[1][0][1]
R = predict_model(x1, y1)[2]


def draw(x0, x1, y1):
    fig = plt.figure()
    plt.figure(figsize=(10, 5))
    ax = Axes3D(fig)
    ax.set_title('C = {0}Y+{1}Year{2}'.format(C1, C2, C0), size=20)
    ax.scatter3D(x0, x1, y1)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")

    fig.savefig('重回帰分析.jpg')


draw(list(x1['Yr']), list(x1['年度']), y1)

df_shift = df.shift(1)
df_shift = df_shift[df_shift.index >= 2012]
df_shift['FIT差額'] = df_now['FIT差額']
df_shift['α'] = C0 + df_shift['G'] + df_shift['I'] + df_shift['X-M'] + df_shift['FIT差額'] + df['年度'] * C2
p = (1 - C1)
df_shift['Yp'] = df_shift['α'] / p
df_now['Yp'] = df_shift['Yp']
df = df_now
df['予測-実際'] = df['Yp'] - df['Yr']
df = df.drop('年度', axis=1)
df.to_excel('重回帰分析.xlsx')

wb = xl.load_workbook('重回帰分析.xlsx')
sheet = wb.active
sheet['A13'].value = '重回帰分析:C ={0}y+{1}'.format(C1, C0)
sheet['A14'].value = '決定係数:{}'.format(R)
for i in ['B', 'C', 'D', 'E', 'G', 'H', 'I', 'J', 'K', 'L', 'M']:
    sheet.column_dimensions[i].width = 15
font = Font(name='メイリオ')
for row in sheet:
    for cell in row:
        sheet[cell.coordinate].font = font

img_dir = '重回帰分析.jpg'
img_to_excel = xl.drawing.image.Image(img_dir)
sheet.add_image(img_to_excel, 'K1')
wb.save('重回帰分析.xlsx')
