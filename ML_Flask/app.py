import flask
from flask import render_template
import pickle
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import pandas as pd

scaler_data=pd.read_csv("scaler_data.csv")
target_max_vol_data=pd.read_csv("target_max_vol_data.csv")

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return render_template('main.html' )
        
    if flask.request.method == 'POST':
        temp = 1
        with open('model_Lasso.pkl', 'rb') as fh:
            loaded_model = pickle.load(fh)
        exp_1 = float(flask.request.form['density'])
        exp_2= float(flask.request.form['elastic_modules'])
        exp_3 = float(flask.request.form['amount_of_hardener'])
        exp_4 = float(flask.request.form['epoxy_group_content'])
        exp_5 = float(flask.request.form['flash_point'])
        exp_6 = float(flask.request.form['surface_density'])
        exp_7 = float(flask.request.form['resin_consumption'])
        exp_8 = float(flask.request.form['patch_angle'])
        exp_9 = float(flask.request.form['patch_step'])
        exp_10 = float(flask.request.form['patch_density'])
        exp_11 = float(flask.request.form['matrix_filler_ratio'])
        df=pd.DataFrame([exp_1, exp_2, exp_3, exp_4, exp_5, exp_6,
                         exp_7, exp_8, exp_9, exp_10, exp_11], 
                        columns=['Плотность','Модуль упругости', 'Количество отвердителя',
                        'Содержание эпоксидных смол', 'Температура вспышки', 'Поверхностная плотность',
                        'Потребление смолы','Угол нашивки', "Шаг нашивки", 'Плотность нашивки', 'Соотношение матрица-напонитель'])
        def normalize(df, scaler_data = scaler_data, target_max_vol_data = target_max_vol_data):
            norm_feat = ['Плотность, кг/м3', 'модуль упругости, ГПа',
                'Количество отвердителя, м.%', 'Содержание эпоксидных групп,%_2',
                'Температура вспышки, С_2', 'Поверхностная плотность, г/м2',
                'Потребление смолы, г/м2', 'Угол нашивки, град', 'Шаг нашивки',
                'Плотность нашивки']
            df[norm_feat] = (df[norm_feat]-scaler_data['min'].values)/(scaler_data['max'].values-scaler_data['min'].values)
            df['Соотношение матрица-наполнитель'] = df['Соотношение матрица-наполнитель'] / target_max_vol_data['Соотношение матрица-наполнитель']
            
            return df
        temp = loaded_model.predict([[df]])
        result=temp[:,0]*target_max_vol_data['Модуль упругости при растяжении, ГПа']
        return render_template('main.html', result = result)

if __name__ == '__main__':
    app.run()
