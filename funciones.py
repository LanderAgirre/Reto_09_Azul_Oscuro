import numpy as np
import torch
from torch import nn
from torchinfo import summary
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import ast
import itertools
from statsmodels.tsa.stattools import kpss, adfuller
from statsmodels.stats.diagnostic import het_arch
def train_model(model, 
                train_dataloader=None,
                val_dataloader=None,
                test_dataloader=None,
                criterion=None, 
                optimizer=None,        
                gamma=None,       
                epochs=None,
                lr=None,
                early_stopper=None,
                clip_value=None,
                model_empty = None,
                path = None):
    
    """
    Entrena un modelo LSTM utilizando conjuntos de datos de entrenamiento, validación y prueba.

    Args:
        * model (torch.nn.Module): modelo LSTM o red neuronal a entrenar.
        * train_dataloader (DataLoader): cargador de datos de entrenamiento.
        * val_dataloader (DataLoader): cargador de datos de validación.
        * test_dataloader (DataLoader): cargador de datos de prueba.
        * criterion (nn.Module): función de pérdida utilizada (por ejemplo, nn.MSELoss()).
        * optimizer (torch.optim): optimizador utilizado para el entrenamiento (por ejemplo, Adam).
        * gamma (float): factor de decaimiento del learning rate en el scheduler exponencial.
        * epochs (int): número máximo de épocas de entrenamiento.
        * lr (float): tasa de aprendizaje inicial.
        * early_stopper (EarlyStopper): mecanismo de parada temprana para evitar sobreentrenamiento.
        * clip_value (float): valor máximo permitido para el clipping de gradientes.
        * model_empty (torch.nn.Module): copia vacía del modelo para recargar los mejores pesos.
        * path (str): ruta donde se guardan los pesos del modelo durante el entrenamiento.

    Returns:
        * history (dict): diccionario con las pérdidas de entrenamiento, validación y prueba.
    """
    
    history = {"loss": [], "val_loss": [],"epoch":[]}
    optimizer = optimizer(model.parameters(), lr = lr)


    if gamma is not None:
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)

    for epoch in range(epochs):

        model.train()
        running_loss = 0.0

        for data in train_dataloader:
            inputs, targets = data

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()

            if clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)

            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        lr_scheduler.step()
        avg_loss = running_loss / len(train_dataloader.dataset)
        history['loss'].append(avg_loss)

        model.eval()
        running_vloss = 0.0

        with torch.no_grad():
            for vdata in val_dataloader:
                vinputs, vtargets = vdata

                voutputs = model(vinputs)
                vloss = criterion(voutputs.squeeze(), vtargets)

                running_vloss += vloss.item() * vinputs.size(0)
        
        avg_vloss = running_vloss/len(val_dataloader.dataset)
        history['val_loss'].append(avg_vloss)
        history['epoch'].append(epoch + 1)

        if early_stopper is not None:
            if early_stopper.early_stop(avg_vloss, model):
                print(f'Early stop en epoch {epoch+1}')
                break
        
        print(f'Epoch {epoch + 1} | Train Loss: {avg_loss:.8f} | Validation Loss: {avg_vloss:.8f}')
    
    model = model_empty
    model.load_state_dict(torch.load(path))

    model.eval()
    running_test_loss = 0.0

    with torch.no_grad():
        for tdata in test_dataloader:
            tinputs, ttargets = tdata

            toutputs = model(tinputs)
            
            tloss = criterion(toutputs.squeeze(), ttargets) 

            running_test_loss += tloss.item() * tinputs.size(0)
            
    avg_test_loss = running_test_loss / len(test_dataloader.dataset)
    history['test_loss'] = avg_test_loss
    print(f'Test Loss: {avg_test_loss}')

    return history


def plot_history(history:dict, plot_list=[], scale="linear", palette = None):
    i = 0
    fig = plt.figure(figsize=(14, 7))
    plt.xlabel("Epoch")
    for plot in plot_list:
        plt.plot(history["epoch"], history[plot], label=plot, color = palette[i])
        i += 1
    plt.yscale(scale)
    plt.legend(fontsize=30)
    plt.show()

def visualize_true_vs_pred_regression(y_true,y_pred, model = ''):

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    fig= plt.figure(figsize=(14,7))
    p1 = max(max(y_pred), max(y_true))
    p2 = min(min(y_pred), min(y_true))
    plt.plot([p1, p2], [p1, p2], 'g-')
    plt.scatter(y_true,y_pred)
    plt.xlabel(f"Real value",fontsize=20)
    plt.ylabel(f"Prediction",fontsize=20)
    plt.title(f"{model} Real vs Predicted (Test). MAE={mae:.2e}, MSE={mse:.2e}, R2={r2:.2e}",fontdict={"fontsize":20})
    plt.show()

def grafico_pasos(pasos, mses):
    plt.figure(figsize=(12, 6))
    plt.plot(pasos, mses, marker='o', linestyle='-', color='blue', alpha = 0.7)
    plt.xlabel("Pasos Utilizados")
    plt.ylabel("MSE Loss") 
    plt.title("MSE en función de los pasos utlizados")
    plt.gca().invert_xaxis()
    plt.xticks(pasos)
    plt.show()

def cartera_simple(df_precios, risk_free_rate=0.0):
    returns = df_precios.pct_change().dropna()
    cov_anual = returns.cov() * 252
    resultados = []
    combinaciones = list(itertools.combinations(df_precios.columns, 3))
    for activos in combinaciones:
        pesos = np.array([1/3, 1/3, 1/3])
        sub_returns = returns[list(activos)]
        port_daily_returns = sub_returns @ pesos
        rent_diaria_media = port_daily_returns.mean()
        rentabilidad_anualizada = (1 + rent_diaria_media) ** 252 - 1
        sub_cov = cov_anual.loc[list(activos), list(activos)]
        volatilidad_anualizada = np.sqrt(np.dot(pesos, np.dot(sub_cov, pesos.T)))
        rentabilidad_acumulada = (1 + port_daily_returns).prod() - 1
        sharpe = (rentabilidad_anualizada - risk_free_rate) / volatilidad_anualizada if volatilidad_anualizada > 0 else np.nan
        cumulative = (1 + port_daily_returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = drawdown.min()
        resultados.append({
            'cartera': activos,
            'rentabilidad_anualizada': float(rentabilidad_anualizada),
            'volatilidad_anualizada': float(volatilidad_anualizada),
            'rentabilidad_acumulada': float(rentabilidad_acumulada),
            'sharpe': float(sharpe),
            'drawdown_max': float(max_dd)})
    df_resultados = pd.DataFrame(resultados)
    df_resultados = df_resultados.sort_values(
        by=['rentabilidad_anualizada', 'volatilidad_anualizada'],
        ascending=[False, False]).reset_index(drop=True)
    return df_resultados

def evaluar_carteras_desde_top(df_precios, df_top, risk_free_rate=0.01):
    if 'cartera' not in df_top.columns:
        raise ValueError("df_top debe contener la columna 'cartera' con las carteras a evaluar")
    carteras = []
    for v in df_top['cartera']:
        if isinstance(v, (list, tuple, set)):
            carteras.append(tuple(map(str, v)))
        elif isinstance(v, str):
            parsed = None
            try:
                parsed = ast.literal_eval(v)
            except Exception:
                parsed = None
            if isinstance(parsed, (list, tuple, set)):
                carteras.append(tuple(map(str, parsed)))
            else:
                parts = [p.strip() for p in v.split(',') if p.strip()]
                if len(parts) > 1:
                    carteras.append(tuple(map(str, parts)))
                else:
                    carteras.append((str(v),))
        else:
            try:
                carteras.append(tuple(map(str, list(v))))
            except Exception:
                carteras.append((str(v),))
    available = set(df_precios.columns.astype(str))
    carteras_validas = []
    carteras_descartadas = []
    for c in carteras:
        if all(sym in available for sym in c):
            carteras_validas.append(c)
        else:
            carteras_descartadas.append(c)
    if len(carteras_validas) == 0:
        raise ValueError("Ninguna cartera válida: comprueba que los símbolos de 'cartera' estén en df_precios.columns")
    returns = df_precios.pct_change().dropna()
    cov_anual = returns.cov() * 252
    corr_matrix = returns.corr()
    resultados = []
    for activos in carteras_validas:
        n = len(activos)
        pesos = np.array([1.0/n] * n)
        sub_returns = returns[list(activos)]
        port_daily_returns = sub_returns @ pesos
        rent_diaria_media = port_daily_returns.mean()
        rentabilidad_anualizada = (1 + rent_diaria_media) ** 252 - 1
        sub_cov = cov_anual.loc[list(activos), list(activos)]
        volatilidad_anualizada = np.sqrt(np.dot(pesos, np.dot(sub_cov, pesos.T)))
        rentabilidad_acumulada = (1 + port_daily_returns).prod() - 1
        sharpe = (rentabilidad_anualizada - risk_free_rate) / volatilidad_anualizada if volatilidad_anualizada > 0 else np.nan
        cumulative = (1 + port_daily_returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = drawdown.min()
        if n == 3:
            c_ab = corr_matrix.loc[activos[0], activos[1]]
            c_ac = corr_matrix.loc[activos[0], activos[2]]
            c_bc = corr_matrix.loc[activos[1], activos[2]]
            corrs = [c_ab, c_ac, c_bc]
            corr_alta = max(corrs)
            idx_alta = corrs.index(corr_alta)
            pares = [(activos[0], activos[1]), (activos[0], activos[2]), (activos[1], activos[2])]
            par_alta = pares[idx_alta]
            tercero = list(set(activos) - set(par_alta))[0]
            corr_tercero_1 = corr_matrix.loc[tercero, par_alta[0]]
            corr_tercero_2 = corr_matrix.loc[tercero, par_alta[1]]
            corr_media_tercero = float(np.mean([corr_tercero_1, corr_tercero_2]))
            score_corr = float(corr_alta - corr_media_tercero)
        else:
            corr_alta = np.nan
            corr_media_tercero = np.nan
            score_corr = np.nan
        resultados.append({
            'cartera': activos,
            'rentabilidad_anualizada': float(rentabilidad_anualizada),
            'volatilidad_anualizada': float(volatilidad_anualizada),
            'rentabilidad_acumulada': float(rentabilidad_acumulada),
            'sharpe': float(sharpe) if not np.isnan(sharpe) else np.nan,
            'drawdown_max': float(max_dd),
            'corr_mas_alta': float(corr_alta) if not np.isnan(corr_alta) else np.nan,
            'corr_media_tercero': float(corr_media_tercero) if not np.isnan(corr_media_tercero) else np.nan,
            'score_corr': float(score_corr) if not np.isnan(score_corr) else np.nan})
    df_resultados = pd.DataFrame(resultados)
    df_resultados = df_resultados.sort_values(by=['score_corr', 'rentabilidad_anualizada'], ascending=[False, False]).reset_index(drop=True)
    print(f"Carteras evaluadas: {len(carteras_validas)}  |  descartadas por símbolos faltantes: {len(carteras_descartadas)}")
    if carteras_descartadas:
        print("Ejemplos descartados:", carteras_descartadas[:5])
    return df_resultados
def test_estacionario(residuo):
    residuo=residuo.dropna()
    adf_test = adfuller(residuo, autolag='AIC')
    p_adf = adf_test[1]
    kpss_test = kpss(residuo, nlags="auto")
    p_kpss = kpss_test[1]
    arch_test = het_arch(residuo)
    p_arch = arch_test[1]
    if p_adf < 0.05 and p_kpss > 0.05:
        estacionario = "La serie es ESTACIONARIA"
    else:
        estacionario = "La serie NO es estacionaria"

    if p_arch < 0.05:
        heterocedasticidad = "Existe HETEROCEDASTICIDAD (varianza no constante)"
    else:
        heterocedasticidad = "No hay heterocedasticidad (varianza constante)"
    resultados = {
        'ADF_pvalue': p_adf,
        'KPSS_pvalue': p_kpss,
        'ARCH_pvalue': p_arch,
        'Conclusión_estacionariedad': estacionario,
        'Conclusión_heterocedasticidad': heterocedasticidad
    }
    return resultados