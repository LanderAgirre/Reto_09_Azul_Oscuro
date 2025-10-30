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

def visualize_true_vs_pred_regression(y_true,y_pred):

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
    plt.title(f"Real vs Predicted (Test). MAE={mae:.2e}, MSE={mse:.2e}, R2={r2:.2e}",fontdict={"fontsize":20})
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