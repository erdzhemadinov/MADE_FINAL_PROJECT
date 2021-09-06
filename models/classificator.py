import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

try:
    from model_troll import Model
except:
    pass

def binary_balance(data, target="trollolo", relation=0.5, 
                   output_size=None, shuffle=False, random_state=42, reset_index=True):
    """
    Возвращает датафрейм с заданным отношением класса 1 к общей выборке (только 2 класса (0, 1))
        data - датафейм или путь до csv файла
        target - название колонки с таргетом
        relation - требуемая относительная часть класса 1 в общей выборке, остальное занимает класс 0.
        output_size - выходной размер датафрейма
        shuffle - флаг отвечающий за перемешивание выходного датафрейма
        reset_index - обновление индексов
    """
    try:
        df = pd.read_csv(data)
    except:
        df = data
        
    if shuffle:
        df = df.sample(frac=1, random_state=random_state)

    if relation > 1 or relation < 0:
        return df
    
    counts = df[target].value_counts()
    sum_elements = counts[0] + counts[1]
    relation_class_1 = counts[1] / (sum_elements)
    
    if relation_class_1 == relation:
        return df
    
    if relation_class_1 < relation:
        delta = sum_elements - int(counts[1] / relation)
        index_drop = np.where(df[target]==0)[0][-delta:]
        counts[0] -= delta
    else:
        delta = sum_elements - int(counts[0] / (1 - relation))
        index_drop = np.where(df[target]==1)[0][-delta:]
        counts[1] -= delta
    
    if output_size is not None and sum_elements - delta > output_size:
        size_class_1 = int(output_size * relation)
        len_drop_class_1 = counts[1] - size_class_1
        len_drop_class_0 = counts[0] - (output_size - size_class_1)
        
        if relation_class_1 < relation:
            index_drop_class_1 = np.where(df[target]==1)[0][-len_drop_class_1:]
            index_drop_class_0 = np.where(df[target]==0)[0][-len_drop_class_0 - delta:]
        else:
            index_drop_class_1 = np.where(df[target]==1)[0][-len_drop_class_1 - delta:]
            index_drop_class_0 = np.where(df[target]==0)[0][-len_drop_class_0:]
        
        index_drop = np.hstack((index_drop_class_1,
                                index_drop_class_0,
                                ))
    
    new_df = df.drop(index=df.index[index_drop])
        
    if reset_index:
        return new_df.reset_index(drop=True)
    else:
        return new_df


class Classificator:
    def __init__(self, model_troll, classificator=None, scaler=None):
        
        self._model_troll = model_troll
        self.classificator = classificator
        self.scaler = scaler
        
    def fit(self, 
            data="datasets/troll_data_all.csv", preproc=True,
            save_models=True,
            target="trollolo", balance_rel=0.5,
            output_size=None, test_size=0.2,
            C=0.3, penalty="l2", solver="saga",
            ):
        
        self.classificator = LogisticRegression(C=C, 
                                                penalty=penalty, 
                                                solver=solver)
        
        # Выравнивание классов в заданной пропорции
        self.data = binary_balance(data, target, balance_rel, output_size)
        self.data_x_preproc = None
        self.target = target
        
        if preproc:
            self._preproc(save=True, path_to_save='emb_numpy/data_full.npy')
            self._train_test_split(test_size=test_size)
        else:
            data_x = np.load('emb_numpy/data_full.npy')
            self._train_test_split(data_x=data_x, test_size=test_size)
            
        self._scale()
        self.classificator.fit(self.x_train_scale, self.y_train)
        
        if save_models:
            self.save_models()
        
        self._solve_metrics()
        self._print_metrics()
        return self.metrics
        
    def predict_proba(self, features):
        return self.classificator.predict_proba(self.scaler.transform(features))[0]
    
    def save_models(self, 
                   classificator_filename = 'logreg_low_cls.pickle',
                   scaler_filename = 'scaler.pickle'):
        
        pickle.dump(self.classificator, open(classificator_filename, 'wb'))
        pickle.dump(self.scaler, open(scaler_filename, 'wb'))
    
    def _predict(self, data_x):
        return self.classificator.predict(data_x)
    
    def _preproc(self, save=True, path_to_save='data_full.npy'):
        
        print("Формирование вектора Х")
        tmp_text = []
        for i in tqdm(range(self.data.shape[0])):
            tmp_text.append(self._model_troll.create_features(self.data["question"][i], 
                                                              self.data["answer"][i])[0])
        self.data_x_preproc = np.array(tmp_text)
        
        if save:
            np.save(path_to_save, tmp_text)
            
    def _train_test_split(self,
                          data_x=None,
                          data_y=None,
                          test_size=0.05, 
                          shuffle=True, 
                          random_state=42):
        
        if data_x is None:
            data_x = self.data_x_preproc
        if data_y is None:
            data_y = self.data[self.target]
        
        self.x_train, self.x_test, self.y_train, self.y_test = \
            model_selection.train_test_split(data_x,
                                             data_y,
                                             test_size=test_size,
                                             shuffle=shuffle,
                                             random_state=random_state,
                                             )
    
    def _scale(self, ):
        self.scaler = StandardScaler()
        self.x_train_scale = self.scaler.fit_transform(self.x_train)
        self.x_test_scale = self.scaler.transform(self.x_test)
    


    def _solve_metrics(self, acc=6):
        self.metrics = {}
        self.y_train_pred = self._predict(self.x_train_scale)
        self.y_test_pred = self._predict(self.x_test_scale)
        
        self.metrics['roc-auc'] = {'train':round(roc_auc_score(self.y_train, self.y_train_pred), acc),
                                   'test':round(roc_auc_score(self.y_test, self.y_test_pred), acc)
                                  }
        
        self.metrics['accuracy'] = {'train':round(accuracy_score(self.y_train, self.y_train_pred), acc),
                                   'test':round(accuracy_score(self.y_test, self.y_test_pred), acc)
                                   }
        
        self.metrics['f1-score'] = {'train':round(f1_score(self.y_train, self.y_train_pred), acc),
                                   'test':round(f1_score(self.y_test, self.y_test_pred), acc)
                                   }
        
        self.metrics['precision'] = {'train':round(precision_score(self.y_train, self.y_train_pred), acc),
                                   'test':round(precision_score(self.y_test, self.y_test_pred), acc)
                                    }
        
        self.metrics['recall'] = {'train':round(recall_score(self.y_train, self.y_train_pred), acc),
                                   'test':round(recall_score(self.y_test, self.y_test_pred), acc)
                                 }
        return self.metrics
        
    def _print_metrics(self, ):
        print("\nROC-AUC metric:   train={}, test={}".format(self.metrics['roc-auc']['train'], self.metrics['roc-auc']['test']))
        print("Accuracy metric:  train={}, test={}".format(self.metrics['accuracy']['train'], self.metrics['accuracy']['test']))
        print("F1-score metric:  train={}, test={}".format(self.metrics['f1-score']['train'], self.metrics['f1-score']['test']))
        print("Precision metric: train={}, test={}".format(self.metrics['precision']['train'], self.metrics['precision']['test']))
        print("Recall metric:    train={}, test={}".format(self.metrics['recall']['train'], self.metrics['recall']['test']))
    
    
if __name__ == "__main__":
    model_troll = Model()
    classificator = Classificator(model_troll)
    classificator.fit(preproc=False, save_models=True)
    
    