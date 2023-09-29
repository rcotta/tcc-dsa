import pandas as pd
import random as rnd
from datetime import datetime
from commons import *

from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics

import re

import os

# P_TRAINNING_CICLE_DURATION = [2500, 5000, 10000]
P_TRAINNING_CICLE_DURATION = [20000, 40000, 60000, 80000, 100000]
P_ABTEST_DURATION_PCT = [0.4] # duração do teste A/B é P_ABTEST_DURATION_PCT * P_TRAINNING_CICLE_DURATION visitas
P_ABTEST_ALTERNATE_MODEL_PCT = [0.15, 0.30, 0.60] # durante teste A/B, % de visitas que serão utilizadas para testar modelo alternativo


def log_event_str(algorithm: str, current_row: int, max_row: int, event_type: str, info: object) -> str:
    return(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t{current_row}\t{max_row}\t{algorithm}\t{event_type}\t{info}")


# recupera os diferentes arquivos normalizados que serão processados
filenames = [f"{DATA_DIR}{filename}" for filename in os.listdir(DATA_DIR) if "visits_normalized" in filename]
maskings = [int(re.search("([0-9]+)", s).groups()[0]) for s in filenames]
fileinfos = dict(zip(filenames, maskings)) # dict com {<filemane>: <params>}

log_file = open(f"{DATA_DIR}/recommendations.log", "w", 1)
results_file_name = f"{DATA_DIR}/results.csv"

def sub_df(df: pd.DataFrame, min_row: int, max_row: int, x_cols: list, y_cols: list) -> tuple[pd.DataFrame, pd.DataFrame]:

    tmp = df.loc[min_row:max_row]
    x_df = tmp.loc[:, x_cols]
    y_df = tmp.loc[:, y_cols]

    return (x_df, y_df)


def add_info_columns(df: pd.DataFrame, predictions, algorithm, model_type, model_version, cycle_duration, ab_duration_pct, masking, alternate_model_pct):
    
    df['predicted'] = predictions
    df['algorithm'] = algorithm
    df['model_type'] = model_type
    df['model_version'] = model_version
    df['cycle_duration'] = cycle_duration
    df['ab_duration_pct'] = ab_duration_pct
    df['alternate_model_pct'] = alternate_model_pct
    df['masking'] = masking



proc_count = 0
write_headers = True

for filename, masking in fileinfos.items():

    # lendo os arquivos de visitas
    visits_df = pd.read_csv(filename)

    # colunas que podem ser recuperadas para treinamento
    X_columns = [column_name for column_name in visits_df.columns if column_name not in [VISIT_ID] and "prop_" not in column_name and column_name != SELECTED_PRODUCT and column_name != PERIOD_COUNT]
    y_column = SELECTED_PRODUCT

    for duration in P_TRAINNING_CICLE_DURATION:

        for test_duration_pct in P_ABTEST_DURATION_PCT:

            for alternate_model_pct in P_ABTEST_ALTERNATE_MODEL_PCT:

                proc_count += 1
                proc_total = len(fileinfos) * len(P_TRAINNING_CICLE_DURATION) * len(P_ABTEST_DURATION_PCT) * len(P_ABTEST_ALTERNATE_MODEL_PCT)
                log_file.write("\n" + log_event_str(algorithm='-', current_row=0, max_row=0, event_type='CICLE_START', info=[filename, duration, test_duration_pct, alternate_model_pct, f"{proc_count}/{proc_total}"]))

                regular_versions = {}
                alternate_versions = {}
                for algorithm in ALGORITHMS: 
                    regular_versions[algorithm] = 1
                    alternate_versions[algorithm] = 1

                # duração, em número de visitas, do período de recomendação com modelo principal
                # e do período de teste A/B
                regular_duration = int(duration * (1 - test_duration_pct))
                test_duration = duration - regular_duration

                # criando os arquivos de resultados
        
                # inicialização
                current_row = max(P_TRAINNING_CICLE_DURATION)

                # models[algorithm][current|alternate] = {model->Obj, total->int, correct->int}
                models = {}

                # recuperar os últimos TRAINNING_SET_SIZE y e X, com predito != 0
                X_train, y_train = sub_df(df=visits_df[visits_df[SELECTED_PRODUCT] != 0], min_row=current_row - regular_duration, max_row=current_row-1, x_cols=X_columns, y_cols=y_column)
                for algorithm in ALGORITHMS: models[algorithm] = train_model(algorithm=algorithm, X_df=X_train, y_df=y_train)
                
                mode_regular = True
                
                while current_row < len(visits_df):

                    if mode_regular:

                        # período regular, modelo principal
                        max_row = current_row + regular_duration - 1
                        
                        # corner case: se não houver espaço para finalizar um teste A/B
                        # após esse período regular, não vamos executar um último teste A/B
                        if (max_row + test_duration - 1) >= len(visits_df): max_row = len(visits_df)

                        X_rows, y_rows = sub_df(df=visits_df[visits_df[SELECTED_PRODUCT] != 0], min_row=current_row, max_row=max_row, x_cols=X_columns, y_cols=y_column)

                        # condição de fim de processamento, eventualmente não temos mais registros a processar
                        if (len(X_rows) == 0):
                            current_row = max_row
                            continue

                        # visit_id,period_count,period,propension_predicted,predicted,actual
                        for algorithm in ALGORITHMS:
                            predictions = models[algorithm].predict(X_rows)
                            results = visits_df.loc[X_rows.index, ['visit_id', 'period', 'selected_product']]

                            add_info_columns(df=results, predictions=predictions, algorithm=algorithm, model_type=REGULAR, model_version=regular_versions[algorithm],
                                            cycle_duration=duration, ab_duration_pct=test_duration_pct, masking=masking, alternate_model_pct=alternate_model_pct)

                            results.to_csv(results_file_name, mode='a', index=False, header=write_headers)

                            hit_ratio_regular = (len(results[results['selected_product'] == results['predicted']]) / len(results))
                            # print(f"reg\tregular\t{algorithm}\t{round(hit_ratio_regular, 2)}")

                            write_headers = False
                        

                    else:

                        # período de teste A/B, concorrer modelos
                        max_row = current_row + test_duration - 1
                    
                        conversions_df = visits_df[visits_df[SELECTED_PRODUCT] != 0]
                        X_rows, y_rows = sub_df(df=conversions_df, min_row=current_row, max_row=max_row, x_cols=X_columns, y_cols=y_column)

                        num_test_items = int(len(X_rows) * test_duration_pct * alternate_model_pct)

                        # set de treinamento do teste A/B
                        test_models = {}
                        X_train, y_train = sub_df(visits_df[visits_df[SELECTED_PRODUCT] != 0], current_row - regular_duration, current_row - 1, X_columns, y_column)
                        for algorithm in ALGORITHMS:
                            test_models[algorithm] = train_model(algorithm=algorithm, X_df=X_train, y_df=y_train)
                            alternate_versions[algorithm] += 1

                        # x e y do teste A/B
                        X_rows_test = X_rows.sample(num_test_items)
                        y_rows_test = y_rows.loc[X_rows_test.index]

                        # x e y do modelo regular
                        X_rows = X_rows[~X_rows.isin(X_rows_test)].dropna()
                        y_rows = y_rows[X_rows.index]

                        # logando início de teste a/b
                        log_file.write("\n" + log_event_str(algorithm='-', current_row=current_row, max_row=max_row, event_type='AB_START', info=[f"duration={duration}", f"test_duration_pct={test_duration_pct}", f"alternate_model_pct={alternate_model_pct}", f"num_test_items={num_test_items}"]))

                        # visit_id,period_count,period,propension_predicted,predicted,actual
                        for algorithm in ALGORITHMS:
                            
                            hit_ratio_regular = -1
                            hit_ratio_alternate = -1

                            predictions = models[algorithm].predict(X_rows)
                            results = visits_df.loc[X_rows.index, ['visit_id', 'period', 'selected_product']]

                            add_info_columns(df=results, predictions=predictions, algorithm=algorithm, model_type=REGULAR, model_version=regular_versions[algorithm],
                                            cycle_duration=duration, ab_duration_pct=test_duration_pct, masking=masking, alternate_model_pct=alternate_model_pct)

                            hit_ratio_regular = (len(results[results['selected_product'] == results['predicted']]) / len(results))

                            predictions_test = test_models[algorithm].predict(X_rows_test)
                            results_test = visits_df.loc[X_rows_test.index, ['visit_id', 'period', 'selected_product']]


                            add_info_columns(df=results_test, predictions=predictions_test, algorithm=algorithm, model_type=ALTERNATE, model_version=alternate_versions[algorithm],
                                            cycle_duration=duration, ab_duration_pct=test_duration_pct, masking=masking, alternate_model_pct=alternate_model_pct)

                            hit_ratio_alternate = (len(results_test[results_test['selected_product'] == results_test['predicted']]) / len(results_test))

                            # print(f"a/b\tregular\t{algorithm}\t{round(hit_ratio_regular, 2)}")
                            # print(f"a/b\talternate\t{algorithm}\t{round(hit_ratio_alternate, 2)}")

                            if (hit_ratio_alternate > hit_ratio_regular):
                                print(f"trocando modelo {algorithm}")
                                models[algorithm] = test_models[algorithm]
                                regular_versions[algorithm] = alternate_versions[algorithm]

                            # logando fim de teste a/b
                            log_file.write("\n" + log_event_str(algorithm=algorithm, current_row=current_row, max_row=max_row, event_type='AB_FINISH', info=[f"hh_regular={round(hit_ratio_regular, 1)}", f"hh_alternate={round(hit_ratio_alternate, 1)}", (hit_ratio_alternate > hit_ratio_regular)]))


                            if len(results) > 0 and len(results_test) > 0:
                                results = pd.concat([results, results_test]).sort_index()
                            else:
                                results = results.sort_index()
                            
                            results.to_csv(results_file_name, mode='a', index=False, header=write_headers)


                    mode_regular = not mode_regular
                    current_row = max_row

