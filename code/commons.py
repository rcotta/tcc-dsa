import random as rnd
from sklearn import model_selection
from sklearn import linear_model
from sklearn import ensemble
from sklearn import naive_bayes
from sklearn import metrics
from sklearn import neural_network
import math
import pandas as pd
import time



# geração de dados
CONVERSION_RATE_OVER_PROPENSION = 0.9 # chances de produto com propensão mínima esperada ser selecionado
RANDOM_CONVERSION_RATE = 0.01 # linhas onde um produto com propensão > 0 é selecionado independente da propensão mínima



CATEGORICAL_CUSTOMER = {
	'gender': ['masc', 'fem', 'others', 'not_informed'],
	'region': ['north', 'northeast', 'central_west', 'southeast', 'south'],
	'age': ['teenager', 'young_adult', 'adult', 'senior', 'elder'], 
	'income': ['low', 'middle', 'upper_middle', 'high', 'very_high'],
	'primary_channel': ['desktop', 'mobile', 'telesales'],
	'traffic_origin': ['direct', 'organic_search', 'paid_search', 'advertising', 'social_media'],
	'previous_complaints': ['yes', 'no'],
	'recent_purchase': ['yes', 'no'],
	'cluster_parents': ['yes', 'no'],
	'cluster_fitness': ['yes', 'no'],
	'cluster_electronics': ['yes', 'no'],
	'cluster_foodsdrinks': ['yes', 'no'],
	'cluster_toys': ['yes', 'no'],
	'cluster_books': ['yes', 'no'],
	'cluster_games': ['yes', 'no'],
	'cluster_audio': ['yes', 'no']
}

DISTRIB_CATEGORICAL_CUSTOMER = {
	'gender': [0.38, 0.42, 0.05, 0.15],						# ['masc', 'fem', 'others', 'non_informed', 'unknown'],
	'region': [0.13, 0.22, 0.14, 0.41, 0.10],				# ['north', 'northeast', 'central_west', 'southeast', 'south']
	'age': [0.10, 0.25, 0.45, 0.15, 0.05],					# ['teenager', 'young_adult', 'adult', 'senior', 'elder']
	'income': [0.06, 0.47, 0.11, 0.08, 0.28],				# ['low', 'middle', 'upper_middle', 'high', 'very_high', 'unknown'],
	'primary_channel': [1.00, 1.00, 1.00], 					# ['desktop', 'mobile', 'telesales'],
	'traffic_origin': [0.15, 0.52, 0.05, 0.10, 0.18],		# ['direct', 'organic_search', 'paid_search', 'advertising', 'social_media'],
	'previous_complaints': [1.00, 1.00], 					# ['yes', 'no'],
	'recent_purchase': [1.00, 1.00], 						# ['yes', 'no'],
	'cluster_parents': [1.00, 1.00], 						# ['yes', 'no'],
	'cluster_fitness': [1.00, 1.00], 						# ['yes', 'no'],
	'cluster_electronics': [1.00, 1.00], 					# ['yes', 'no'],
	'cluster_foodsdrinks': [1.00, 1.00], 					# ['yes', 'no'],
	'cluster_toys': [1.00, 1.00], 							# ['yes', 'no'],
	'cluster_books': [1.00, 1.00], 							# ['yes', 'no'],
	'cluster_games': [1.00, 1.00], 							# ['yes', 'no'],
	'cluster_audio': [1.00, 1.00] 							# ['yes', 'no']
}


# constantes dos recomendadores de ML
REGULAR = 'regular'
ALTERNATE = 'alternate'
ALGORITHMS = ['naive_bayes'] # ['neural_network', 'naive_bayes', 'random_forest', 'logistic_regression']

# localização e nomes dos arquivos
DATA_DIR = "data/"
VISITS_FILENAME = f"{DATA_DIR}visits.csv"
VISITS_NORMALIZED_FILENAME = f"{DATA_DIR}visits_normalized.csv"
PRODUCTS_FILENAME = f"{DATA_DIR}products.json"
PRODUCTS_STATS_FILENAME = f"{DATA_DIR}products_stats.csv"

# constantes para logging
EVENT_SWAP_MODEL = 'swap_model'
EVENT_KEEP_MODEL = 'keep_model'
EVENT_TRAIN_ALTERNATE_MODEL = 'train_alternate_model'

# constantes para índices dos dicionários
SELECTED_PRODUCT = 'selected_product'
PERIOD_COUNT = 'period_count'
PERIOD = 'period'
VISIT_ID = 'visit_id'



###### OK #########
# seleciona aleatoriamente uma opção com base em uma distribuição de probabilidade fornecida
# retorna o índice da opção selecionada
def get_from_distrib(distribution = []):

	if (type(distribution) == dict): distribution = list(distribution.values())

    # valor de parada
	t = rnd.random() * sum(distribution)

	# cursor que indicará quando atingimos a condição de parada
	current = 0.0

	for i in range(len(distribution)):
		current += distribution[i]
		if current >= t:
			return i

	return len(distribution) - 1


# retorna um array com num_items elementos aleatorios
# o elemento de índice max_value_index será o de maior valor
# elementos nas posições 0 a max_value_index encontram-se em ordem crescente, e em ordem decrescente após
# a soma dos elementos do array é ~1.0 se normalized=False
# se normalized=True, o menor valor será zero, e o maior será 1.0
def random_product_values(num_items, max_value_index=None, normalized=False, max_variation_pct=1):

	result = []

	if max_value_index == None: max_value_index = rnd.randint(0, num_items - 1)

	for i in range(num_items):
		inc = 1 if i <= max_value_index else -1
		if i == 0: result.append(rnd.random())
		else: result.append(result[i - 1] + (result[i - 1] * inc * rnd.random() * max_variation_pct))

	if not normalized:
		result = [p / sum(result) for p in result]
	else:
		vmin = min(result)
		vmax = max(result)
		result = [(p - vmin) / (vmax - vmin) for p in result]

	return result


def compute_propension(period, product, customer):
	
	# se produto está desabilitado, propensão é zero
	if not product['enabled']: return 0.0

	# propension = [para cada atributo] f(produto.atributo.peso x produto.atributo.likelihood[valor do atributo no cliente])
	# somatório peso do atributo no produto x likelihood do atributo para o valor do atributo no cliente
	# ex: dado cliente com região==N; peso região x peso da região norte no produto
	propension = sum([product[t]['weight'] * product[t]['likelihood'][customer[t]] for t in list(customer.keys())])
	propension += product['period']['weight'] * product['period']['likelihood'][period % 12]
	return propension


def get_expanded_column_names(var_name, categories):
	return [f"{var_name}_{category}" for category in categories]


def expand_column(var_name, values, categories):

	num_cols = len(categories)
	expanded_column_names = get_expanded_column_names(var_name, categories)
	columns = [[] for i in range(num_cols)]

	for value in values:
		for i in range(num_cols):
			columns[i].append(0 if i != value else 1)

	result = {}
	for i in range(num_cols):
		result[expanded_column_names[i]] = columns[i]

	return result



def stringify_record(record, decimals=4, separator=",", force_int=False):
    
	result = []

	for k, v in record.items():

		if isinstance(v, dict):
			nested = stringify_record(v, decimals, separator)
			result.append(nested)

		elif isinstance(v, float):
			if force_int: result.append(str(int(v)))
			else: result.append(("{v:." + str(decimals) + "f}").format(v=v))

		else:
			result.append(str(v))

	return separator.join(result)


def stringify_headers(record, decimals=4, separator=","):
    result = []

    for k, v in record.items():
        if isinstance(v, dict):
            nested = stringify_headers(v, decimals, separator)
            result.append(nested)
        else:
            result.append(str(k))

    return separator.join(result)



# seleciona um produto:
# - se random() < RANDOM_CONVERSION_RATE, seleciona um produto aleatoriamente (outliers)
# - caso contrário, seleciona um dos produtos acima do quantil
# - ou retorna 0 (nenhum produto selecionado)
def pick_product(product_ids: list, quantiles: list, propensions:list) -> int:


	# verifica se teremos uma conversão aleatória
	if rnd.random() <= RANDOM_CONVERSION_RATE:
		potential_products = [product_ids[i] for i in range(len(product_ids)) if propensions[i] > 0.0]


	else:
		# seleciona produtos cujo propensão está acima do quantil definido para cada produto
		potential_products = [product_ids[i] for i in range(len(product_ids)) if propensions[i] > quantiles[i]]
		# sorteia quais produtos serão mantidos
		potential_products = [p for p in potential_products if rnd.random() <= CONVERSION_RATE_OVER_PROPENSION]
		
	
	# se não temos produto selecionado, selecionar produto 0 (nenhum produto selecionado)
	# se temos mais de um produto selecionado, selecionar apenas 1
	if len(potential_products) == 0:
		selected_product = 0
	else:
		selected_product = rnd.choice(potential_products)
	
	return selected_product	


def train_model(X_df: pd.DataFrame, y_df: pd.DataFrame, algorithm: str):

	lm = None

	if algorithm == 'logistic_regression':
		lm = linear_model.LogisticRegression(multi_class='ovr', solver='liblinear')
	elif algorithm == 'random_forest':
		lm = ensemble.RandomForestClassifier(max_depth=10)
	elif algorithm == 'naive_bayes':
		lm = naive_bayes.MultinomialNB(force_alpha=True)
	elif algorithm == 'neural_network':
		lm = neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(150, 150, 150), random_state=1, max_iter=5000)
	else:
		raise Exception("Please provide one of values logistic_regression, random_forest, naive_bayes for algorithm")
	
	lm.fit(X_df, y_df.values.ravel())

	return lm


__TIMING = {}
def time_contab(event: str, calculate: bool = False):

	if not calculate:
		if not event in __TIMING: __TIMING[event] = []
		__TIMING[event].append(time.time())

	else:
		if not event in __TIMING: return 0
		total, i = 0
		while i < len(__TIMING[event]) - 2:
			total += __TIMING[event][i+1] - __TIMING[event][i]
			i += 2
		return total

