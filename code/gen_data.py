import json
import random as rnd
import numpy as np
import pandas as pd
from commons import *
import json as json
import math

# constantes para geração de dados
NUM_PERIODS = 12 # total de períodos
MAX_VISITS_PER_PERIOD = 60000 # total de visitas por período
ACTIVE_PRODUCTS = 100 # total de produtos ativos ao longo do período
PRODUCTS_REPLACED_PER_PERIOD = 30 # total de produtos ativados e desativados por período
TOTAL_PRODUCTS = 150 # número total de produtos

HIDDEN_ATTRIBUTE_PCTS = [0.0] # % das linhas que terão um ou mais atributos (idade, genero) mascarados (ainda que sejam considerados nos cálculos das propensões)
HIDEABLE_ATTRIBUTES = ['gender', 'age', 'income', 'region']

# TARGET_CONVERSION_RATE = 0.002 # queremos conversões em (1 - TARGET_CONVERSION_RATE) % dos casos
TARGET_CONVERSION_RATE = 0.00045

assert ACTIVE_PRODUCTS <= TOTAL_PRODUCTS
assert PRODUCTS_REPLACED_PER_PERIOD < (TOTAL_PRODUCTS - ACTIVE_PRODUCTS)

# porcentagem das visitas que serão utilizadas para testar o modelo alternativo
# tamanho do conjunto de visitas utilizado para *avaliação* de um modelo treinado
# observar a tendência de termos AB_TEST_VISITS * AB_TEST_ITEMS_PCT visitas para validar um modelo alternativo,
# e que AB_TEST_VISITS deve ser MENOR que MAX_TRAINNING_SET_SIZE
MAX_TRAINNING_SET_SIZE = MAX_VISITS_PER_PERIOD # tamanho do conjunto máximo de visitas para treinar um novo modelo; manter preferencialmente múltiplo de 3
TRAINNING_SET_SIZE_VARIATIONS = 3 # variações no tamanho do trainning set


# cria um cliente aleatório, transformando
# todas as variáveis categóricas em variáveis dummy
# formato do customer: {<atributo>: <valor>}; ex: {'age': 1}
def create_random_customer():
	
	customer = {}

	for k, distrib in DISTRIB_CATEGORICAL_CUSTOMER.items():
	
		k_value = get_from_distrib(distribution=distrib)
		customer[k] = k_value

		for i in range(len(distrib)):
			column_name = f"{k}_{CATEGORICAL_CUSTOMER[k][i]}"

	return customer


def create_random_product(id, enabled=True):

	product = {'id': id, 'enabled': enabled}
	categoricals = CATEGORICAL_CUSTOMER

	weights = [rnd.random() for i in range(len(categoricals) + 1)] # +1 for the period
	weights = [w / sum(weights) for w in weights]

	keys = list(categoricals.keys())

	for i in range(len(keys)):
		k = keys[i]
		num_items = len(categoricals[k])
		product[k] = {'weight': weights[i], 'likelihood': random_product_values(num_items=num_items, max_value_index=rnd.randint(0, num_items - 1), normalized=True, max_variation_pct=(rnd.random() * 0.15))}

	product[PERIOD] = {'weight': weights.pop(), 'likelihood': random_product_values(num_items=NUM_PERIODS, max_value_index=rnd.randint(0, NUM_PERIODS - 1), normalized=True, max_variation_pct=(rnd.random() * 0.15 / NUM_PERIODS))}

	return product


# geração dos dados
visits_per_period = [int(MAX_VISITS_PER_PERIOD * (0.95 + (rnd.random() * 0.05))) for t in range(NUM_PERIODS)]

# cria os produtos aleatorios, salva os ids dos produtos
products_ids = [i + 1 for i in range(TOTAL_PRODUCTS)]
activate = rnd.sample(population=products_ids, k=ACTIVE_PRODUCTS)
products = [create_random_product(i+1, enabled=i+1 in activate) for i in range(TOTAL_PRODUCTS)]

# salva os dados dos produtos
first = True
products_file = open(PRODUCTS_FILENAME, "wt")
products_file.write(json.dumps(products))
products_file.close()

visits_normalized_file_name = f"{DATA_DIR}visits_normalized.csv"
visits_file = open(VISITS_FILENAME, "wt")
header_written = False
visit_id = 0


# cria arquivo com visitas e propensões sem normalização
print("Gerando arquivo de visitas")
for period, visits in enumerate(visits_per_period):

	for i in range(visits):

		visit_id += 1

		customer = create_random_customer()
		propensions_dict = {}
		# para todos os produtos, habilitados ou não
		# produtos não habilitados tem propensão zero
		for product in products:
			product_id = product['id']
			propensions_dict["prop_" + str(product_id)] = compute_propension(period=period % 12, customer=customer, product=product)

		stats = {'visit_id': visit_id, 'period_count': period, 'period': period % 12, 'customer': customer, 'propensions': propensions_dict}

		if not header_written:
			visits_file.write(stringify_headers(record=stats, decimals=4, separator=","))
			header_written = True

		visits_file.write("\n" + stringify_record(record=stats, decimals=4, separator=","))


	# troca PRODUCTS_REPLACED_PER_PERIOD_PCT (muda flag enabled)
	active_products = [product for product in products if product['enabled']]
	inactive_products = [product for product in products if not product['enabled']]
	for product in rnd.sample(active_products, k=PRODUCTS_REPLACED_PER_PERIOD): product['enabled'] = False
	for product in rnd.sample(inactive_products, k=PRODUCTS_REPLACED_PER_PERIOD): product['enabled'] = True

visits_file.close()


# cria arquivo com propensões normalizadas

# salva para arquivo secundário mínimo e máximo original (para repetir operação posteriormente)
visits_df = pd.read_csv(VISITS_FILENAME)
products_stats_file = open(PRODUCTS_STATS_FILENAME, "wt")
products_minmax = {}
header_written = False
products_stats = {}
quantiles = []

for p in products:

	stats = {'product_id' : p['id']}
	products_stats[p['id']] = stats

	column_name = f"prop_{p['id']}"
	stats['min'] = visits_df[column_name].min()
	stats['max'] = visits_df[column_name].max()

	# normaliza os valores no dataframe
	print(f"Normalizando dataframe para coluna {column_name}")
	visits_df[column_name] = (visits_df[column_name] - stats['min']) / (stats['max'] - stats['min'])

	# recupera o quantil para considerar produto como adquirido
	quantile = visits_df[column_name].quantile(1 - TARGET_CONVERSION_RATE)
	quantiles.append(quantile)
	stats['quantile'] = quantile	

	# escreve cabeçalho do arquivo
	if not header_written:
		products_stats_file.write(stringify_headers(stats, ","))
		header_written = True

	# escreve registro
	products_stats_file.write("\n" + stringify_record(stats))

products_stats_file.close()		


# expande as variáveis categóricas em variáveis dummy
print("Expandindo variáveis categóricas em variáveis dummy")
for k, v in CATEGORICAL_CUSTOMER.items():
	
	expanded_columns = expand_column(var_name=k, values=visits_df[k], categories=v)
	del visits_df[k]

	for dummy_name, column_data in expanded_columns.items():
		visits_df.insert(loc=len(visits_df.columns) - TOTAL_PRODUCTS, column=dummy_name, value=column_data)


# removendo - para algumas entradas - informações de gênero, renda, região e idade  simulando
# situações onde essas informações são desconhecidas para os clientes

hideable_attrs = HIDEABLE_ATTRIBUTES
hideable_attrs_columns = {}
for attr in hideable_attrs: hideable_attrs_columns[attr] = get_expanded_column_names(attr, CATEGORICAL_CUSTOMER[attr])

# insere as colunas utilizadas para marcar as linhas que tiveram atributos escondidos
for attr in hideable_attrs: visits_df.insert(loc=len(visits_df.columns) - TOTAL_PRODUCTS, column=f"unknown_{attr}", value=0)

# seleciona os produtos com base nas propensões
print("Selecionando produtos com base nas propensoes")

purchased = []
for i in range(len(visits_df)):
	# propensões estão nas últimas TOTAL_PRODUCTS colunas
	propensions = list(visits_df.loc[i][-TOTAL_PRODUCTS:])
	# escolhe um produto com base nas propensões
	purchased.append(pick_product(product_ids=products_ids, quantiles=quantiles, propensions=propensions))

visits_df.insert(loc=len(visits_df.columns), column='selected_product', value=purchased)

# ajusta os tipos das colunas do dataframe
column_types = {}
for column_name in visits_df.columns: column_types[column_name] = 'float' if ('prop_' in column_name or 'quantile' == column_name) else 'int'

print("Ajustando os tipos das colunas")
visits_df = visits_df.astype(column_types)

# vamos salvar duas versões do arquivo de visitas normalizado, uma sem atributos escondidos,
# outra com atributos escondidos
# print("Salvando arquivo sem atributos escondidos ...")
# visits_df.to_csv(VISITS_NORMALIZED_FILENAME, index=False, float_format='%.4f')

# escondendo atributos: x = 1
for hidden_attributes_pct in HIDDEN_ATTRIBUTE_PCTS:

	# esconde um ou mais atributos de VISITS_TO_HIDE_PCT% itens
	items_to_hide = round(len(visits_df) * hidden_attributes_pct)
	tmp_df = visits_df.copy(deep=True)
	
	for attr, columns in hideable_attrs_columns.items():
		print (f"Escondendo {items_to_hide}/{len(visits_df)} {attr}")
		modify_df = tmp_df.sample(items_to_hide)
		modify_df.loc[:, columns] = 0 # hiding attribute values
		modify_df.loc[:, f"unknown_{attr}"] = 1 # marking how as hidden attributes

		# atualizando o dataframe
		tmp_df.update(modify_df)

	tmp_df.to_csv(VISITS_NORMALIZED_FILENAME.replace(".csv", f"_masked-{round(hidden_attributes_pct * 100)}.csv"), index=False, float_format='%.4f')

# salva o arquivo normalizado
print(visits_df['selected_product'].value_counts())




	
	